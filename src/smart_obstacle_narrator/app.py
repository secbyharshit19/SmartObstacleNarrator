"""Command-line desktop app. Run from the extracted project root."""
import argparse
from dataclasses import asdict
from pathlib import Path
import queue
import time
import sys
from .config import load_config, read_yaml
from .pipeline import Pipeline
from .speech import SpeechWorker, PyttsxBackend, SilentBackend
from .logging_utils import EventLog
from .types import WarningEvent

def parser():
    p = argparse.ArgumentParser(description='Experimental Smart Obstacle Narrator')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--demo', action='store_true', help='Synthetic detections, NOT a trained model')
    mode.add_argument('--baseline', action='store_true', help='COCO baseline; only person/chair/bicycle/vehicle')
    p.add_argument('--weights', default=None)
    p.add_argument('--source', default='0', help='Camera index or video path')
    p.add_argument('--camera-id', default=None, help='Calibration identity; video defaults to its path')
    p.add_argument('--config', default='configs/app.yaml')
    p.add_argument('--calibration', default='configs/camera.yaml')
    p.add_argument('--device', default=None)
    p.add_argument('--no-audio', action='store_true', help='Simulated speech feedback; no audible output')
    p.add_argument('--headless', action='store_true', help='No window; use Ctrl+C to exit')
    p.add_argument('--audio-first', action='store_true', help='No window, with startup speech')
    p.add_argument('--seconds', type=float, default=0, help='Stop after N seconds; 0 means until quit/EOF')
    p.add_argument('--log', default=None)
    p.add_argument('--record', default=None, help='Explicitly save annotated video (.mp4)')
    p.add_argument('--snapshot', default=None, help='Save final annotated frame')
    return p

def run(args):
    import cv2
    import numpy as np
    from .display import render
    from .demo import demo_detections
    cfg, calibration = load_config(args.config), read_yaml(args.calibration)
    if args.device:
        cfg['device'] = args.device
    mode = 'SYNTHETIC DEMO' if args.demo else 'COCO BASELINE (4 classes)' if args.baseline else 'CUSTOM MODEL'
    log_path = args.log or f'results/runs/{time.strftime("%Y%m%d-%H%M%S")}-{time.time_ns()%1000000}.jsonl'
    log = EventLog(log_path)
    capture = video = writer = speaker = None
    last_canvas = None
    detector = None
    try:
        if not args.demo:
            from .detector import Detector
            weights = args.weights or ('yolo11n.pt' if args.baseline else 'models/v1/best_v1.pt')
            detector = Detector(weights, cfg, args.baseline)
            # Warm-up is excluded from measured frame timing.
            detector.detect(np.zeros((480,640,3), dtype=np.uint8))
            if args.source.isdecimal():
                from .capture import LatestCapture
                capture = LatestCapture(int(args.source), cfg['capture_width'], cfg['capture_height'])
            else:
                video = cv2.VideoCapture(args.source)
                if not video.isOpened():
                    raise RuntimeError(f'Cannot open video: {args.source}')
        pipeline = Pipeline(cfg, calibration, args.camera_id or args.source)
        backend = SilentBackend if args.no_audio else lambda: PyttsxBackend(cfg['speech_rate'])
        speaker = SpeechWorker(cfg['speech_gap'], cfg['continuity_gap'], backend)
        speaker.ready.wait(timeout=5)
        if speaker.status == 'error':
            print(f'AUDIO UNAVAILABLE: {speaker.error}', file=sys.stderr)
        log.write('session', mode=mode, audible=not args.no_audio, source=args.source,
                  config=cfg, calibration=calibration, device=detector.device if detector else 'simulation')
        print(f'{mode}. Log: {log_path}. Q quits the window; Ctrl+C quits headless mode.')
        start = time.monotonic()
        if args.audio_first and not args.no_audio:
            now = time.monotonic()
            e = WarningEvent('startup', -1, 'Experimental obstacle narrator ready. Use your usual mobility aid.',
                             100, now, now+4, ('ahead','unknown'))
            speaker.update({-1:(e.signature,now)}, e)
            # Startup is before capture processing; give worker a chance to claim event.
            time.sleep(.1)
        sequence, count = 0, 0
        tracks, last_warning, paused, muted = [], '', False, False
        previous_processed = None
        next_video_time = start
        video_fps = (video.get(cv2.CAP_PROP_FPS) or 25) if video else 25
        if not 0 < video_fps <= 240:
            video_fps = 25
        while not args.seconds or time.monotonic()-start < args.seconds:
            now = time.monotonic()
            while True:
                try:
                    feedback = speaker.feedback.get_nowait()
                except queue.Empty:
                    break
                payload = asdict(feedback)
                payload['feedback_kind'] = payload.pop('kind')
                log.write('speech_'+feedback.kind, **payload)
                if feedback.kind == 'started':
                    pipeline.acknowledge(feedback.event, feedback.timestamp)
                    last_warning = feedback.event.text
                    print(last_warning)
            if paused:
                speaker.update({}, None)
                if last_canvas is not None:
                    cv2.imshow('Smart Obstacle Narrator', last_canvas)
                key = cv2.waitKey(30) & 0xff
                if key == ord('q'):
                    break
                if key == ord('p'):
                    paused = False
                    pipeline.reset()
                    next_video_time = time.monotonic()
                continue
            if args.demo:
                frame = np.full((480,640,3), (44,40,34), dtype=np.uint8)
                captured = now
                detections = demo_detections(now-start)
                sequence += 1
                time.sleep(.03)
            elif capture:
                if capture.error:
                    raise RuntimeError(capture.error)
                packet = capture.read(sequence)
                if packet is None:
                    if previous_processed is not None and now-previous_processed > 2:
                        raise RuntimeError('No fresh camera frames for two seconds')
                    if previous_processed is None and now-start > 5:
                        raise RuntimeError('Camera opened but returned no frames')
                    time.sleep(.005)
                    continue
                sequence, captured, frame = packet
                detections = None
            else:
                # Sequential file playback preserves frames and follows source pacing when possible.
                delay = next_video_time-time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 1/video_fps))
                ok, frame = video.read()
                if not ok:
                    break
                next_video_time = max(next_video_time+1/video_fps, time.monotonic())
                sequence += 1
                captured = time.monotonic()
                detections = None
            infer_start = time.monotonic()
            if detections is None:
                detections = detector.detect(frame)
            infer_end = time.monotonic()
            age = infer_end-captured
            if age > cfg['max_frame_age']:
                speaker.update({}, None)
                log.write('stale_frame', frame_id=sequence, age_s=age)
                tracks, event = [], None
            else:
                tracks, event = pipeline.process(detections, infer_end, (frame.shape[1],frame.shape[0]))
                snapshot = {t.id:(t.signature,t.last_seen) for t in tracks}
                speaker.update(snapshot, event)
            count += 1
            fps = 1/(infer_end-previous_processed) if previous_processed else 0
            previous_processed = infer_end
            log.write('frame', frame_id=sequence, captured=captured, processed=infer_end,
                      inference_ms=1000*(infer_end-infer_start), frame_age_ms=1000*age,
                      detections=[asdict(d) for d in detections],
                      tracks=[{'id':t.id, **asdict(t.observation)} for t in tracks],
                      candidate=asdict(event) if event else None)
            status = f'{"MUTED" if muted else speaker.status} | {speaker.error or ("SILENT SIMULATION" if args.no_audio else "offline speech")} | {detector.device if detector else "no model"}'
            last_canvas = render(frame, tracks, last_warning, status, mode, fps)
            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True,exist_ok=True)
                    writer = cv2.VideoWriter(args.record,cv2.VideoWriter_fourcc(*'mp4v'),
                                             20,(last_canvas.shape[1],last_canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot open video writer')
                writer.write(last_canvas)
            if not (args.headless or args.audio_first):
                cv2.imshow('Smart Obstacle Narrator',last_canvas)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                if key == ord('m'):
                    muted = not muted
                    speaker.set_muted(muted)
                if key == ord('p'):
                    paused = True
                    speaker.update({}, None)
                if key == ord('r'):
                    e = pipeline.repeat(time.monotonic())
                    speaker.update({t.id:(t.signature,t.last_seen) for t in tracks},e)
                if key == ord('h'):
                    print('Q quit | M mute (active sentence may finish) | P pause | R repeat current | H help')
        log.write('summary', frames=count, elapsed_s=time.monotonic()-start)
    finally:
        if speaker:
            speaker.close()
        if capture:
            capture.close()
        if video:
            video.release()
        if writer:
            writer.release()
        if args.snapshot and last_canvas is not None:
            Path(args.snapshot).parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(args.snapshot,last_canvas)
        if not (args.headless or args.audio_first):
            cv2.destroyAllWindows()
        log.close()

def main():
    args = parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)

if __name__ == '__main__':
    main()

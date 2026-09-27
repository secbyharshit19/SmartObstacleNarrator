# Evaluation protocol

Freeze the dataset test split and field scenario script before final evaluation. Keep location/video groups independent. Record model version, code revision, camera, resolution, device, config and environment versions.

## Software and hardware checks

Run pytest. Then test Windows SAPI speech, webcam startup/disconnect, missing model, missing camera, mute, pause, repeat, CPU fallback and a 15-minute continuous run. No speech backlog should accumulate. Windows/device checks cannot be replaced by Linux synthetic tests.

## Metrics

Detection: precision, recall, mAP50, mAP50:95, confusion matrix and per-class failures. Distance: MAE, median absolute error, relative error and coverage. Direction: accuracy with independent labels and boundary-case counts. System: processed FPS, inference median/p95, frame age, driver-start delay, warnings/minute, missed events and false warnings/minute.

The logger's inference timing includes detector preprocessing and moving outputs back to CPU, which synchronizes completed predictions; it is not a raw CUDA-kernel benchmark. Use explicit device timers for isolated GPU-kernel measurements. FPS comes from processed-frame timestamps, not camera-advertised FPS. Queue delay starts from the latest submitted event and excludes confirmation time. Event-ground-truth latency includes the event's labeled onset. Acoustic onset needs synchronized recorded audio/video, not just a driver callback.

```powershell
.\.venv\Scripts\python.exe -m smart_obstacle_narrator.evaluation results/session.jsonl --truth results/event_truth.csv --distances results/distance_truth.csv --directions results/direction_truth.csv --output results/measured_metrics.json
```

CSV templates are included in results/. Event columns: start_s,end_s,label,direction. Time zero is the first logged frame's capture timestamp. Define one row per required warning encounter and another row for a required escalation. The evaluator greedily matches one spoken warning per event, with matching class and direction inside the permitted interval. Unmatched warnings count as false/redundant; unmatched events count as misses. This simple matcher cannot resolve simultaneous same-class/same-direction encounters—review those manually and document them.

Distance columns: actual_m,estimated_m; leave the estimate empty when unknown. Direction columns: actual,predicted, with values left/ahead/right.

## Field scenarios

Use at least five scripted repetitions of hallway, classroom, staircase, supervised sidewalk, parking, crowd, low light, changed camera height, measured distance and occlusion scenarios. Include empty scenes and confusing negatives. Use a sighted operator and supervisor. Never walk blindfolded or rely on this system for road/stair safety.

Record:

| Scenario | Trials | Required events | Matched | Missed | False | p95 onset | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Hallway | pending | pending | pending | pending | pending | pending | |
| Classroom | pending | pending | pending | pending | pending | pending | |
| Stairs | pending | pending | pending | pending | pending | pending | |
| Parking | pending | pending | pending | pending | pending | pending | |
| Low light | pending | pending | pending | pending | pending | pending | |

Proposed project targets: 10 processed FPS on the declared laptop; p95 eligible-event-to-acoustic onset under 1.5 s; 90% required warnings correct in the frozen scripted suite. These are targets, not achieved results or safety thresholds. Report failed targets honestly. Synthetic run summaries bundled here do not count toward any of them.

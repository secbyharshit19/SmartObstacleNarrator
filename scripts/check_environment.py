"""Inspect local dependencies/device and optionally test actual audible speech."""
import argparse,importlib.metadata,platform
p=argparse.ArgumentParser();p.add_argument('--speak',action='store_true');a=p.parse_args()
print(platform.platform(),platform.python_version())
for name in ('ultralytics','opencv-python','torch','torchvision','numpy','PyYAML','pyttsx3'):
    try:print(name,importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:print(name,'MISSING')
try:
    import torch
    print('CUDA available:',torch.cuda.is_available())
    if torch.cuda.is_available():print(torch.cuda.get_device_name(0))
except ImportError:pass
if a.speak:
    from smart_obstacle_narrator.speech import SpeechWorker,PyttsxBackend
    from smart_obstacle_narrator.types import WarningEvent
    import time
    w=SpeechWorker(0,10,PyttsxBackend);w.ready.wait(5)
    if w.status=='error':raise SystemExit(w.error)
    now=time.monotonic();e=WarningEvent('check',0,'Smart obstacle narrator audio test.',100,now,now+10,('ahead','unknown'))
    w.update({0:(e.signature,now)},e)
    try:
        while True:
            f=w.feedback.get(timeout=10);print(f.kind,f.detail)
            if f.kind in ('finished','error'):break
    finally:w.close()

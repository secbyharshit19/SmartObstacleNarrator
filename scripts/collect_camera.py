"""Capture real images: S saves; Q quits. Each run must have a unique session directory."""
import argparse
from pathlib import Path
import time
import cv2
p=argparse.ArgumentParser();p.add_argument('--camera',type=int,default=0);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Use a new session directory to preserve grouping')
a.output.mkdir(parents=True)
cap=cv2.VideoCapture(a.camera)
if not cap.isOpened():raise SystemExit('Camera unavailable')
try:
    while True:
        ok,frame=cap.read()
        if not ok:break
        cv2.imshow('S save | Q quit',frame)
        key=cv2.waitKey(1)&0xff
        if key==ord('q'):break
        if key==ord('s'):
            path=a.output/f'{time.time_ns()}.jpg'
            if not cv2.imwrite(str(path),frame):raise RuntimeError('Save failed')
            print(path)
finally:
    cap.release();cv2.destroyAllWindows()

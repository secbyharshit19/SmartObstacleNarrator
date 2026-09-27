"""Calibrate a known-height approximation and check independent measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median,mean
from datetime import datetime,timezone
import yaml
from .types import CLASSES

def calibrate(rows: list[dict], camera_id: str, width: int, height: int, object_class: str,
              object_height: float, max_relative_error: float = .25) -> dict:
    if object_class not in CLASSES or not math.isfinite(object_height) or object_height<=0:
        raise ValueError('Invalid object class/height')
    if width<=0 or height<=0 or not 0<max_relative_error<1:
        raise ValueError('Invalid resolution or error limit')
    measurements=[]
    for r in rows:
        z,h=float(r['distance_m']),float(r['pixel_height'])
        if not all(math.isfinite(x) and x>0 for x in (z,h)) or h>=height or r['role'] not in ('fit','test'):
            raise ValueError('Invalid measurement or role')
        measurements.append((r['role'],z,h))
    fit=[(z,h) for role,z,h in measurements if role=='fit']
    test=[(z,h) for role,z,h in measurements if role=='test']
    if len(fit)<3 or len(test)<3 or len({z for z,h in fit})<3 or len({z for z,h in test})<3:
        raise ValueError('Need >=3 fit and >=3 held-out measurements at distinct distances in each set')
    focal=median(z*h/object_height for z,h in fit)
    relative=[abs(object_height*focal/h-z)/z for z,h in test]
    mae=mean(abs(object_height*focal/h-z) for z,h in test)
    passed=max(relative)<=max_relative_error
    return dict(enabled=passed,camera_id=str(camera_id),width=width,height=height,focal_y_px=focal,
                valid_min_m=min(z for z,h in test),valid_max_m=max(z for z,h in test),
                relative_uncertainty=max(max(relative),.1),object_heights_m={object_class:object_height},
                calibration_date=datetime.now(timezone.utc).isoformat(),
                evaluation={'mae_m':mae,'max_relative_error':max(relative),'heldout_samples':len(test),
                            'accepted':passed,'assumption':'Same measured object type, pose, camera, zoom, resolution'})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',type=Path);p.add_argument('--camera-id',default='0')
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    p.add_argument('--class-name',required=True,choices=CLASSES);p.add_argument('--object-height',type=float,required=True)
    p.add_argument('--output',type=Path,default=Path('configs/camera_measured.yaml'))
    a=p.parse_args(); rows=list(csv.DictReader(a.csv.open(encoding='utf-8',newline='')))
    result=calibrate(rows,a.camera_id,a.width,a.height,a.class_name,a.object_height)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(yaml.safe_dump(result,sort_keys=False),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['enabled']:raise SystemExit('Calibration exceeds error limit; remains disabled')

if __name__=='__main__':main()

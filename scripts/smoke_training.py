"""Execution-only test of save/resume/release. Synthetic data is NOT obstacle training evidence."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from smart_obstacle_narrator.training import train_release,PlannedInterruption
from smart_obstacle_narrator.dataset import digest
p=argparse.ArgumentParser();p.add_argument('--workdir',type=Path,required=True);a=p.parse_args()
if a.workdir.exists():raise SystemExit('Use a new work directory')
a.workdir.mkdir(parents=True);data=a.workdir/'dataset';rows=[]
rng=np.random.default_rng(741)
for split in ('train','val','test'):
    for i in range(2):
        im=data/'images'/split/f'{i}.png';im.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rng.integers(0,255,(64,64,3),dtype=np.uint8)).save(im)
        label=data/'labels'/split/f'{i}.txt';label.parent.mkdir(parents=True,exist_ok=True)
        label.write_text('\n'.join(f'{c} {.15+(c%4)*.23} {.3+(c//4)*.4} .12 .2' for c in range(8)))
        rows.append(dict(image=im.relative_to(data).as_posix(),label=label.relative_to(data).as_posix(),group=f'{split}{i}',source='synthetic execution fixture',split=split,sha256=digest(im)))
with (data/'manifest.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
torch.set_num_threads(2)
config=dict(model=str(Path('yolo11n.pt').resolve()),epochs=2,imgsz=64,batch=2,workers=0,save_period=1,
            plots=False,amp=False,mosaic=0,fliplr=0,scale=0,translate=0,device='cpu')
config.pop('device')
try:
    train_release(data,a.workdir/'storage','recovery','smoke',config,device='cpu',allow_incomplete=True,
                  near_duplicates_reviewed=True,interrupt_after=1)
except PlannedInterruption:
    print('Checkpoint interruption reached as expected')
else:raise AssertionError('Interruption did not occur')
result=train_release(data,a.workdir/'storage','recovery','smoke',config,resume=True,device='cpu',
                     allow_incomplete=True,near_duplicates_reviewed=True)
assert result['fixture_only']
(a.workdir/'verification.json').write_text(json.dumps({'checkpoint_resume':'passed','release_export':'passed',
        'purpose':'Synthetic execution test only; no custom obstacle accuracy claim','packages':result['packages']},indent=2))
print('Checkpoint recovery and release test passed')

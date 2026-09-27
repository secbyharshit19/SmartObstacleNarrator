"""Durable training/release workflow. Intended for mounted Drive or another persistent directory."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import yaml
from .dataset import validate,digest,EXTENSIONS
from .types import CLASSES

class PlannedInterruption(RuntimeError):
    """Used only to verify recovery after a saved epoch."""

def write_json(path: Path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,default=str,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)

def dataset_hash(root: Path) -> str:
    import hashlib
    h=hashlib.sha256()
    for folder in ('images','labels'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and (p.suffix.lower() in EXTENSIONS if folder=='images' else p.suffix=='.txt'):
                h.update(str(p.relative_to(root)).encode());h.update(digest(p).encode())
    if (root/'manifest.csv').exists():h.update(digest(root/'manifest.csv').encode())
    return h.hexdigest()

def train_release(dataset: Path, storage: Path, run_id: str, version: str, config: dict,
                  resume: bool=False, device: str='0', allow_incomplete: bool=False,
                  near_duplicates_reviewed: bool=False, interrupt_after: int | None=None) -> dict:
    from ultralytics import YOLO
    dataset,storage=dataset.resolve(),storage.resolve()
    if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) for name in (run_id, version)):
        raise ValueError('Run ID and version must be simple names')
    report=validate(dataset)
    if not report['valid']:raise ValueError(json.dumps(report['errors'],indent=2))
    if not (dataset/'manifest.csv').exists() and not allow_incomplete:
        raise ValueError('Dataset needs manifest.csv with session groups')
    if any(n==0 for split in report['instances'].values() for n in split.values()) and not allow_incomplete:
        raise ValueError('At least one class is missing from a split; fix split/data before release training')
    if report['near_duplicate_candidates'] and not near_duplicates_reviewed:
        raise ValueError('Review cross-split near-duplicate candidates; remove leakage, then explicitly acknowledge review')
    identity=dataset_hash(dataset)
    run_dir=storage/'experiments'/run_id
    release=storage/'models'/version
    if release.exists():raise FileExistsError(f'Immutable release exists: {release}')
    if run_dir.exists() and not resume:raise FileExistsError('Run exists; use resume or a new run ID')
    run_dir.mkdir(parents=True,exist_ok=True)
    metadata_path=run_dir/'run_metadata.json'
    if resume:
        old=json.loads(metadata_path.read_text(encoding='utf-8'))
        if old['dataset_sha256']!=identity or old['training_config']!=config:
            raise ValueError('Resume dataset/config differs from checkpoint run')
        checkpoint=run_dir/'weights'/'last.pt'
        if not checkpoint.is_file():raise FileNotFoundError('No last.pt; restore a valid epoch checkpoint before resuming')
    else:
        try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):commit='source archive (no Git commit)'
        write_json(metadata_path,dict(dataset_sha256=identity,training_config=config,git_commit=commit,
                                    created=datetime.now(timezone.utc).isoformat(),allow_incomplete=allow_incomplete))
    write_json(run_dir/'dataset_validation.json',report)
    data={'path':str(dataset),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(CLASSES))}
    data_yaml=dataset/'runtime_data.yaml';data_yaml.write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    model=YOLO(str(checkpoint) if resume else config.get('model','yolo11n.pt'))
    def saved(trainer):
        last=Path(trainer.last)
        if last.exists():
            write_json(run_dir/'checkpoint_status.json',{'epoch':int(trainer.epoch),'path':str(last),
                       'bytes':last.stat().st_size,'sha256':digest(last),'saved_at':datetime.now(timezone.utc).isoformat()})
        if interrupt_after is not None and trainer.epoch+1>=interrupt_after:
            raise PlannedInterruption('Intentional stop AFTER checkpoint save; restart with resume=True')
    model.add_callback('on_model_save',saved)
    if resume:
        model.train(resume=True,device=device)
    else:
        arguments={k:v for k,v in config.items() if k!='model'}
        model.train(data=str(data_yaml),project=str(storage/'experiments'),name=run_id,
                    exist_ok=True,device=device,**arguments)
    best=run_dir/'weights'/'best.pt'
    if not best.is_file():raise RuntimeError('Training completed without best.pt')
    selected=YOLO(str(best))
    result_dir=storage/'results'/version
    result_dir.mkdir(parents=True,exist_ok=True)
    metric_results={}
    for split in ('val','test'):
        metrics=selected.val(data=str(data_yaml),split=split,imgsz=config.get('imgsz',640),
                             batch=config.get('batch',8),device=device,plots=True,
                             project=str(result_dir),name=split,exist_ok=True)
        metric_results[split]={'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),
                              'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),
                              'per_class_mAP50_95':{CLASSES[i]:float(v) for i,v in enumerate(metrics.box.maps)}}
    examples=sorted((dataset/'images'/'test').glob('*'))[:8]
    if examples:
        selected.predict([str(p) for p in examples],save=True,project=str(result_dir),name='examples',
                         exist_ok=True,device=device,imgsz=config.get('imgsz',640),verbose=False)
    release.mkdir(parents=True)
    weight_name=f'best_{version}.pt'
    shutil.copy2(best,release/weight_name)
    versions={name:importlib.metadata.version(name) for name in ('ultralytics','torch','torchvision','numpy','opencv-python','PyYAML')}
    metadata=dict(version=version,run_id=run_id,weight_file=weight_name,weight_sha256=digest(release/weight_name),
                  classes=list(CLASSES),dataset_sha256=identity,training_config=config,packages=versions,
                  metrics=metric_results,created=datetime.now(timezone.utc).isoformat(),
                  experimental=True,fixture_only=allow_incomplete)
    write_json(release/'metadata.json',metadata)
    write_json(release/'classes.json',list(CLASSES))
    write_json(result_dir/'metrics.json',metric_results)
    (release/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    write_json(storage/'models'/'current_model.json',{'version':version,'relative_path':f'{version}/{weight_name}'})
    return metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--storage',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--version',required=True)
    p.add_argument('--config',type=Path,default=Path('configs/training.yaml'));p.add_argument('--device',default='0')
    p.add_argument('--resume',action='store_true');p.add_argument('--near-duplicates-reviewed',action='store_true')
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    print(json.dumps(train_release(a.dataset,a.storage,a.run_id,a.version,cfg,a.resume,a.device,
                                  near_duplicates_reviewed=a.near_duplicates_reviewed),indent=2))

if __name__=='__main__':main()

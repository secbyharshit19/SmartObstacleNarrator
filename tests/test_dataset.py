import csv
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import pytest
from smart_obstacle_narrator.dataset import prepare,validate,read_labels,safe_path

@pytest.fixture
def prepared(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();rows=[]
    rng=np.random.default_rng(42)
    for i in range(12):
        Image.fromarray(rng.integers(0,255,(40,40,3),dtype=np.uint8)).save(raw/f'{i}.png')
        (raw/f'{i}.txt').write_text('0 .5 .5 .4 .4\n')
        rows.append(dict(image=f'{i}.png',label=f'{i}.txt',group=f'group{i}',source='synthetic test fixture'))
    manifest=raw/'input.csv'
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    out=tmp_path/'prepared';prepare(manifest,raw,out)
    return out

def test_group_split_and_manifest(prepared):
    r=validate(prepared);assert r['valid']
    rows=list(csv.DictReader((prepared/'manifest.csv').open()))
    groups={}
    for row in rows:groups.setdefault(row['group'],set()).add(row['split'])
    assert all(len(s)==1 for s in groups.values())

def test_exact_leakage_detected(prepared):
    src=next((prepared/'images'/'train').glob('*'))
    dst=next((prepared/'images'/'test').glob('*'));shutil.copy2(src,dst)
    assert not validate(prepared)['valid']

def test_missing_label_fails(prepared):
    next((prepared/'labels'/'train').glob('*')).unlink()
    assert not validate(prepared)['valid']

@pytest.mark.parametrize('row',['8 .5 .5 .2 .2','0 .95 .5 .2 .2','0 nan .5 .2 .2','0 .5 .5 0 .2','0 .5 .5 .2','0.5 .5 .5 .2 .2'])
def test_bad_labels(tmp_path,row):
    p=tmp_path/'bad.txt';p.write_text(row)
    with pytest.raises(ValueError):read_labels(p)

def test_path_escape(tmp_path):
    with pytest.raises(ValueError):safe_path(tmp_path,'../secret.txt')

def test_training_identity_ignores_generated_caches(prepared):
    from smart_obstacle_narrator.training import dataset_hash
    before=dataset_hash(prepared)
    (prepared/'labels'/'train.cache').write_bytes(b'generated runtime cache')
    assert dataset_hash(prepared)==before
    label=next((prepared/'labels'/'train').glob('*.txt'))
    label.write_text('0 .5 .5 .3 .3\n')
    assert dataset_hash(prepared)!=before

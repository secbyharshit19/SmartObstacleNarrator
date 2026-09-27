"""Dataset preparation, audit, grouped splits, and video-frame extraction."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import yaml
from .types import CLASSES

EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.webp'}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_path(root: Path, relative: str) -> Path:
    target = (root/relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes dataset root: {relative}')
    return target

def read_labels(path: Path) -> list[tuple]:
    if not path.is_file():
        raise ValueError(f'Missing label (negative images need an empty .txt): {path}')
    result = []
    for line_no,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path}:{line_no}: expected five fields')
        values = list(map(float,parts))
        cls,x,y,w,h = values
        if not all(math.isfinite(v) for v in values) or not cls.is_integer() or not 0 <= cls < len(CLASSES):
            raise ValueError(f'{path}:{line_no}: invalid class/numeric data')
        if not (0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and x+w/2<=1+1e-6 and y-h/2>=-1e-6 and y+h/2<=1+1e-6):
            raise ValueError(f'{path}:{line_no}: box outside normalized image bounds')
        result.append((int(cls),x,y,w,h))
    if len(result) != len(set(result)):
        raise ValueError(f'Duplicate annotation rows in {path}')
    return result

def perceptual_hash(image) -> int:
    # 64-bit difference hash, used to flag candidates, not prove duplicates.
    from PIL import Image
    small = image.convert('L').resize((9,8),Image.Resampling.BILINEAR)
    pixels = list(small.get_flattened_data() if hasattr(small, 'get_flattened_data') else small.getdata())
    n = 0
    for y in range(8):
        for x in range(8):
            n = (n << 1) | int(pixels[y*9+x] > pixels[y*9+x+1])
    return n

def validate(root: Path, near_duplicates: bool = True) -> dict:
    from PIL import Image
    root = root.resolve()
    errors, warnings, hashes, phashes = [], [], {}, []
    counts = {s:{name:0 for name in CLASSES} for s in ('train','val','test')}
    images_per_split, negatives = {}, {}
    image_paths = set()
    for split in counts:
        files = sorted(p for p in (root/'images'/split).glob('*') if p.suffix.lower() in EXTENSIONS)
        images_per_split[split] = len(files)
        negatives[split] = 0
        if not files:
            errors.append(f'No images in {split}')
        stems = set()
        for image_path in files:
            relative = str(image_path.relative_to(root)).replace('\\','/')
            image_paths.add(relative)
            if image_path.stem in stems:
                errors.append(f'Duplicate image stem in {split}: {image_path.stem}')
            stems.add(image_path.stem)
            try:
                with Image.open(image_path) as im:
                    im.load()
                    if im.width < 16 or im.height < 16:
                        raise ValueError('Image too small')
                    if near_duplicates:
                        phashes.append((split,relative,perceptual_hash(im)))
                labels = read_labels(root/'labels'/split/(image_path.stem+'.txt'))
                if not labels:
                    negatives[split] += 1
                for cls,*_ in labels:
                    counts[split][CLASSES[cls]] += 1
                sha = digest(image_path)
                if sha in hashes:
                    errors.append(f'Exact duplicate: {relative} and {hashes[sha]}')
                hashes[sha] = relative
            except Exception as exc:
                errors.append(f'{relative}: {exc}')
        for label in (root/'labels'/split).glob('*.txt'):
            if label.stem not in stems:
                errors.append(f'Orphan label: {label.relative_to(root)}')
        for name,n in counts[split].items():
            if n == 0:
                warnings.append(f'{split}: no {name} instances')
    manifest = root/'manifest.csv'
    if manifest.exists():
        groups, listed = {}, set()
        for row in csv.DictReader(manifest.open(encoding='utf-8',newline='')):
            if not row.get('group') or row.get('split') not in counts:
                errors.append('Manifest requires nonempty group and valid split'); continue
            groups.setdefault(row['group'],set()).add(row['split'])
            listed.add(row['image'].replace('\\','/'))
            try:
                image = safe_path(root,row['image'])
                if row.get('sha256') and image.is_file() and digest(image) != row['sha256']:
                    errors.append(f'Checksum mismatch: {row["image"]}')
            except ValueError as exc:
                errors.append(str(exc))
        for group,splits in groups.items():
            if len(splits) > 1:
                errors.append(f'Group leakage: {group} in {sorted(splits)}')
        if listed != image_paths:
            errors.append('Manifest image set differs from actual image set')
    else:
        warnings.append('No manifest.csv; grouped leakage cannot be verified')
    candidates = []
    if near_duplicates:
        for i,(split,path,phash) in enumerate(phashes):
            for split2,path2,phash2 in phashes[i+1:]:
                if split != split2 and (phash^phash2).bit_count() <= 4:
                    if len(candidates) < 500:
                        candidates.append([path,path2])
        if candidates:
            warnings.append('Cross-split near-duplicate candidates require visual review; capped at 500')
    return {'valid':not errors,'errors':errors,'warnings':warnings,'instances':counts,
            'images':images_per_split,'negative_images':negatives,'near_duplicate_candidates':candidates}

def prepare(manifest: Path, root: Path, output: Path, seed: int = 42):
    rows = list(csv.DictReader(manifest.open(encoding='utf-8',newline='')))
    if not rows or not {'image','label','group','source'}.issubset(rows[0]):
        raise ValueError('Manifest columns required: image,label,group,source')
    groups = sorted({r['group'] for r in rows})
    if '' in groups or len(groups) < 3:
        raise ValueError('At least three nonempty independent groups required')
    if output.exists():
        raise FileExistsError('Output already exists; choose a new dataset version')
    random.Random(seed).shuffle(groups)
    n_val = max(1,round(len(groups)*.15)); n_test = max(1,round(len(groups)*.15))
    assignment = {g:('val' if i<n_val else 'test' if i<n_val+n_test else 'train') for i,g in enumerate(groups)}
    seen = set()
    prepared = []
    # Check every input before creating output.
    for row in rows:
        image, label = safe_path(root,row['image']), safe_path(root,row['label'])
        if image.suffix.lower() not in EXTENSIONS:
            raise ValueError(f'Unsupported image: {image}')
        read_labels(label)
        sha = digest(image)
        if sha in seen:
            raise ValueError(f'Exact duplicate input: {image}; remove duplicates before splitting')
        seen.add(sha)
        prepared.append((row,image,label,sha))
    output.mkdir(parents=True)
    with (output/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=['image','label','group','source','split','sha256'])
        writer.writeheader()
        for i,(row,image,label,sha) in enumerate(prepared):
            split = assignment[row['group']]
            name = f'{i:06d}_{sha[:10]}'
            img_rel = f'images/{split}/{name}{image.suffix.lower()}'
            lab_rel = f'labels/{split}/{name}.txt'
            for source,rel in ((image,img_rel),(label,lab_rel)):
                dest = output/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            writer.writerow(dict(image=img_rel,label=lab_rel,group=row['group'],source=row['source'],split=split,sha256=sha))
    data = {'path':str(output.resolve()), 'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(CLASSES))}
    (output/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    report = validate(output)
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def extract(video: Path, out: Path, seconds: float):
    import cv2
    if seconds <= 0:
        raise ValueError('Interval must be positive')
    if out.exists():
        raise FileExistsError('Use a new session output directory')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot read {video}')
    out.mkdir(parents=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release(); raise ValueError('Video has invalid FPS')
    step = max(1,round(seconds*fps)); index=0; saved=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if index%step==0:
                if not cv2.imwrite(str(out/f'frame_{index:07d}.jpg'),frame):
                    raise RuntimeError('Failed to save frame')
                saved+=1
            index+=1
    finally:cap.release()
    return {'saved':saved,'session':out.name,'note':'Annotate before preparation; all frames belong to one split group'}

def coco_subset(annotation: Path, images: Path, output: Path, limit: int):
    from .detector import BASELINE_MAP
    data=json.loads(annotation.read_text(encoding='utf-8'))
    mapping={c['id']:CLASSES.index(BASELINE_MAP[c['name']]) for c in data['categories'] if c['name'] in BASELINE_MAP}
    annotations={}
    for a in data['annotations']:
        if a['category_id'] in mapping:
            annotations.setdefault(a['image_id'],[]).append(a)
    if output.exists():raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    selected=[im for im in data['images'] if im['id'] in annotations][:limit]
    manifest=[]
    for im in selected:
        source=safe_path(images,im['file_name'])
        if not source.is_file():raise FileNotFoundError(source)
        rows=[]
        for a in annotations[im['id']]:
            if a.get('iscrowd'):
                continue
            x,y,w,h=a['bbox']; x1=max(0,x);y1=max(0,y);x2=min(im['width'],x+w);y2=min(im['height'],y+h)
            if x2>x1 and y2>y1:
                rows.append(f"{mapping[a['category_id']]} {(x1+x2)/2/im['width']:.8f} {(y1+y2)/2/im['height']:.8f} {(x2-x1)/im['width']:.8f} {(y2-y1)/im['height']:.8f}")
        # Skip crowd-containing images rather than silently treating crowds as background.
        if any(a.get('iscrowd') for a in annotations[im['id']]):continue
        dest=output/'images'/Path(im['file_name']).name; dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest)
        lab=output/'labels'/(dest.stem+'.txt');lab.parent.mkdir(exist_ok=True);lab.write_text('\n'.join(rows),encoding='utf-8')
        manifest.append({'image':str(dest.relative_to(output)),'label':str(lab.relative_to(output)),
                         'group':f'coco_{im["id"]}','source':f'COCO image {im["id"]}; license_id={im.get("license","unknown")} - audit required'})
    with (output/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['image','label','group','source']);writer.writeheader();writer.writerows(manifest)
    return {'images':len(manifest),'warning':'Audit stairs/poles/barriers/doors and related-image groups BEFORE merging; preserve source licenses'}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('validate');q.add_argument('root',type=Path);q.add_argument('--output',type=Path);q.add_argument('--skip-near',action='store_true')
    q=sub.add_parser('prepare');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--seed',type=int,default=42)
    q=sub.add_parser('extract');q.add_argument('video',type=Path);q.add_argument('--output',type=Path,required=True);q.add_argument('--interval',type=float,default=1)
    q=sub.add_parser('coco');q.add_argument('--annotations',type=Path,required=True);q.add_argument('--images',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--limit',type=int,default=1000)
    a=p.parse_args()
    if a.command=='validate':
        result=validate(a.root,not a.skip_near)
        if a.output:a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    elif a.command=='prepare':result=prepare(a.manifest,a.root,a.output,a.seed)
    elif a.command=='extract':result=extract(a.video,a.output,a.interval)
    else:result=coco_subset(a.annotations,a.images,a.output,a.limit)
    print(json.dumps(result,indent=2))
    if result.get('valid') is False:raise SystemExit(1)

if __name__=='__main__':main()

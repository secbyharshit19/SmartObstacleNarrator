# Dataset guide

## Scope

Class IDs: 0 person; 1 chair; 2 bicycle; 3 vehicle (car/motorcycle/bus/truck); 4 stairs; 5 pole; 6 barrier; 7 door.

Stairs: one visible contiguous flight, not individual steps. Pole: exposed freestanding post; exclude trees and door frames initially. Barrier: freestanding barricade obstructing passage; exclude walls/fences/ropes. Door: consistently box the door/doorway assembly; it is a landmark, not an inferred blockage or open/closed state.

Target 2,500–4,000 distinct images, including 10–15% negatives, with at least half locally collected. Per-class target image appearances: person 500, chair 400, bicycle 400, vehicle 600, stairs 600, pole 500, barrier 400, door 400. Images can contain several classes. These are planning targets, not guaranteed training sufficiency.

## Collect

Use 20–30 independent scene/session groups across multiple locations, days, devices and camera heights. Include indoor/outdoor lighting, oblique views, partial occlusion, moderate blur, confusing shadows, and examples of each class at multiple scales. Use safe controlled locations; no blindfolded navigation or road exposure is needed.

```powershell
.\.venv\Scripts\python.exe scripts/collect_camera.py --camera 0 --output datasets/raw/hallway_day1
.\.venv\Scripts\python.exe -m smart_obstacle_narrator.dataset extract phone_clip.mp4 --output datasets/raw/stairs_day2 --interval 1
```

All images from one clip or closely related session belong to one group. Group related physical scenes together even if collected on different days when they would reveal the same background or object identity to the test set.

## Annotate

Use CVAT with exactly the eight class names in the specified order. Export Ultralytics YOLO detection labels. Check the exported class IDs; an export can order names differently. Remap or re-export before merging. Boxes contain class center_x center_y width height, normalized to [0,1]. Label every target object; true negatives need an empty .txt. Do not mark an unlabeled positive image as negative.

Pilot: all members annotate the same 30 images, reconcile interpretations, then divide sessions. Cross-review at least 10%. Keep difficult cases in validation/test, not only attractive examples.

## Manifest and split

Create `datasets/raw/manifest.csv`, with paths relative to `datasets/raw`:

```csv
image,label,group,source
hallway_day1/001.jpg,labels/001.txt,hallway_A_day1,own camera with consent
parking_day2/002.jpg,labels/002.txt,parking_B_day2,own phone
```

This is an example, not existing data. Put paths to your real files in the manifest. Use consistent unique label filenames.

```powershell
.\.venv\Scripts\python.exe -m smart_obstacle_narrator.dataset prepare --manifest datasets/raw/manifest.csv --root datasets/raw --output datasets/prepared/v1
.\.venv\Scripts\python.exe -m smart_obstacle_narrator.dataset validate datasets/prepared/v1 --output results/dataset_validation.json
```

The split shuffles whole groups with seed 42 and assigns approximately 70/15/15 by group count. It does not guarantee image-count or class balance. Inspect counts; adjust group composition or collect data if a class is missing. Training blocks incomplete real releases.

Checks include image decoding, labels, box bounds, duplicate rows, exact duplicates, orphan labels, group leakage, manifest coverage, checksum mismatch and near-duplicate candidates. Difference-hash similarity is a review aid, not proof. Do not simply approve related frames across splits.

## Public data

COCO provides person/chair/bicycle and vehicle categories. Custom stairs/poles/barriers/doors need your annotation or an independently verified source. Optional import:

```powershell
.\.venv\Scripts\python.exe -m smart_obstacle_narrator.dataset coco --annotations instances_train2017.json --images coco/train2017 --output datasets/coco_subset --limit 1000
```

The importer maps COCO category IDs into the project order and excludes target crowd images. It does not audit newly added classes or establish related-image grouping. Review all target objects and source licenses before merging. No COCO images are downloaded automatically.

## Augment and version

Keep originals. Only training receives augmentation: moderate brightness/scale/translation, horizontal flips, modest perspective-free rotation and optional mosaic. Never augment before splitting. Rare-class collection is preferable to duplicating images.

Create v2 instead of editing a released v1. The training hash covers images, labels and manifest; generated cache files are excluded. ZIP the **contents** of the prepared directory, with images/, labels/ and manifest.csv at the archive root.

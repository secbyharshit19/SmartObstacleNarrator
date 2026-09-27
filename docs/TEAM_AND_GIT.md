# Team ownership and Git

| Member | Files | Inputs -> outputs | Tests | Branch |
|---|---|---|---|---|
| A | detector.py, training.py, notebook 01 | dataset -> model/detections/metrics | class mapping, model load, save/resume | feature/detection |
| B | geometry.py, calibration.py, camera config | boxes + calibration -> direction/optional depth | boundaries, clipping, unknown, held-out errors | feature/geometry |
| C | temporal.py, narration.py, speech.py | observations -> prioritized speech | association, cooldown, stale queue, driver failure | feature/narration |
| D | app.py, capture.py, display.py, dataset.py, notebook 02 | shared modules -> complete app | integration, camera failure, install and UI | feature/application |

All members collect and annotate data; D coordinates the manifest, not all labor. Tests use fixtures so members can work without waiting for a trained custom model. Changes to types.py and class order need team review.

Keep main demo-ready; integrate on develop; merge short-lived feature branches through a reviewed pull request. Each PR states the problem, behavior change, test evidence and known limits. Do not edit the same notebook concurrently. Commit small logical changes with descriptive messages.

Ignore .pt/.onnx, environments, raw/prepared data, private recordings and secrets. Put model/data versions in Drive with checksums and retrieval instructions. Use release tags and record Git commit in model metadata. Never replace a good model version in place.

## Staged acceptance

0 environment and devices; 1 diverse collection; 2 reviewed labels; 3 reproducible baseline; 4 targeted data improvement; 5 calibrated optional distance; 6 tested direction; 7 multi-object priority; 8 stable warning policy; 9 asynchronous speech; 10 integrated video; 11 usable interface; 12 measured tests; 13 profiled optimization; 14 evidence-backed documents; 15 rehearsed offline demonstration.

Software stages are supplied in this package. Data collection, real custom-model training, calibration and device/field acceptance must use your actual material.

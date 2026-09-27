# Models

The root-level yolo11n.pt is a bundled unmodified COCO baseline. Use --baseline. It supports project narration for person/chair/bicycle/vehicle only.

No custom v1 model has been trained on real project data. After Colab training, download models/v1/ from Drive here, retaining best_v1.pt, metadata.json, classes.json and environment.txt. Use the release checksum and exact class order. Never deploy the synthetic smoke-test checkpoint.

Large model files are ignored by Git. Keep Drive release IDs and checksums in release documentation.

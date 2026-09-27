# Third-party provenance

The included `yolo11n.pt` is an unmodified COCO-pretrained Ultralytics baseline, downloaded from:
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt

Upstream source and model documentation:
https://github.com/ultralytics/ultralytics
https://docs.ultralytics.com/models/yolo11/

Ultralytics provides AGPL-3.0 and enterprise licensing options. This project is distributed under the included AGPL-3.0 text. Upstream authors retain ownership of their code and weights. Model source/architecture and training implementation are available in the upstream repository. Baseline provenance and checksum are recorded in models/baseline_metadata.json.

Other dependencies retain their own licenses, including OpenCV, PyTorch, NumPy, PyYAML, Pillow, pyttsx3, pytest and nbformat. Dependencies are installed separately, not bundled as binaries in this archive.

COCO data is not bundled. Dataset image permissions and licenses must be reviewed at the image/source level. A COCO export command does not grant redistribution rights. Synthetic screenshots and fixtures are clearly labeled. The baseline smoke-test video used a repeated Ultralytics bus sample locally; that source video is not included and its detections are not a field evaluation.

References:
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/modes/predict/
- https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- https://github.com/nateshmbhat/pyttsx3
- https://cocodataset.org/

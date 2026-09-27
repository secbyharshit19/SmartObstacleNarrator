# Smart Obstacle Narrator
## Implementation report

B.Tech Computer Science project. Team names, roll numbers, supervisor, institution approval and submission date should be supplied by the team before academic submission.

**Status:** The software implementation and execution tests are available. Custom training on real project data, camera calibration, Windows hardware acceptance and field evaluation remain pending. This report does not represent those activities as completed.

### Abstract

Smart Obstacle Narrator is an experimental assistive computer-vision application that converts selected object detections into short spoken observations. The implementation combines a YOLO adapter, camera-relative direction, conditional known-height distance estimation, temporal association, priority scoring, warning suppression, offline speech and a local display. The architecture separates capture and speech from the main processing loop and rejects stale warnings. Two independent notebooks support persistent training and media inference through Google Drive. The supplied COCO-pretrained baseline detects four project categories. The eight-class custom release requires an annotated project dataset. Automated tests, a synthetic application run, real baseline inference and a synthetic checkpoint recovery exercise verify software execution. These checks do not establish reliability as a mobility aid.

### 1. Introduction

Visual surroundings contain more objects than a user can reasonably listen to as continuous narration. Repeating every detection can create distraction, delay and stale information. This project investigates concise object warnings with explicit handling of uncertainty and timing. The first deployment target is a Windows laptop with an integrated or USB webcam and offline audio.

### 2. Problem statement

The system must identify a limited set of relevant objects, determine their image-relative direction, estimate depth only under justified assumptions and choose a small number of spoken warnings. It must avoid continuous repetition and cannot imply safety when no object is detected. The engineering question includes detection quality, event selection, speech scheduling and failure handling.

### 3. Technical background and literature sources

YOLO supplies single-stage object detection through a pretrained neural model and custom fine-tuning. Ultralytics documents the selected YOLO11 family, Python prediction outputs and checkpoint resume behavior [1–3]. COCO supplies overlapping common-object categories but does not satisfy the complete custom class list [4]. OpenCV documents camera intrinsics and calibration [5]. CVAT provides compatible detection annotation export [6]. pyttsx3 provides access to installed offline speech engines [7]. These are implementation references, not fabricated empirical comparisons with competing assistive products. A formal related-work literature review should add verified research papers required by the college.

### 4. Existing approaches

Ordinary object detectors return class scores and bounding boxes, leaving users or downstream systems to interpret them. Generic image captioning can describe a scene but does not inherently solve repetitive warning suppression or fresh event scheduling. Depth sensors and advanced mapping can add information but increase hardware and integration requirements. The present scope uses ordinary cameras and transparent rules to keep the initial system understandable and testable.

### 5. Proposed system

The proposed application supports person, chair, bicycle, vehicle, stairs, pole, barrier and door. Its custom dataset must use this exact ordered class contract. Door detection is treated as a landmark and does not establish a closed door or blocked route. Pole and stairs distance remain unknown unless a defensible separate method is later introduced. Speech uses deterministic templates with no LLM.

Three modes separate evidence: generated synthetic boxes exercise software, bundled COCO weights provide four-class baseline detection, and a custom release supports the final eight-class model after real training.

### 6. System requirements

Development targets Python 3.11 and Windows 11, with a webcam and installed speech voice. CPU fallback is supported; an NVIDIA GPU can accelerate inference when compatible PyTorch drivers/wheels are installed. Training targets a Colab T4 runtime, subject to allocation. Runtime dependencies include Ultralytics, OpenCV, NumPy, PyYAML and pyttsx3. No cloud service is required during local inference after setup.

### 7. Methodology

The process consists of scene collection, consistent annotation, grouped split preparation, pretrained fine-tuning, class-specific error analysis, conditional calibration, temporal policy implementation and supervised evaluation. Dataset checks run before training. Validation chooses model checkpoints, while the frozen test split supplies final reporting. Independent video scenarios measure actual warning behavior rather than only image-level metrics.

### 8. System architecture

A capture worker supplies one latest frame. The detector returns boxes in original-frame pixels. Geometry produces direction and optional depth. Lightweight association maintains temporary object state and stabilization. Eligibility suppresses unconfirmed or recently announced events. Priority selects among eligible observations. A speech worker owns one active utterance and one pending event. JSONL logs and the UI expose system behavior.

Shared records provide clear input/output contracts. Modules can be tested independently with fixtures. Configuration controls thresholds and calibration so code does not embed session-specific dimensions or timings.

### 9. Dataset

The collection target is 2,500–4,000 distinct images with 10–15% negatives and substantial locally collected material. Counts are planning assumptions. Training/validation/test separation is approximately 70/15/15 by scene group. Images from a video or related scene remain together. Exact hashes detect duplicates; difference-hash candidates receive visual review. Every visible target class requires annotation, including added classes absent from public-source labels.

**Current result:** no real custom dataset was provided or collected in this build. Collection, annotation counts and class distributions remain pending.

### 10. Model training

Notebook 01 mounts Drive, stages data locally, validates it, trains the selected model and saves run outputs directly to persistent storage. Checkpoints include latest/best/periodic weights. A checksum record documents saved progress. Resume verifies the exact dataset and configuration identity. Released versions contain immutable weights, class order, checksums, environment versions and evaluation metrics. Notebook 02 independently loads a released version.

A tiny six-image synthetic execution fixture verified deliberate interruption after a saved epoch, resumed training, validation and release serialization. Its model is explicitly marked fixture-only and is rejected by release verification. No accuracy conclusion follows from this exercise.

### 11. Distance and direction

Known-height depth uses Z≈Hf_y/h. It estimates optical-axis depth rather than nearest surface distance. Calibration fits a median focal value from known distances and checks independent measurements. The application rejects incompatible camera identity/resolution, clipped objects, very small boxes and out-of-range estimates. Unknown dimensions remain unknown.

Direction uses the bounding-box center divided by frame width. Initial boundaries are 0.35 and 0.65. Hysteresis and a stability interval limit boundary flicker. A wide object can receive central-overlap priority even when its center is off to one side. These image cues do not establish traversability.

### 12. Priority and narration

After confidence and temporal confirmation, a weighted score combines class importance, proximity, central overlap and persistence, scaled by confidence. Unknown depth receives a neutral proximity contribution. The system chooses among eligible events so a suppressed top-scoring object does not prevent another from being heard.

The same signature normally waits eight seconds before a reminder. A stable change in direction or closer band can trigger a new event. Disappearance expires the association. Pending events are checked against current state immediately before speaking. Templates only use distance wording when an estimate is valid.

### 13. Implementation

The package provides typed data boundaries, separate inference and training modules, YAML configuration, a command-line desktop app, dataset preparation, camera collection, calibration, evaluation, notebooks and tests. A dedicated speech worker uses driver callbacks to acknowledge actual software speech start. The main loop keeps processing while speech runs. Camera and audio faults are surfaced rather than hidden.

The desktop UI shows classes, confidence, direction, conditional distance, current warning, system status and processed FPS. Synthetic and baseline modes are visibly labeled. No private images are automatically retained. Explicit recording creates an annotated video, while JSON timestamps supply timing evidence.

### 14. Results

| Result category | Status |
|---|---|
| Automated software checks | 50 tests passed in recorded suite |
| Synthetic application pipeline | Completed with expected warning sequence |
| Pretrained YOLO adapter/video | Executed using bundled COCO weights |
| Checkpoint interruption/resume/release | Passed on synthetic CPU fixture |
| Notebook syntax/schema | Passed |
| Custom eight-class precision/recall/mAP | Pending real dataset and training |
| Calibrated distance accuracy | Pending physical measurements |
| Windows audio and webcam latency | Pending target-device testing |
| Real-world warning reliability | Pending supervised field evaluation |

Machine-specific baseline timing is stored in results/baseline_cpu_tuned_summary.json. It used repeated frames from a static sample and silent speech, not a live camera or field trial. It must not be quoted as real-world assistive performance. Software tests are not detection-accuracy percentages.

### 15. Testing

Tests cover direction boundaries, uncertainty, invalid depth, multiple same-class associations, cooldown, disappearance, changed state, unsupported boxes, speech queue limits, expired events, engine failures, dataset corruption/leakage, calibration fitting, notebook parsing, real model loading and full app logging. End-to-end tests exposed and corrected a speech-log field conflict. A recovery test exposed and corrected hashing of generated dataset caches.

The remaining hardware protocol checks camera/audio availability, unplugging, mute/pause, CPU fallback and a 15-minute continuous run. Field trials cover hallway, classroom, stairs, parking, low light, crowd, height variation and occlusion with a sighted operator and supervision.

### 16. Limitations

The model can miss objects or misclassify them. Generic box dimensions cannot establish true depth for arbitrary objects. Lightweight association can switch temporary identities in crowds. A currently speaking sentence may delay a new warning. The system cannot identify all drop-offs, infer a safe route, replace a cane or guarantee obstacle coverage. No user study with visually impaired participants has been performed.

### 17. Future scope

After the V1 study, useful extensions include verified phone-camera transport, optimized inference exports, improved association, calibrated depth sensing, haptic warnings, Hindi narration with tested voices, door-state recognition and specialist-guided usability evaluation. Additions should respond to measured failures instead of expanding features without evidence.

### 18. Conclusion

The implementation supplies an executable foundation for obstacle narration with transparent decision rules, conservative distance behavior, persistent training workflows and testable concurrency. Completion of the academic experiment requires the team's actual annotated data, custom training, device calibration and supervised evaluation. Those measured results should replace the pending fields before final submission.

### References

1. Ultralytics YOLO11: https://docs.ultralytics.com/models/yolo11/
2. Ultralytics training: https://docs.ultralytics.com/modes/train/
3. Ultralytics prediction: https://docs.ultralytics.com/modes/predict/
4. COCO: https://cocodataset.org/ and https://docs.ultralytics.com/datasets/detect/coco/
5. OpenCV calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
6. CVAT YOLO export: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/
7. pyttsx3: https://github.com/nateshmbhat/pyttsx3

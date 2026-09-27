# Training, persistence and deployment

## Google Drive layout

| Path under MyDrive/SmartObstacleNarrator | Contents |
|---|---|
| source/SmartObstacleNarrator.zip | Delivered source package |
| datasets/dataset_v1.zip | Prepared dataset contents, not an extra parent directory |
| experiments/run_id/ | best.pt, last.pt, epoch checkpoints, training arguments, curves, metadata |
| models/v1/ | best_v1.pt, metadata.json, classes.json, environment.txt |
| models/current_model.json | Selected immutable release pointer |
| results/v1/ | Validation/test metrics, plots and example predictions |
| notebooks/ | Optional copies of the two notebooks |

Open 01_training.ipynb in Colab. Select a GPU runtime and run cells sequentially. Its assertion checks whether a GPU is actually allocated. The expected target is T4; allocation is not guaranteed by the software. Initial batch 8, imgsz 640, 80 epochs and patience 15 are starting values, not measured optimal settings.

Dataset bytes are staged locally for reads. Experiment outputs live on mounted Drive throughout training. Each model-save callback records a checkpoint checksum and epoch. save_period=1 keeps periodic checkpoints, which consume storage; reduce frequency after recovery has been tested if needed.

## Recovery exercise

Before a long run, set the notebook's INTERRUPT_AFTER=1. The intentional exception occurs after the first checkpoint save. In a fresh runtime, rerun setup/data staging, keep RUN_ID, VERSION, CONFIG and the dataset unchanged, set RESUME=True and INTERRUPT_AFTER=None. Resume restores the saved optimizer/training state.

If last.pt is corrupt, preserve it for diagnosis and restore a complete periodic epoch checkpoint as last.pt in a recovery copy of the experiment. Never resume from untrusted weights. A completed training run cannot be continued with resume=True as though it were interrupted; start a new fine-tuning experiment instead.

Data/config identity checks prevent accidental resume against changed data. Cache files do not affect the hash. Network or Drive failure can still interrupt writes; this is tested best-effort recovery, not zero-loss storage.

## Evaluation and release

The training run selects best.pt on validation behavior. The release workflow evaluates it on validation and test splits, writes precision/recall/mAP, per-class mAP, confusion matrices and examples, and copies the selected model into a new immutable version directory. Existing versions are never overwritten. Freeze the test set and stop tuning on it.

Copy the entire version directory to local models/v1/. The inference notebook mounts the same Drive and verifies the model checksum and class order. No variable or output from the training notebook needs to remain in memory.

## Local deployment

This is a desktop application, not a hosted website. The actual webcam and offline Windows speaker are local. Colab cannot access the laptop webcam through ordinary cv2.VideoCapture(0), and cloud pyttsx3 cannot play through Windows SAPI. The second notebook supports uploaded images/videos and an explicitly simulated warning timeline.

Install the package and dependencies once, keep weights locally, and run the app. Optional later ONNX export should only follow measured profiling and output comparison. No cloud inference service or monthly hosting is required for V1.

## Execution-only recovery test

```powershell
.\.venv\Scripts\python.exe scripts/smoke_training.py --workdir results/recovery_fixture_run
```

This intentionally trains on six synthetic noise images solely to exercise checkpoint interruption, resume, validation and serialization. Its model is marked fixture_only and release verification refuses it. Its metrics are not project results. Use a new work directory for every test.

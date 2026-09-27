# Verification record

This is a record of executed software checks, not a field-validation certificate.

## Executed

- **50 automated tests passed**, with no failures in `results/software_tests.xml`.
- Real YOLO11n weights loaded and ran through the adapter on CPU.
- The actual headless app completed a 16-second synthetic scenario with silent speech and emitted the expected warnings. Its annotated video is `results/synthetic_demo.mp4`.
- A 50-frame video made from repeated Ultralytics bus-sample frames passed through actual baseline inference. This static repeated scene is an execution smoke test, not a test dataset.
- A six-image synthetic training fixture deliberately stopped after epoch checkpoint saving, resumed from that checkpoint, validated and exported model metadata/weights. `results/checkpoint_recovery_verification.json` records success. The fixture-only model is not distributed as a custom obstacle model.
- Both notebooks passed nbformat validation and all Python code cells compiled.
- The 14-slide PowerPoint passed structural/layout checks and its rendered slides were visually inspected. It was not opened in Microsoft PowerPoint here.

## Bugs found and corrected

1. Speech feedback and log records both used the `kind` keyword, causing an integration exception. The log now uses an explicit feedback_kind field and a full app regression test covers it.
2. YOLO-generated label caches changed the dataset fingerprint and prevented resume. Only source image/label files and the manifest contribute to the fingerprint now; a regression test covers generated caches.
3. The initial concurrent CPU smoke run suffered high latency. The adapter now restores a configured two-thread pool after device warm-up. The isolated repeat completed 50 frames without stale-frame drops. The initial and tuned logs remain available. Their conditions differ, so they are not a controlled speedup benchmark.
4. Long object labels near the right edge could clip. The display now uses the full canvas and bounds label placement.

## Interpretation of included numbers

`results/baseline_cpu_tuned_summary.json` reports this container's 50-frame static-video run. Source playback was approximately 10 FPS, so processed FPS does not establish maximum inference throughput. Silent speech provides software callbacks, not audible latency. The synthetic demo has no detector inference cost. Neither run estimates accuracy, safety, Windows performance or generalization.

## Not executed or claimed

- Real project-data collection, annotation or custom eight-class training.
- A Colab T4 session with mounted Google Drive. The underlying training/recovery code ran on local CPU, while Colab-specific mount/setup cells received syntax/schema review only.
- Windows 11 SAPI speech, physical laptop/USB webcam interaction or RTX 3050 inference.
- Physical camera calibration, distance MAE or direction accuracy from real ground truth.
- A 15-minute hardware soak, supervised field evaluation or accessibility user study.

The dependency stack used for these checks is in results/build_environment.json. Windows/Colab dependency resolution may differ. Freeze the actual successful target environment after testing.

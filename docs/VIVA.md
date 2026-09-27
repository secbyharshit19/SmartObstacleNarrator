# Viva and interview preparation

1. **What problem does the project solve?** It turns selected object detections into short, prioritized spoken observations while controlling repetition and stale audio.
2. **Where is the machine learning?** YOLO predicts object classes, confidence scores and bounding boxes. Direction, scoring and sentence generation are deterministic engineering modules.
3. **Why YOLO11n?** It is a lightweight pretrained starting point with a documented training/inference interface. The choice must still be evaluated on the custom data and laptop.
4. **What is a CNN?** A neural network that learns spatial features through convolution-like operations and hierarchical representations of images.
5. **Detection versus classification?** Classification assigns labels to an image or crop. Detection also locates individual objects with boxes.
6. **What is IoU?** Intersection area divided by union area of two boxes. It is used in detection evaluation and our short-term association.
7. **What is NMS?** Non-maximum suppression removes overlapping lower-scoring predictions for conventional detection outputs. YOLO families can differ in postprocessing design.
8. **What is transfer learning?** Starting from pretrained representations and fine-tuning on the project's labeled classes instead of learning all features from random initialization.
9. **Precision?** Correct positive predictions divided by all positive predictions under the selected evaluation matching rule.
10. **Recall?** Detected ground-truth objects divided by all ground-truth objects. Missed hazard classes require attention even when overall metrics look good.
11. **What is AP?** Area under a precision-recall curve for a class and matching threshold. mAP averages AP across classes and, for mAP50:95, across IoU thresholds.
12. **Why avoid random video-frame splitting?** Nearby frames share nearly identical backgrounds and objects. Random splitting can leak scene information and inflate evaluation results.
13. **Why include negative images?** They expose background patterns and false-positive situations. An empty label is valid only when all target classes are genuinely absent.
14. **Why no universal distance estimate?** Box height combines physical size, pose, camera geometry, visibility and depth. Without constraints, different scenes can produce the same box.
15. **Explain Z=Hf/h.** Under pinhole assumptions, apparent height decreases roughly inversely with optical-axis depth. Known physical height and pixel focal length allow an approximation.
16. **Why does resolution matter?** Focal length expressed in pixels scales with image resolution and cropping. Our default implementation rejects mismatches.
17. **Does a confidence score mean a probability?** Not necessarily a calibrated probability. It is a model score used for filtering and ranking.
18. **Why use hysteresis?** Separate entry/exit behavior reduces rapid state changes near a threshold.
19. **How does spam prevention work?** A stable object gets a state signature. Speech-start acknowledgment begins cooldown; closer or changed direction can qualify for another warning.
20. **Why not identify objects only by class and side?** Multiple same-class objects can occupy the same side. Temporary one-to-one association preserves separate event state.
21. **What can break association?** Occlusion, fast motion and crowded crossings can swap or fragment IDs. No identity guarantee is claimed.
22. **Why separate speech execution?** Blocking speech on the inference thread freezes processing and allows scene information to become stale.
23. **Why a bounded queue?** Old warnings should not accumulate behind current speech. Only the latest relevant pending event is retained.
24. **What is a race condition?** Concurrent operations observe or update shared state in an unexpected order. A condition lock protects speech state and a thread-safe feedback queue returns events.
25. **When does cooldown start?** At the speech driver's start callback, not when an event is first proposed.
26. **What is the GIL implication?** Python threads do not guarantee parallel Python CPU computation, but they separate blocking capture/speech tasks and native-library execution. Measure actual behavior.
27. **What survives a Colab disconnect?** Completed artifacts on mounted Drive. The current epoch or interrupted write may be lost. Recovery is tested from a saved checkpoint.
28. **Resume versus loading best weights?** Resume restores training state for an interrupted run. Starting from best weights in a new run is fine-tuning, not exact continuation.
29. **How is reproducibility recorded?** Class order, configuration, data hash, model checksum, environment versions, run ID and code commit/source identity.
30. **What is end-to-end latency?** Time from the defined scene event to audible warning onset. It includes more than model inference and should be measured with synchronized audio/video.
31. **Why does FPS not establish usability?** High throughput can coexist with missed hazards, stale buffering or excessive speech. Event correctness and warning latency also matter.
32. **Why no LLM?** Fixed warnings do not require open-ended generation. Templates are cheaper, deterministic and easier to test.
33. **What is actually verified now?** The supplied software tests, baseline model/video smoke path, synthetic app run and checkpoint recovery. Real custom training and field/device performance remain pending.
34. **What is the contribution beyond running YOLO?** Conservative geometry, stateful warning selection, asynchronous fresh speech, dataset checks, durable releases and event-level evaluation.
35. **Can it replace a cane?** No. It is an experimental assistive prototype with incomplete coverage and possible errors.

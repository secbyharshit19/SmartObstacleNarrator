# Five-minute demonstration

1. State that this is an experimental assistive prototype. Show the mode label.
2. Start `--baseline` to demonstrate the actual bundled detector, or load your trained custom release. Do not use synthetic mode while claiming detection accuracy.
3. Show a person/chair; move it between image directions and explain unmirrored camera orientation.
4. Hold it still to show cooldown. Add another object and explain priority ordering.
5. Show conditional distance only with a measured object and matching calibration. Otherwise explicitly show “distance unknown.”
6. If a real eight-class model has been trained, demonstrate stairs/pole/barrier/door from held-out recordings. Otherwise explain that these are pending custom-data training; synthetic boxes are only a software preview.
7. Show metrics from your own test run. Software test counts do not equal obstacle detection accuracy.
8. Explain two observed failure cases and the no-path-clear policy.
9. Keep a recorded backup, labeled as recorded, and an offline copy of the model.

Before presenting: run the Windows audio test, verify webcam index, plug in the laptop, disable disruptive notifications, check class mapping, confirm model version, and do a short rehearsal. Do not change thresholds during the scored demonstration.

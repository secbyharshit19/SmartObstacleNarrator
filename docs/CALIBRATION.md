# Conditional distance calibration

The default camera configuration disables meters. A pole/stair box cannot be converted into reliable distance by assuming a generic height.

For a controlled known-object experiment:

1. Fix the actual camera resolution at 640x480, disable digital zoom/cropping, and use a forward orientation.
2. Select one fully visible upright measured reference, such as a particular chair whose overall height is known.
3. Measure camera-to-object axial depth at at least three fit distances and three independent test distances.
4. Save images with the actual camera. Measure full pixel height using `python scripts/measure_box.py image.jpg`. Manual and detector box boundaries must follow the same definition; evaluate detector-derived heights too.
5. Fill a CSV with role,distance_m,pixel_height. Role is fit or test. Do not use the same photos in both roles.
6. Fit the median focal estimate f_y=Z*h/H. The utility checks held-out maximum relative error against 25% by default. Passing is a demo acceptance choice, not professional calibration certification.

```powershell
.\.venv\Scripts\python.exe -m smart_obstacle_narrator.calibration datasets/my_measurements.csv --camera-id 0 --width 640 --height 480 --class-name chair --object-height 0.9 --output configs/camera_measured.yaml
.\.venv\Scripts\python.exe -m smart_obstacle_narrator.app --baseline --calibration configs/camera_measured.yaml
```

The 0.9 above is an example command: replace it with your actual measured height. No measurement values are prefilled. The generated calibrated operating range is limited to the held-out distance range. A failed calibration remains disabled.

Only use the enabled class with objects matching the measured dimensions and pose. The software cannot recognize which chair is your reference chair. Do not enable a class-wide assumed height and then demonstrate arbitrary differently sized objects as though their meters were accurate.

The approximation gives optical-axis depth. It does not directly measure shortest distance, time to collision, or traversability. Clipping, small boxes, wrong camera ID, wrong resolution and out-of-range estimates return unknown. Partial occlusion and tilted posture may still escape these checks. Distortion correction/checkerboard calibration is a useful later refinement.

Evaluate distance MAE AND valid-estimate coverage. Do not report accuracy only for a handpicked easy subset without showing rejected/unknown cases.

# Troubleshooting

| Symptom | Action |
|---|---|
| py -3.11 unavailable | Install Python 3.11; verify `py -0p`; use the installed interpreter's full path |
| son command not found | Use `.venv\Scripts\python.exe -m smart_obstacle_narrator.app` from the project root |
| No module named smart_obstacle_narrator | Install with `python -m pip install -r requirements.txt` using the same venv |
| Missing config file | Change directory to the extracted project root or pass absolute --config/--calibration paths |
| Camera cannot open | Allow camera access in Windows settings; close Teams/browser camera use; try --source 1 |
| Camera reports no frames/disconnect | Reconnect device; restart application; verify camera independently |
| No speech | Check Windows speaker device/volume and installed voices; run scripts/check_environment.py --speak |
| Speech driver hangs | Use --no-audio to isolate; restart; test the selected OS voice; active utterances are not force-killed |
| Custom model missing | Run --baseline now; train the custom model later and copy its complete release |
| Class order mismatch | Use the correct custom model; do not force a COCO model into custom mode |
| Distance always unknown | This is expected without measured calibration. Check ID, resolution, class heights and clipping |
| Frames marked stale on CPU | Measure latency; lower imgsz from 640 to 416 and re-evaluate detection. Do not blindly increase freshness limits |
| CUDA unavailable | Confirm NVIDIA driver and compatible official PyTorch wheel; use --device cpu meanwhile |
| CUDA out of memory in Colab | Lower batch to 4 or 2 for a new run; close other GPU work; do not silently change resume settings |
| Resume hash differs | Data/config changed; restore exact version or start a new run. Generated cache files are excluded |
| Resume says training finished | Start a new fine-tuning run; resume is for interruption recovery |
| Colab cannot see laptop webcam | Run the desktop app locally; use uploads in Colab |
| Archive extra directory | Dataset ZIP must contain images/, labels/, manifest.csv at root |
| Near-duplicate warnings | Inspect the pairs; regroup/remove leakage before acknowledging review |
| OpenCV window error on Linux | Use --headless; do not install headless OpenCV as the Windows desktop runtime |
| MP4 preview fails in browser | Download and play locally or transcode; JSON timestamps remain authoritative |

Only load trusted .pt checkpoints; serialized model files are executable artifacts in the underlying framework. Keep model checksum metadata with every release.

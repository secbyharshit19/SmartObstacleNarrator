# Architecture and interfaces

The runtime package is `src/smart_obstacle_narrator/`. Run commands from the repository root so configuration paths resolve.

| Module | Responsibility |
|---|---|
| types.py | Immutable Detection, Observation, WarningEvent and SpeechFeedback records; ordered class contract |
| config.py | YAML loading and fail-fast validation |
| detector.py | YOLO adapter; explicit COCO-to-project baseline mapping; custom class-order check |
| capture.py | One latest-frame buffer; monotonic capture timestamp; disconnect reporting |
| geometry.py | Camera-relative direction, central overlap, conditional Z=H*f_y/h |
| temporal.py | Greedy one-to-one class/IoU/center matching; expiry; median distance; direction/band stability |
| narration.py | Confirmation, priority, cooldown eligibility and sentence templates |
| pipeline.py | Pure decision orchestration and speech-start acknowledgment |
| speech.py | One speech engine on one worker; active plus one pending event; freshness validation |
| display.py | Bounding boxes, warning/status and keyboard legend |
| app.py | CLI, camera/video/demo modes, controls and integration |
| logging_utils.py | JSONL logs without automatic image recording |
| dataset.py | YOLO audits, grouped split, hashes, near-duplicate flags, video sampling and COCO import |
| calibration.py | Known-height fit with held-out error check |
| training.py | Persistent training, checkpoint hash, resume identity and release metadata |
| release.py | Release checksum and class validation |
| evaluation.py | Timing summaries and optional human-ground-truth matching |

## Coordinates and time

Boxes are x1,y1,x2,y2 in original-frame pixels, not model letterbox coordinates. Sizes are width,height. Unknown distance is None. Clock values are monotonic seconds; they are not wall-clock dates. Direction assumes the camera is unmirrored and faces forward.

## Priority

After confidence and persistence gates:

`100 * (.35*class_weight + .30*proximity + .20*central_overlap + .15*persistence) * (.5+.5*confidence)`

Proximity: unknown .5, far .3, near .7, close 1.0. Unknown does not mean far and never justifies “nearby.” This is a heuristic ordering, not a collision probability. Every value is configurable in app.yaml. Ranking occurs after cooldown eligibility, so a suppressed high-priority object does not starve others.

## Temporal policy

Three observations spanning at least .3 seconds confirm an object. A .5-second gap resets tentative continuity; 1 second without a match expires it. Eight seconds suppresses identical reminders. A stable direction change or closer distance band permits a new event. Direction boundaries use .03 normalized hysteresis and .4-second stability. Distance is a five-value median with conservative uncertainty-based bands and hysteresis. Invalid distance is cleared immediately.

IDs are temporary association IDs, not person identities. Crowded crossings can swap them. Greedy matching is deliberately lightweight and must be evaluated rather than described as robust tracking.

## Speech concurrency

Only the worker touches pyttsx3. The main loop publishes the current relevant-track snapshot and latest eligible event. A speech-start callback posts feedback; the main loop then acknowledges that event's signature. A maximum of one pending warning prevents backlog. Old events, mismatched signatures and stale observations are dropped before speaking. The global gap is measured after an utterance finishes.

Mute/pause clears pending work but cannot guarantee interruption of an active sentence. Driver-start timestamps approximate software onset, not acoustic onset. Engine faults remain visible; there is no “silent success” claim.

## Distance limitations

Bounding boxes alone cannot determine posture, occlusion or true dimensions. Rejecting clipped boxes does not detect all occlusions. Only measured object classes should be enabled for a controlled demonstration. Configured relative uncertainty is an engineering allowance, not a statistical confidence interval. Changing camera, resolution, zoom or crop invalidates the default calibration match.

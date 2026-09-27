"""Desktop overlay, with an explicit simulation/baseline/custom mode indicator."""
import cv2
import numpy as np

SAFETY = 'Experimental prototype. Not a replacement for mobility aids.'

def render(frame, tracks, warning: str, status: str, mode: str, fps: float, paused: bool = False):
    width = max(900, frame.shape[1])
    image = np.full((frame.shape[0], width, 3), (27,24,20), dtype=np.uint8)
    image[:, :frame.shape[1]] = frame
    for t in tracks:
        o = t.observation
        x1,y1,x2,y2 = map(int, o.detection.box)
        color = (80,190,255) if o.distance_band=='close' else (130,220,100)
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
        distance = 'distance unknown' if o.distance_m is None else f'~{o.distance_m:.1f}m'
        label = f'#{t.id} {o.detection.label} {o.detection.confidence:.2f} | {o.direction} | {distance}'
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
        cv2.putText(image, label, (max(0,min(x1,width-text_width-8)),max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    canvas = np.full((image.shape[0]+165,width,3), (27,24,20), dtype=np.uint8)
    canvas[:image.shape[0],:image.shape[1]] = image
    lines = [f'SMART OBSTACLE NARRATOR | {mode} | {fps:.1f} processed FPS',
             ('PAUSED | ' if paused else '') + status, warning or 'Waiting for a stable observation',
             'Q quit   M mute   P pause   R repeat   H help', SAFETY]
    for i,line in enumerate(lines):
        cv2.putText(canvas,line[:130],(16,image.shape[0]+26+i*29),cv2.FONT_HERSHEY_SIMPLEX,
                    .52,(235,235,235) if i != 4 else (120,180,255),1,cv2.LINE_AA)
    return canvas

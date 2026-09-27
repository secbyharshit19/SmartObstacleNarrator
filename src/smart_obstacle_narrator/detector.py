"""Ultralytics adapter. Baseline mapping is explicitly different from custom mode."""
from pathlib import Path
from .types import CLASSES, Detection

BASELINE_MAP = {'person':'person', 'chair':'chair', 'bicycle':'bicycle',
                'car':'vehicle', 'motorcycle':'vehicle', 'bus':'vehicle', 'truck':'vehicle'}

class Detector:
    def __init__(self, weights: str, cfg: dict, baseline: bool = False):
        from ultralytics import YOLO
        import torch
        if not baseline and not Path(weights).is_file():
            raise FileNotFoundError(f'Custom model not found: {weights}. Train first, or use --baseline.')
        if not baseline and (Path(weights).parent/'metadata.json').is_file():
            from .release import verify_release
            if verify_release(Path(weights).parent) != Path(weights).resolve():
                raise ValueError('Selected weights differ from the versioned release metadata')
        self.model = YOLO(weights)
        self.cfg, self.baseline = cfg, baseline
        self.device = ('0' if torch.cuda.is_available() else 'cpu') if cfg['device']=='auto' else cfg['device']
        self.cpu_threads = int(cfg.get('cpu_threads', 2))
        if self.cpu_threads < 1:
            raise ValueError('cpu_threads must be positive')
        if self.device == 'cpu':
            torch.set_num_threads(self.cpu_threads)
        names = self.model.names
        ordered = tuple(names[i] for i in range(len(names))) if isinstance(names, dict) else tuple(names)
        if not baseline and ordered != CLASSES:
            raise ValueError(f'Custom model class order mismatch: {ordered}; expected {CLASSES}')
        if baseline and not set(BASELINE_MAP).issubset(set(ordered)):
            raise ValueError('Baseline mode requires a COCO model with the expected category names')

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(frame, imgsz=self.cfg['imgsz'], conf=self.cfg['confidence'],
                                     device=self.device, verbose=False)
        r = results[0]
        # Ultralytics device setup may set a larger pool on first inference.
        # Restore the configured pool after warm-up to avoid CPU oversubscription.
        if self.device == 'cpu':
            import torch
            torch.set_num_threads(self.cpu_threads)
        if r.boxes is None:
            return []
        boxes = r.boxes.xyxy.cpu().tolist()
        confidences = r.boxes.conf.cpu().tolist()
        ids = r.boxes.cls.cpu().tolist()
        output = []
        for box, confidence, class_id in zip(boxes, confidences, ids):
            label = r.names[int(class_id)]
            if self.baseline:
                label = BASELINE_MAP.get(label)
            if label in CLASSES:
                output.append(Detection(label, float(confidence), tuple(map(float, box))))
        return output

from pathlib import Path
import pytest
import numpy as np
from smart_obstacle_narrator.detector import Detector

def test_real_pretrained_adapter(cfg):
    weights=Path(__file__).parents[1]/'yolo11n.pt'
    if not weights.exists():pytest.skip('Optional baseline weights absent')
    cfg['device']='cpu';cfg['imgsz']=320
    d=Detector(str(weights),cfg,True)
    detections=d.detect(np.zeros((240,320,3),dtype=np.uint8))
    assert isinstance(detections,list)
    assert all(x.label in cfg['classes'] for x in detections)
    with pytest.raises(ValueError):Detector(str(weights),cfg,False)

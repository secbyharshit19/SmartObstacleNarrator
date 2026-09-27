from pathlib import Path
import pytest
from smart_obstacle_narrator.config import load_config

@pytest.fixture
def cfg():
    return load_config(Path(__file__).parents[1]/'configs'/'app.yaml')

@pytest.fixture
def calibration():
    return dict(enabled=True,camera_id='0',width=640,height=480,focal_y_px=400,
                valid_min_m=.5,valid_max_m=5,relative_uncertainty=.1,object_heights_m={'chair':1.0})

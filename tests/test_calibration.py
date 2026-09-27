import pytest
from smart_obstacle_narrator.calibration import calibrate

def measurements(error=0):
    return [dict(role=role,distance_m=z,pixel_height=400/z*(1+error if role=='test' else 1))
            for role in ('fit','test') for z in (1.5,2.5,3.5)]

def test_calibration_and_heldout():
    c=calibrate(measurements(),'0',640,480,'chair',1)
    assert c['enabled'] and c['focal_y_px']==400
    assert c['evaluation']['mae_m']<1e-12

def test_bad_calibration_disabled():
    c=calibrate(measurements(.5),'0',640,480,'chair',1)
    assert not c['enabled']

def test_need_independent_samples():
    with pytest.raises(ValueError):calibrate(measurements()[:3],'0',640,480,'chair',1)

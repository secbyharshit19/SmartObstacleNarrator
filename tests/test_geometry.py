import pytest
from smart_obstacle_narrator.geometry import direction,band,DistanceEstimator,observe
from smart_obstacle_narrator.types import Detection

@pytest.mark.parametrize('center,expected',[(.1,'left'),(.35,'ahead'),(.5,'ahead'),(.65,'ahead'),(.9,'right')])
def test_direction_boundaries(cfg,center,expected):
    x=center*640
    assert direction((x-10,20,x+10,40),640,cfg)==expected

def test_hysteresis(cfg):
    assert direction((220,20,240,40),640,cfg,'left')=='left'
    assert direction((220,20,240,40),640,cfg)=='ahead'

def test_depth(calibration):
    z,reason=DistanceEstimator(calibration,'0').estimate(Detection('chair',.9,(100,100,200,300)),(640,480))
    assert z==2
    assert 'approximation' in reason

@pytest.mark.parametrize('label,box,size,camera,reason',[
    ('person',(100,100,200,300),(640,480),'0','unknown object height'),
    ('chair',(0,100,200,300),(640,480),'0','clipped object'),
    ('chair',(100,100,200,110),(640,480),'0','box too small'),
    ('chair',(100,100,200,300),(1280,960),'0','resolution mismatch'),
    ('chair',(100,100,200,300),(640,480),'1','camera mismatch'),
    ('chair',(100,100,200,121),(640,480),'0','outside calibrated range')])
def test_unknown_cases(calibration,label,box,size,camera,reason):
    z,why=DistanceEstimator(calibration,camera).estimate(Detection(label,.9,box),size)
    assert z is None and why==reason

def test_disabled():
    assert DistanceEstimator({'enabled':False},'0').estimate(Detection('chair',.9,(1,1,2,2)),(640,480))[0] is None

def test_uncertainty_avoids_close(cfg):
    assert band(1.4,cfg,.25)=='near'
    assert band(None,cfg)=='unknown'

def test_wide_obstacle_central_overlap(cfg):
    o=observe(Detection('barrier',.9,(0,100,500,300)),(640,480),cfg,DistanceEstimator({'enabled':False},'0'))
    assert o.central_overlap==1

def test_reject_invalid_calibration(calibration):
    calibration['focal_y_px']=float('nan')
    with pytest.raises(ValueError):DistanceEstimator(calibration,'0')

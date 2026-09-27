from smart_obstacle_narrator.pipeline import Pipeline
from smart_obstacle_narrator.types import Detection
import pytest

D=Detection('person',.9,(250,50,390,420))

def stable(p,detections,start=0):
    p.process(detections,start,(640,480))
    p.process(detections,start+.2,(640,480))
    return p.process(detections,start+.4,(640,480))

def test_persistence_cooldown_and_reset(cfg):
    p=Pipeline(cfg,{'enabled':False})
    assert p.process([D],0,(640,480))[1] is None
    assert p.process([D],.2,(640,480))[1] is None
    tracks,event=p.process([D],.4,(640,480))
    assert event.text=='Person ahead.'
    p.acknowledge(event,.4)
    assert p.process([D],.6,(640,480))[1] is None
    p.process([],2,(640,480))
    _,again=stable(p,[D],3)
    assert again and again.track_id!=event.track_id

def test_lower_priority_not_starved(cfg):
    p=Pipeline(cfg,{'enabled':False})
    ds=[D,Detection('door',.9,(30,50,150,440))]
    _,first=stable(p,ds)
    assert first.text.startswith('Person')
    p.acknowledge(first,.4)
    _,second=p.process(ds,.6,(640,480))
    assert second and second.text.startswith('Door')

def test_multiple_same_class_kept_separate(cfg):
    p=Pipeline(cfg,{'enabled':False})
    ds=[Detection('person',.9,(10,10,100,400)),Detection('person',.9,(140,10,220,400))]
    tracks,_=stable(p,ds)
    assert len({t.id for t in tracks})==2

def test_low_confidence_and_invalid_boxes(cfg):
    p=Pipeline(cfg,{'enabled':False})
    ds=[Detection('person',.1,D.box),Detection('stairs',.9,(-10,10,50,50)),Detection('door',.9,(2,2,float('nan'),100))]
    assert stable(p,ds)[0]==[]

def test_no_distance_language_without_calibration(cfg):
    _,e=stable(Pipeline(cfg,{'enabled':False}),[Detection('pole',.99,(100,10,200,470))])
    assert e.text=='Pole on your left.'

def test_short_gap_does_not_instantly_confirm(cfg):
    p=Pipeline(cfg,{'enabled':False})
    p.process([D],0,(640,480))
    p.process([D],.1,(640,480))
    assert p.process([D],.8,(640,480))[1] is None

def test_acknowledged_only_on_start(cfg):
    p=Pipeline(cfg,{'enabled':False})
    _,e=stable(p,[D])
    assert p.process([D],.5,(640,480))[1] is not None
    p.acknowledge(e,.5)
    assert p.process([D],.6,(640,480))[1] is None

def test_monotonic_time(cfg):
    p=Pipeline(cfg,{'enabled':False});p.process([],1,(640,480))
    with pytest.raises(ValueError):p.process([],0,(640,480))

def test_stable_direction_change(cfg):
    p=Pipeline(cfg,{'enabled':False})
    d=Detection('person',.9,(180,50,260,420))
    _,e=stable(p,[d]);p.acknowledge(e,.4)
    d2=Detection('person',.9,(240,50,320,420))
    assert p.process([d2],.5,(640,480))[1] is None
    p.process([d2],.7,(640,480))
    _,changed=p.process([d2],1.0,(640,480))
    assert changed and changed.text=='Person ahead.'

def test_closer_band_reannounces(cfg,calibration):
    p=Pipeline(cfg,calibration)
    far=Detection('chair',.9,(250,100,350,200))
    _,e=stable(p,[far]);p.acknowledge(e,.4)
    near=Detection('chair',.9,(230,80,370,440))
    events=[]
    for t in [.5,.7,.9,1.1,1.3,1.5]:
        _,ev=p.process([near],t,(640,480))
        if ev:events.append(ev)
    assert any('very close' in ev.text for ev in events)

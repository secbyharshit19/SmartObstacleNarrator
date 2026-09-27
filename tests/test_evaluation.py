import json
from smart_obstacle_narrator.evaluation import evaluate

def test_event_matching_and_missing_distance(tmp_path):
    records=[dict(kind='session',mode='SYNTHETIC DEMO',audible=False),
             dict(kind='frame',processed=10,captured=10,inference_ms=1,frame_age_ms=2),
             dict(kind='frame',processed=12,captured=12,inference_ms=3,frame_age_ms=4),
             dict(kind='speech_started',timestamp=10.5,event=dict(track_id=1,event_id='1',created=10.4,text='Person ahead.',signature=['ahead','unknown']))]
    log=tmp_path/'log.jsonl';log.write_text('\n'.join(map(json.dumps,records)))
    truth=tmp_path/'truth.csv';truth.write_text('start_s,end_s,label,direction\n0,1,person,ahead\n1,2,pole,left\n')
    distances=tmp_path/'distances.csv';distances.write_text('actual_m,estimated_m\n2,2.5\n3,\n')
    result=evaluate(log,truth,distances)
    assert result['events']['matched']==1 and result['events']['missed']==1
    assert result['distance']['coverage']==.5 and result['distance']['mae_m']==.5

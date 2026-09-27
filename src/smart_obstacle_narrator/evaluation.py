"""Observed software timing and optional human-ground-truth evaluation."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
import numpy as np

def distribution(values):
    return {'n':len(values),'median':float(np.median(values)) if values else None,
            'p95':float(np.percentile(values,95)) if values else None}

def evaluate(log: Path, truth: Path | None = None, distances: Path | None = None,
             directions: Path | None = None) -> dict:
    records=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
    frames=[r for r in records if r['kind']=='frame']
    spoken=[r for r in records if r['kind']=='speech_started' and r['event']['track_id']!=-1]
    session=next((r for r in records if r['kind']=='session'),{})
    duration=(frames[-1]['processed']-frames[0]['processed']) if len(frames)>1 else 0
    result={'mode':session.get('mode'),'audible':session.get('audible'), 'frames':len(frames),
            'processed_fps':(len(frames)-1)/duration if duration>0 else None,
            'inference_ms':distribution([r['inference_ms'] for r in frames]),
            'frame_age_ms':distribution([r['frame_age_ms'] for r in frames]),
            'queue_to_driver_start_ms':distribution([1000*(r['timestamp']-r['event']['created']) for r in spoken]),
            'warning_count':len(spoken),'warnings_per_min':len(spoken)*60/duration if duration>0 else None,
            'stale_frames':sum(r['kind']=='stale_frame' for r in records),
            'note':'Driver start is not measured acoustic onset. Synthetic/silent runs do not measure model/audio quality.'}
    if truth:
        events=list(csv.DictReader(truth.open(encoding='utf-8',newline='')))
        origin=frames[0]['captured'] if frames else 0
        matches={}; false=0; latency=[]
        for r in spoken:
            e=r['event'];t=r['timestamp']-origin;label=e['text'].split()[0].lower();direction=e['signature'][0]
            candidates=[(i,g) for i,g in enumerate(events) if i not in matches and g['label']==label and
                        g['direction']==direction and float(g['start_s'])<=t<=float(g['end_s'])]
            if candidates:
                i,g=min(candidates,key=lambda item:float(item[1]['end_s']))
                matches[i]=e['event_id'];latency.append(1000*(t-float(g['start_s'])))
            else:false+=1
        result['events']={'ground_truth':len(events),'matched':len(matches),'missed':len(events)-len(matches),
                          'false_warnings':false,'recall':len(matches)/len(events) if events else None,
                          'false_warning_fraction':false/len(spoken) if spoken else None,
                          'false_warnings_per_min':false*60/duration if duration>0 else None,
                          'event_to_driver_start_ms':distribution(latency)}
    if distances:
        rows=list(csv.DictReader(distances.open(encoding='utf-8',newline='')))
        errors=[];relative=[]
        for row in rows:
            actual=float(row['actual_m'])
            if not math.isfinite(actual) or actual<=0:raise ValueError('Actual distances must be positive')
            if row.get('estimated_m','').strip():
                estimated=float(row['estimated_m'])
                if not math.isfinite(estimated) or estimated<=0:raise ValueError('Invalid estimate')
                errors.append(abs(actual-estimated));relative.append(abs(actual-estimated)/actual)
        result['distance']={'total':len(rows),'estimated':len(errors),'coverage':len(errors)/len(rows) if rows else None,
                            'mae_m':mean(errors) if errors else None,'median_absolute_error_m':float(np.median(errors)) if errors else None,
                            'mean_relative_error':mean(relative) if relative else None}
    if directions:
        rows=list(csv.DictReader(directions.open(encoding='utf-8',newline='')))
        valid={'left','ahead','right'}
        if any(r['actual'] not in valid or r['predicted'] not in valid for r in rows):raise ValueError('Invalid direction label')
        result['direction']={'n':len(rows),'accuracy':mean(r['actual']==r['predicted'] for r in rows) if rows else None}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path)
    p.add_argument('--truth',type=Path);p.add_argument('--distances',type=Path);p.add_argument('--directions',type=Path)
    p.add_argument('--output',type=Path,default=Path('results/evaluation.json'))
    a=p.parse_args();result=evaluate(a.log,a.truth,a.distances,a.directions)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

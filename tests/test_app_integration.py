"""Run the real application loop with synthetic detections and a silent speech worker."""
import json
from pathlib import Path
import subprocess
import sys

def test_headless_application_logs_warning(tmp_path):
    root=Path(__file__).parents[1]
    log=tmp_path/'events.jsonl'
    result=subprocess.run([sys.executable,'-m','smart_obstacle_narrator.app','--demo','--no-audio',
                           '--headless','--seconds','1.2','--log',str(log)],cwd=root,
                           capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    records=[json.loads(x) for x in log.read_text().splitlines()]
    assert any(r['kind']=='speech_started' and r['event']['text']=='Person ahead.' for r in records)
    assert records[-1]['kind']=='summary'

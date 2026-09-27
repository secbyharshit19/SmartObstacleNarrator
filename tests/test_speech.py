import threading
import time
from smart_obstacle_narrator.speech import SpeechWorker,SilentBackend
from smart_obstacle_narrator.types import WarningEvent

def event(i=1,expires=1):
    now=time.monotonic()
    return WarningEvent(str(i),i,'Person ahead.',50,now,now+expires,('ahead','unknown'))

def test_feedback_and_worker_shutdown():
    w=SpeechWorker(0,.5,SilentBackend);assert w.ready.wait(1)
    e=event();w.update({1:(e.signature,e.created)},e)
    assert w.feedback.get(timeout=1).kind=='started'
    assert w.feedback.get(timeout=1).kind=='finished'
    w.close();assert not w.thread.is_alive()

def test_expired_not_spoken():
    w=SpeechWorker(0,.5,SilentBackend);w.ready.wait(1)
    e=event(expires=-1);w.update({1:(e.signature,e.created)},e)
    assert w.feedback.get(timeout=1).kind=='dropped'
    w.close()

def test_obsolete_signature_dropped():
    w=SpeechWorker(0,.5,SilentBackend);w.ready.wait(1)
    e=event();w.update({1:(('left','unknown'),e.created)},e)
    assert w.feedback.get(timeout=1).kind=='dropped'
    w.close()

def test_queue_is_bounded_while_speaking():
    started=threading.Event();finish=threading.Event()
    class Blocking:
        def say(self,text,on_start):
            on_start();started.set();finish.wait(1)
        def close(self):pass
    w=SpeechWorker(0,.5,Blocking);w.ready.wait(1)
    e=event();w.update({1:(e.signature,e.created)},e);assert started.wait(1)
    for i in range(2,30):
        pending=event(i);w.update({i:(pending.signature,pending.created)},pending)
    assert w.pending.track_id==29
    w.set_muted(True);assert w.pending is None
    finish.set();w.close()

def test_backend_failure_visible():
    def fail():raise RuntimeError('audio unavailable')
    w=SpeechWorker(0,.5,fail);w.ready.wait(1)
    assert w.status=='error' and 'audio unavailable' in w.error
    w.close()

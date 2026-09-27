"""Synthetic boxes for deterministic software verification, never model evidence."""
from .types import Detection

def demo_detections(elapsed: float) -> list[Detection]:
    phase = elapsed % 16
    if phase < 4:
        return [Detection('person', .9, (270,80,370,390))]
    if phase < 8:
        return [Detection('person', .9, (55,80,155,390)), Detection('chair', .85, (280,230,400,420))]
    if phase < 10:
        return []
    if phase < 13:
        return [Detection('stairs', .93, (200,240,530,460))]
    return [Detection('pole', .92, (500,60,535,430)), Detection('barrier', .9, (200,240,440,430))]

"""Verify the weights and class contract before loading a custom release."""
import json
from pathlib import Path
from .dataset import digest
from .types import CLASSES

def verify_release(directory: Path) -> Path:
    meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if tuple(meta['classes'])!=CLASSES:raise ValueError('Release class order mismatch')
    weights=(directory/meta['weight_file']).resolve()
    if not weights.is_relative_to(directory.resolve()):raise ValueError('Invalid weight path')
    if digest(weights)!=meta['weight_sha256']:raise ValueError('Model checksum mismatch')
    if meta.get('fixture_only'):raise ValueError('Fixture-only smoke model is not a custom release')
    return weights

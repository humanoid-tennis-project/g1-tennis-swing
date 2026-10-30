"""Small, dependency-light recording plan helpers."""
import hashlib
import json
from pathlib import Path

REVISION = "4af17dd4d398aad4116a35b5a74de4388658d356"
GVHMR_REVISION = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def motion_windows(frames, core_frames=900, context_frames=60):
    """Nonoverlapping ownership + inference context; no stroke boundaries."""
    if frames < 2 or core_frames < 2 or context_frames < 0:
        raise ValueError("Invalid frame plan")
    rows = []
    for start in range(0, frames, core_frames):
        end = min(frames, start + core_frames)
        rows.append(dict(index=len(rows), core_start=start, core_end=end,
                         input_start=max(0, start-context_frames),
                         input_end=min(frames, end+context_frames)))
    # A one-frame tail cannot independently support all official postprocessing.
    if len(rows) > 1 and rows[-1]["core_end"]-rows[-1]["core_start"] == 1:
        tail = rows.pop()
        rows[-1].update(core_end=tail["core_end"], input_end=tail["input_end"])
    return rows

"""Experiment tracking behind one small interface: trackio or wandb (same API), or nothing.

Everything logged here is also written to `train_log.jsonl` / `metrics.json` in the run directory,
so a tracker is a convenience for comparing runs, never the only record.
"""
from typing import Dict, Optional

TRACKERS = ("none", "trackio", "wandb")


class Tracker:
    def __init__(self, kind: str, project: str, name: Optional[str], config: Dict,
                 space: Optional[str] = None, enabled: bool = True):
        self.kind, self.mod, self.run = kind, None, None
        if kind == "none" or not enabled:
            return
        if kind not in TRACKERS:
            raise ValueError("tracker must be one of %s, got %r" % (TRACKERS, kind))
        try:
            mod = __import__(kind)
        except ImportError:
            raise ImportError("tracker %r needs the `%s` package: pip install %s" % (kind, kind, kind))
        kw = {"project": project, "name": name, "config": config}
        if space:
            if kind == "trackio":
                kw["space_id"] = space
            else:
                kw["entity"] = space
        self.run = mod.init(**kw)
        self.mod = mod

    def log(self, metrics: Dict, step: Optional[int] = None):
        if self.mod is not None:
            self.mod.log(metrics, step=step)

    def finish(self):
        if self.mod is not None:
            self.mod.finish()
            self.mod = None

    @property
    def url(self) -> Optional[str]:
        return getattr(self.run, "url", None) if self.run is not None else None


def flat(prefix: str, d: Dict) -> Dict:
    """{"a": 1, "b": {"c": 2}} -> {"prefix/a": 1, "prefix/b/c": 2}; non-numeric leaves are dropped."""
    out = {}
    for k, v in d.items():
        key = "%s/%s" % (prefix, k)
        if isinstance(v, dict):
            out.update(flat(key, v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out

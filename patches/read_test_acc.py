#!/usr/bin/env python3
"""Extract test_accuracy from the latest .pt checkpoint without loading model weights.

.pt files are zip archives. Only data.pkl (structure + metadata, ~15KB) is read;
the large tensor blobs (data/0, data/1, ...) are never touched.
Usage: python3 read_test_acc.py <checkpoint_dir>
"""
import sys, zipfile, pickle, glob, os


class _Stub:
    def __call__(self, *a, **kw): return self
    def __setitem__(self, k, v): pass


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if "torch" in m:
            return lambda *a, **kw: _Stub()
        return super().find_class(m, n)

    def persistent_load(self, pid):
        return _Stub()


def read_acc(ckpt_dir: str) -> str:
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not pts:
        return "-"
    try:
        with zipfile.ZipFile(pts[-1]) as z:
            name = z.namelist()[0].split("/")[0]
            d = _U(z.open(f"{name}/data.pkl")).load()
        m = d.get("metrics", {}) or {}
        v = m.get("test_accuracy")
        return f"{float(v):.4f}" if v is not None else "-"
    except Exception:
        return "-"


if __name__ == "__main__":
    print(read_acc(sys.argv[1]) if len(sys.argv) > 1 else "-")

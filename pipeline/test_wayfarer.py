"""Self-check for pipeline helpers. Run: python pipeline/test_wayfarer.py"""
import numpy as np

from wayfarer import stray_cameras

rng = np.random.default_rng(0)
# a 16 m long walk (corridor-like, far ends are legit) plus a few cameras the mapper threw out
walk = np.c_[np.linspace(0, 16, 300), rng.normal(0, 1, 300), np.full(300, 1.4)]
stray = np.array([[457, 3, 1], [2e7, 0, 0], [-300, 50, 2]])
bad = stray_cameras(np.r_[walk, stray])
assert sorted(bad) == [300, 301, 302], bad
assert len(stray_cameras(walk)) == 0
print('ok: stray cameras dropped, long walk kept')

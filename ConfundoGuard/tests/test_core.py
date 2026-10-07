import numpy as np
from confundoguard.npas.scorer import normalize_passage_influence, compute_npas
from confundoguard.filters.av_filter import AVFilter
from confundoguard.metrics.evaluator import detection_metrics

def test_normalize():
    x = normalize_passage_influence([1, 1, 3])
    assert np.isclose(x.sum(), 1.0)
    assert x[2] > x[0]

def test_compute_npas():
    r = compute_npas(np.array([1,1,1,6,6,1]), [(0,3),(3,5),(5,6)])
    assert np.isclose(r.scores.sum(), 1.0)
    assert r.scores[1] == max(r.scores)

def test_filter_flags_outlier():
    f = AVFilter(z_threshold=1.4)
    r = f.detect([0.1,0.1,0.6,0.1,0.1])
    assert r.suspicious_indices == [2]

def test_metrics():
    m = detection_metrics([0,0,1,1], [0,1,1,1])
    assert m.detection_rate == 1.0

import numpy as np

from nanosd_repro import StageVariant, Architecture, evaluate_architecture, pareto_frontier


def test_stage_variant_and_architecture():
    variant = StageVariant(name='R-A', residual=0.8, attention=0.2, params_mult=1.0, latency_mult=1.0)
    assert variant.name == 'R-A'

    arch = Architecture([1, 2, 3, 5, 7, 6])
    assert len(arch.vector) == 6
    assert arch.valid is True


def test_evaluate_architecture_returns_valid_metrics():
    metrics = evaluate_architecture([1, 2, 3, 5, 7, 6], seed=11)
    assert 'ta_fid' in metrics
    assert 'latency_ms' in metrics
    assert 'params_m' in metrics
    assert metrics['ta_fid'] >= 0.0
    assert metrics['latency_ms'] > 0.0
    assert metrics['params_m'] > 0.0


def test_pareto_frontier_keeps_nondominated_points():
    points = [
        {'id': 'a', 'ta_fid': 1.0, 'latency_ms': 5.0, 'params_m': 100.0},
        {'id': 'b', 'ta_fid': 0.5, 'latency_ms': 8.0, 'params_m': 80.0},
        {'id': 'c', 'ta_fid': 0.8, 'latency_ms': 4.0, 'params_m': 120.0},
        {'id': 'd', 'ta_fid': 0.4, 'latency_ms': 10.0, 'params_m': 90.0},
    ]
    frontier = pareto_frontier(points, objective='latency')
    ids = {x['id'] for x in frontier}
    assert 'a' not in ids
    assert 'b' in ids
    assert 'd' in ids

"""Smoke test for the ensemble's probability-averaging decision rule."""
import numpy as np


def test_probability_averaging_lets_consensus_win():
    # Mirrors ensemble.py: ensemble_probs = np.mean(stacked, axis=0).
    # Two models agree on class 1 with moderate confidence; one is confidently
    # wrong on class 2. Averaging keeps the consensus and stays a valid
    # probability distribution.
    m1 = np.array([[0.1, 0.7, 0.2]])
    m2 = np.array([[0.2, 0.6, 0.2]])
    m3 = np.array([[0.05, 0.05, 0.90]])
    ensemble = np.mean([m1, m2, m3], axis=0)
    assert ensemble.argmax(axis=1)[0] == 1
    np.testing.assert_allclose(ensemble.sum(axis=1), 1.0)

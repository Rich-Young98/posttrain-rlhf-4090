"""Unit tests for variance sampler weights (no torch needed)."""
import json

from variance_sampler import compute_weights, load_pass_rates


def test_band_filter():
    rates = [0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0, None]
    w = compute_weights(rates, p_low=0.1, p_high=0.9)
    assert w[0] == 0.0 and w[1] == 0.0 and w[5] == 0.0 and w[6] == 0.0 and w[7] == 0.0
    assert w[2] > 0 and w[3] > 0 and w[4] > 0
    # 0.5 should have the largest weight (4p(1-p)=1)
    assert w[3] == max(w)


def test_all_zero_falls_back_to_uniform():
    w = compute_weights([0.0, 1.0, None], p_low=0.1, p_high=0.9)
    assert w == [1.0, 1.0, 1.0]


def test_power():
    w1 = compute_weights([0.5], power=1.0)[0]
    w2 = compute_weights([0.5], power=2.0)[0]
    assert abs(w1 - 1.0) < 1e-9
    assert abs(w2 - 1.0) < 1e-9
    w3 = compute_weights([0.2], power=1.0)[0]
    w4 = compute_weights([0.2], power=2.0)[0]
    assert w4 < w3


def test_load_pass_rates(tmp_path=None):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"0": 0.1, "2": 0.5}, f)
        path = f.name
    try:
        out = load_pass_rates(path, 3, default=0.25)
        assert out == [0.1, 0.25, 0.5]
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_band_filter()
    test_all_zero_falls_back_to_uniform()
    test_power()
    test_load_pass_rates()
    print("all tests passed")

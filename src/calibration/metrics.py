import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
            confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin

    return float(ece)


def brier_score(y_true, y_prob):
    return float(brier_score_loss(y_true, y_prob))


def reliability_bins(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

        if np.any(in_bin):
            bins.append({
                "bin": i + 1,
                "lower": float(bin_lower),
                "upper": float(bin_upper),
                "count": int(np.sum(in_bin)),
                "accuracy": float(np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))),
                "confidence": float(np.mean(y_prob[in_bin]))
            })
        else:
            bins.append({
                "bin": i + 1,
                "lower": float(bin_lower),
                "upper": float(bin_upper),
                "count": 0,
                "accuracy": None,
                "confidence": None
            })

    return bins
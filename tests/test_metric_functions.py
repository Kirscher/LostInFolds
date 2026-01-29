
import numpy as np

from src.ensemble_metrics.metric_functions import (
    calib_stats, get_correct_binary_multirater, get_max_prob_for_pred_classes)


def test_calib_stats():
    samples = np.array([[0.78, 0.22],
                    [0.36, 0.64],
                    [0.08, 0.92],
                    [0.58, 0.42],
                    [0.49, 0.51],
                    [0.85, 0.15],
                    [0.30, 0.70],
                    [0.63, 0.37],
                    [0.17, 0.83]])
    true_labels = np.array([0,1,0,0,0,0,1,1,1])
    confidences = np.max(samples, axis=1)
    predicted_labels = np.argmax(samples, axis=1)
    correct = (predicted_labels == true_labels).astype(int)
    expected_discrepancies = np.array([0.045,  0.0625, 0.2])
    expected_prob_total = np.array([2/9, 4/9, 3/9])
    expected_nonzero = 3
    bin_discrepancies, prob_total, num_nonzero = calib_stats(correct, confidences, n_bins=5)
    assert np.allclose(bin_discrepancies, expected_discrepancies, atol=1e-3), "Calibration discrepancies do not match expected values."
    assert np.allclose(prob_total, expected_prob_total, atol=1e-3), "Probability totals do not match expected values."
    assert num_nonzero == expected_nonzero, "Number of non-zero bins does not match expected value."


def test_get_correct_binary_multirater():
    gt_raters = np.array([
        [[0, 1], [1, 0]],
        [[0, 0], [1, 1]],
        [[1, 1], [1, 0]]
    ])
    pred = np.array([[0, 1], [0, 0]])
    expected_correct = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1])
    correct = get_correct_binary_multirater(gt_raters, pred)
    assert np.array_equal(correct, expected_correct), "Binary correctness array does not match expected values."


def test_get_max_prob_for_pred_classes(): 
    probs_per_fold = {
        0: np.array([[[0.2, 0.1], [0.5, 0.4]],
                     [[0.2, 0.7], [0.3, 0.4]],
                     [[0.6, 0.2], [0.2, 0.2]]]),
        1: np.array([[[0.3, 0.2], [0.4, 0.3]],
                     [[0.3, 0.6], [0.4, 0.5]],
                     [[0.4, 0.2], [0.2, 0.2]]])
    }
    consensus_pred = np.array([[2, 1], [0, 1]])
    expected_max_probs = np.array([[0.6, 0.7], [0.5, 0.5]])
    max_probs = get_max_prob_for_pred_classes(probs_per_fold, consensus_pred)
    assert np.allclose(max_probs, expected_max_probs, atol=1e-3), "Max probabilities for predicted classes do not match expected values."


if __name__ == "__main__":
    test_calib_stats()
    test_get_correct_binary_multirater()
    test_get_max_prob_for_pred_classes()
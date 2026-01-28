import numpy as np

from src.ensemble_metrics.utils import calculate_majority_consensus


def test_majority_consensus_2d():
    rater1 = np.array([[0, 1, 2], [0, 0, 1], [1, 1, 0]])
    rater2 = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]])
    rater3 = np.array([[1, 1, 2], [0, 0, 1], [1, 1, 1]])
    raters = [rater1, rater2, rater3]
    expected_consensus = np.array([[0, 1, 2], [0, 0, 1], [1, 1, 0]])
    consensus = calculate_majority_consensus(raters)
    assert np.array_equal(consensus, expected_consensus), "Majority consensus calculation failed."


def test_majority_consensus_3d():
    rater1 = np.array([[[0, 1], [2, 0]], [[0, 1], [1, 1]]])
    rater2 = np.array([[[0, 1], [1, 0]], [[0, 1], [0, 1]]])
    rater3 = np.array([[[1, 1], [2, 0]], [[0, 0], [1, 1]]])
    raters = [rater1, rater2, rater3]
    expected_consensus = np.array([[[0, 1], [2, 0]], [[0, 1], [1, 1]]])
    consensus = calculate_majority_consensus(raters)
    assert np.array_equal(consensus, expected_consensus), "Majority consensus calculation failed."
    
    
if __name__ == "__main__":
    test_majority_consensus_2d()
    test_majority_consensus_3d()
    print("All tests passed.")
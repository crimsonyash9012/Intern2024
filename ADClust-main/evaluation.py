import numpy as np
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    assert y_pred.size == y_true.size, "The size of y_pred and y_true must be equal."
    n_samples = y_pred.size
    n_classes = max(y_pred.max(), y_true.max()) + 1

    # Create a contingency table
    contingency_table = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_samples):
        contingency_table[y_pred[i], y_true[i]] += 1

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(contingency_table.max() - contingency_table)
    optimal_match = contingency_table[row_ind, col_ind].sum()

    accuracy = optimal_match / n_samples
    return accuracy

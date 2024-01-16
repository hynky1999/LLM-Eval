import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates the accuracy of the model using torchmetrics.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy
    """
    # the number of classes is inferred from the y_true
    # First coallesce Nones
    y_true = np.array([y if y is not None else "None" for y in y_true])
    y_pred = np.array([y if y is not None else "None" for y in y_pred])
    accuracy = np.mean(y_true == y_pred)
    return accuracy

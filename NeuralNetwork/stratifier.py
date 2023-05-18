from typing import Optional, Dict, Tuple

from perceptron import Perceptron

import numpy as np

import random


def stratify(
    x_data: np.ndarray,
    y_data: np.ndarray,
    hyperparams: Dict,
    k=10,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    if seed is not None:
        np.random.seed(seed)

    data_size = x_data.shape[0]

    indices = list(range(data_size))
    random.shuffle(indices)

    x_data, y_data = x_data[indices], y_data[indices]

    fold_size = data_size // k

    each_fold_indices = [
        set(indices[k * fold_size : (k + 1) * fold_size]) for k in range(k)
    ]

    indices = set(indices)

    accuracies = []
    f1_scores = []

    num_classes = y_data.shape[-1]

    for fold_indices in each_fold_indices:
        train_indices = list(indices - fold_indices)
        eval_indices = list(fold_indices)

        x_train, y_train = x_data[train_indices], y_data[train_indices]
        x_eval, y_eval = x_data[eval_indices], y_data[eval_indices]

        net = Perceptron(**hyperparams["net"])
        net.train(
            train_x=x_train,
            train_y=y_train,
            eval_x=x_eval,
            eval_y=y_eval,
            **hyperparams["train"],
            console_log=False,
            show_progress_bar=True,
            train_name=f"Fold {len(accuracies) + 1}/{k}",
        )

        accuracy, f1_score = validate(net, x_eval, y_eval, num_classes=num_classes)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)

    return float(np.mean(accuracies)), float(np.mean(f1_scores))


def make_confusion_matrix(
    pred_class: np.ndarray, true_class: np.ndarray, num_classes: int
) -> np.ndarray:
    confusion = np.zeros([num_classes, num_classes], dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            true_class_is_i = true_class == i
            true_class_i_pred_class_j = pred_class[true_class_is_i] == j
            confusion[i, j] = true_class_i_pred_class_j.sum()

    return confusion


def validate(
    net: Perceptron, x_eval: np.ndarray, y_eval: np.ndarray, num_classes: int
) -> Tuple[float, float]:
    y_pred = net.forward(x_eval).argmax(axis=-1)
    y_true = y_eval.argmax(axis=-1)

    confusion = make_confusion_matrix(y_pred, y_true, num_classes=num_classes)

    class_indices = set(range(confusion.shape[0]))

    f1_scores = []
    accuracies = []
    for i in class_indices:
        other_indices = class_indices - {i}
        true_positive = confusion[i, i]
        false_positive = confusion[list(other_indices), i].sum()
        false_negative = confusion[i, list(other_indices)].sum()
        true_negative = confusion.sum() - (
            true_positive + false_negative + false_positive
        )

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1_score)

        accuracy = (true_positive + true_negative) / confusion.sum()
        accuracies.append(accuracy)

    mean_f1 = float(np.mean(f1_scores))
    mean_acc = float(np.mean(accuracies))

    return mean_acc, mean_f1

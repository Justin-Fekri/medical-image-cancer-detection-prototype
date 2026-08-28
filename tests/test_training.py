import numpy as np

from train import SyntheticDataset, binary_metrics, stratified_split


def test_stratified_split_preserves_both_classes():
    dataset = SyntheticDataset(n_per_class=10, seed=3)
    train_set, validation_set = stratified_split(dataset, validation_fraction=0.2, seed=3)
    assert len(train_set) == 16
    assert len(validation_set) == 4
    validation_labels = [int(dataset.pairs[index][1]) for index in validation_set.indices]
    assert sorted(validation_labels) == [0, 0, 1, 1]


def test_binary_metrics_for_separable_predictions():
    metrics = binary_metrics(
        probabilities=np.asarray([0.05, 0.2, 0.8, 0.95]),
        targets=np.asarray([0, 0, 1, 1]),
    )
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0

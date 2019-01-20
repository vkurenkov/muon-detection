import numpy  as np
import pandas as pd
import utils

from sklearn.base            import ClassifierMixin
from sklearn.metrics         import make_scorer
from sklearn.model_selection import StratifiedKFold


def find_threshold_for_efficiency(a, e, w):
    if e < 0 or e > 1:
        raise ValueError("Efficiency e must be in [0, 1]")

    # Decreasing order
    idx = np.argsort(a)[::-1]
    a_sort = a[idx]
    if w is None:
        w = np.ones(a.shape)
        
    w_sort = w[idx]
    ecdf = np.cumsum(w_sort)
    if (ecdf[-1]) <= 0:
        raise ValueError("Total weight is < 0")

    target_weight_above_threshold = e * ecdf[-1]
    enough_passing = ecdf >= target_weight_above_threshold
    first_suitable = np.argmax(enough_passing)
    last_unsuitable_inv = np.argmin(enough_passing[::-1])

    if last_unsuitable_inv == 0:
        raise ValueError("Bug in code")
    last_unsuitable_plus = len(a) - last_unsuitable_inv
    return 0.5*(a_sort[first_suitable] + a_sort[last_unsuitable_plus])


def get_rejection_at_efficiency_raw(
        labels, predictions, weights, quantile):
    signal_mask = (labels >= 1)
    background_mask = ~signal_mask
    
    if weights is None:
        signal_weights = None
    else:
        signal_weights = weights[signal_mask]
    threshold = find_threshold_for_efficiency(predictions[signal_mask], 
                                              quantile, signal_weights)
    rejected_indices = (predictions[background_mask] < threshold)
    
    if weights is not None:
        rejected_background = weights[background_mask][rejected_indices].sum()
        weights_sum = np.sum(weights[background_mask])
    else:
        rejected_background = rejected_indices.sum()
        weights_sum = np.sum(background_mask)
        
    return rejected_background, weights_sum         


def get_rejection_at_efficiency(labels, predictions, threshold, sample_weight=None):
    rejected_background, weights_sum = get_rejection_at_efficiency_raw(
        labels, predictions, sample_weight, threshold)
    
    return rejected_background / weights_sum


def rejection90(labels, predictions, sample_weight=None):
    return get_rejection_at_efficiency(labels, predictions, 0.9, sample_weight=sample_weight)


rejection90_sklearn = make_scorer(
    get_rejection_at_efficiency, needs_threshold=True, threshold=0.9)


def cross_validate(model: ClassifierMixin, dataset: pd.DataFrame, cv=3) -> None:
    x = dataset.drop("label", axis=1)
    y = dataset["label"]

    skf = StratifiedKFold(n_splits=cv)
    
    train_scores = []
    test_scores  = []
    for train, test in skf.split(x, y):
        train_x       = x[train].values
        train_y       = y[train].values
        train_weights = train_x.weight.values
        train_score   = rejection90_sklearn(model, train_x, train_y, train_weights)

        test_x       = x[test].values
        test_y       = y[test].values
        test_weights = test_x.weight.values
        test_score   = rejection90_sklearn(model, test_x, test_y, test_weights)

        train_scores.append(train_score)
        test_scores.append(test_score)


    train_mean = np.mean(train_scores)
    train_std  = np.std(train_scores)
    test_mean  = np.mean(test_scores)
    test_std   = np.std(test_scores)

    print("Train score: {} (+/- {})".format(train_mean, train_std))
    print("Test score: {} (+/- {})".format(test_mean, test_std))

def create_submission(model: ClassifierMixin, test_dataset: pd.DataFrame, path: str) -> None:
    prediction = model.predict_proba(test_dataset.values)[:, 1]
    submission = pd.DataFrame(data={"prediction": prediction}, index=test_dataset.index)

    submission.to_csv("sample_submission.csv", index_label=utils.ID_COLUMN)

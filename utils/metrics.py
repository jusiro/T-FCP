import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(refs, preds):
    return np.round(accuracy_score(refs, np.argmax(preds, axis=-1))*100, 2)

def evaluate_conformal(pred_sets, labels, alpha=0.1):

    mean_size, median_size = set_size(pred_sets)
    coverage = empirical_set_coverage(pred_sets, labels)
    singletons = calculate_singletons(pred_sets)
    class_cov_gap = avg_class_coverage_gap(pred_sets, labels, alpha=alpha)
    empties = calculate_empties(pred_sets)
    singleton_cov = singleton_emp_cov(pred_sets, labels)

    return [coverage, mean_size, median_size, singletons, empties, class_cov_gap, singleton_cov]


def set_size(pred_sets):
    """
        Compute the size of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
        returns:
            sizes [numpy.array]: mean size of the predicted sets
    """
    mean_size   = np.mean([len(pred_set) for pred_set in pred_sets])
    median_size = np.median([len(pred_set) for pred_set in pred_sets])
    return mean_size, median_size


def calculate_singletons(pred_sets):
    """
        Compute the size of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
        returns:
            singletons [numpy.array]: number of sets containing a unique class
    """
    singletons = np.mean([len(pred_set) == 1 for pred_set in pred_sets])
    return singletons


def calculate_empties(pred_sets):
    """
        Compute the size of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
        returns:
            empties [numpy.array]: number of sets containing a unique class
    """
    empties = np.mean([len(pred_set) == 0 for pred_set in pred_sets])
    return empties


def empirical_set_coverage(pred_sets, labels):
    """
        Compute the empirical coverage of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
            labels [numpy.array]: true labels
        returns:
            coverage [float]: empirical coverage
    """
    coverage = np.mean([label in pred_set for label, pred_set in zip(labels, pred_sets)])
    return coverage


def singleton_emp_cov(pred_sets, labels):
    cov_s1, cov_s2, cov_s3 = np.nan, np.nan, np.nan

    TP = np.array([label in pred_set for label, pred_set in zip(labels, pred_sets)])

    idx_s1 = np.argwhere([len(pred_set) == 1 for pred_set in pred_sets])
    idx_s2 = np.argwhere([len(pred_set) == 2 for pred_set in pred_sets])
    idx_s3 = np.argwhere([len(pred_set) >= 3 for pred_set in pred_sets])

    if len(idx_s1) > 0:
        cov_s1 = TP[idx_s1].mean()
    if len(idx_s2) > 0:
        cov_s2 = TP[idx_s2].mean()
    if len(idx_s3) > 0:
        cov_s3 = TP[idx_s3].mean()

    return cov_s1


def avg_class_coverage_gap(pred_sets, labels, alpha=0.1):

    # Get sample-wise accuracy
    correct = np.int8([labels[i] in pred_sets[i] for i in range(len(labels))])

    violation = []
    for i_label in list(np.unique(labels)):
        idx = np.argwhere(labels == i_label)
        violation.append(abs(correct[idx].mean() - (1 - alpha)))

    # Get mean violation
    covgap = 100 * np.median(violation)

    return covgap


def cov_perc_satisfaction(cov_seeds, alpha, tol=0.005):
    return np.mean((np.array(cov_seeds)) > (1-alpha-tol))
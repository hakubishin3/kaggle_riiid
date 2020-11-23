from sklearn.metrics import roc_auc_score


def calc_metrics(y_true, y_pred):
    """ Calculate metrics excluding unpredicted values
    """
    not_pred_idx = y_pred != 0
    y_true = y_true[not_pred_idx]
    y_pred = y_pred[not_pred_idx]

    auc = roc_auc_score(y_true, y_pred)

    result = {
        "auc": auc,
    }
    return result

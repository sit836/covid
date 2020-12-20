import numpy as np
import pandas as pd
from sklearn.base import clone


def dropcol_importances(rf, X_train, y_train):
    """
    A brute force drop-column importance mechanism: Drop a column entirely, retrain the model, and recompute the
    performance score.
    """
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
        data={'Feature': X_train.columns,
              'Importance': imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I

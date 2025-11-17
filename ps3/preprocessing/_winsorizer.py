import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizes (clips) the data between a specified upper and lower quantile

    Parameters
    ----------
    lower_quantile : float
        Training data
    upper_quantile : float
        Name of ID column
    X : pd.DataFrame
        Feature matrix

    Returns
    -------
    X_winsorized
        Winsorized (clipped) feature matrix
    """
    def __init__(self, lower_quantile, upper_quantile):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)

        return self

    def transform(self, X):
        X_winsorized = np.clip(X, self.lower_quantile_, self.upper_quantile_)

        return X_winsorized
        

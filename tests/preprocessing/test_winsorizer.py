import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer

# define the parameters to test with
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)


def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)    # example feature matrix

    test_win = Winsorizer(lower_quantile=lower_quantile,
                        upper_quantile=upper_quantile)

    test_win.fit(X)
    
    # Get the learned limits 
    lower_limits = test_win.lower_quantile_
    upper_limits = test_win.upper_quantile_

    # Transform the data
    X_transformed = test_win.transform(X)

    
    # Assert that all transformed values are >= the learned lower limit
    assert np.all(X_transformed >= lower_limits)
    
    # Assert that all transformed values are <= the learned upper limit
    assert np.all(X_transformed <= upper_limits)

    # Compute expected upper and lower quantiles for given X
    expected_lower = np.quantile(X, lower_quantile, axis=0)
    expected_upper = np.quantile(X, upper_quantile, axis=0)
    
    # Assert that the correct limits were learned
    assert np.allclose(lower_limits, expected_lower)
    assert np.allclose(upper_limits, expected_upper)

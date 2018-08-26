# Partial Least squares linear regression

from __future__ import division
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
import MAPE
import plotresults
def PLLR(train_xf, train_yf, test_xf, test_yf, filename):
    # Partial Least squares linear regression
    regr_pls = PLSRegression()
    stime = time.time()
    regr_pls.fit(train_xf, train_yf)
    training_time = time.time() - stime
    print("Time for PLR fitting: %.6f" % (training_time))
    stime = time.time()
    y_pred_pls = regr_pls.predict(test_xf)
    test_time = time.time() - stime
    print("Time for PLR predicting: %.6f" % (test_time))
    np.savetxt(filename, y_pred_pls, delimiter=',')

    print("PLR Mean squared error: %.6f" % mean_squared_error(test_yf, y_pred_pls))
    # Explained variance score: 1 is perfect prediction
    r2 = r2_score(test_yf, y_pred_pls)
    print('PLR Variance score: %.2f' % r2)
    mape = MAPE.MAPE(test_yf, y_pred_pls)
    print("PLR Mean Percentage error: %.6f" % mape)
    return mape, r2, training_time, test_time, y_pred_pls
#Support vector regression

from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.svm import SVR
import MAPE
import plotresults

def SVR(train_xf, train_yf, test_xf, test_yf, filename):
    # Support vector regression (SVR)
    svr = SVR(C=10, epsilon=0.001, kernel='poly', degree=2)
    stime = time.time()
    svr.fit(train_xf, train_yf)
    training_time = time.time() - stime
    print("Time for SVR fitting: %.6f" % (training_time))
    stime = time.time()
    y_pred_svr = svr.predict(test_xf)
    test_time = time.time() - stime
    print("Time for SVR predicting: %.6f" % (test_time))

    #np.savetxt(filename, y_pred_gpr_mean, delimiter=',')

    print("SVR Mean squared error: %.6f" % mean_squared_error(test_yf, y_pred_svr))
    # Explained variance score: 1 is perfect prediction
    r2 = r2_score(test_yf, y_pred_svr)
    print('SVR Variance score: %.2f' % r2)
    mape = MAPE.MAPE(test_yf, y_pred_svr)
    print("GPR Mean Percentage error: %.6f" % mape)
    return mape, r2, training_time, test_time, y_pred_svr

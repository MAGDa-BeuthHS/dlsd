from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeserie, max_lags=10):
    """
    Calculates teststatistic for stationarity of timeseries
    :param timeserie: timeserie
    :param max_lags: maximum lag of estimation#
    :return test statistics
    """
    # Perform Dickey-Fuller test:
    dftest = adfuller(timeserie, autolag=None, maxlag=max_lags)
    test_results = {'Test_Statistic': dftest[0], 'p-value': dftest[1], '#Lags_Used': dftest[2], 'Num_of_Obs': dftest[3]}
    for key, value in dftest[4].items():
        test_results['Crit_Value_%s' % key] = value
    return test_results


def estimate_acf_pacf(timeserie, max_lags=10):
    """
    Calculates and plots the autocorrelation function of one timeserie
    :param max_lags: time lags
    :param timeserie:
    :return:
    """
    return acf(timeserie, nlags=max_lags), pacf(timeserie, nlags=max_lags, method='ols')

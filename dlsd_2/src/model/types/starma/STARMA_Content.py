import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from dlsd_2.src.calc.time_series_analysis import *
from dlsd_2.src.io.dataset.Dataset import Dataset
from dlsd_2.src.model.Model_Content import Model_Content
from pySTARMA import starma_model as stm
from pySTARMA import utils as util


class STARMA_Content(Model_Content):
    def __init__(self):
        super(STARMA_Content, self).__init__()
        self.pystarma_model = None
        self.ts_matrix = Dataset()  # contains all sensors
        self.wa_matrices = None  # contains spatial weights
        self.ar = 0
        self.ma = 0
        self.lags = ''
        self.max_t_lag = 25
        self.sample_size = 0.1
        self.iterations = 2
        self.prediction_dataset_object = Dataset()
        self.num_target_rows = None
        self.num_target_sensors = None
        self.num_target_offsets = None
        self.input_target_maker = None

    def set_input_target_maker(self, input_target_maker, test=False):
        self.input_target_maker = input_target_maker
        self.num_target_sensors = len(input_target_maker.get_target_sensor_idxs_list())
        self.num_target_offsets = len(input_target_maker.get_target_time_offsets_list())
        self._set_parameters_from_input_target_maker() if test is False else logging.debug("Testing")

    def _set_parameters_from_input_target_maker(self):
        self.num_target_rows = self.input_target_maker.get_target_dataset_object().get_number_rows()

    def set_pystarma_time_series(self, file_path):
        self.ts_matrix = Dataset()
        self.ts_matrix = pd.read_csv(file_path, header=0, lineterminator='\n', parse_dates=['date'],
                                     date_parser=lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S'))

    def set_pystarma_weight_matrices(self, file_path, file_names):
        for f in range(file_names):
            wa_order = pd.read_csv(f, header=0, lineterminator='\n', parse_dates=['time'],
                                   date_parser=lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S'))
            self.wa_matrices.append(wa_order.as_matrix())

    def preprocess_pystarma_model_with_source_maker(self, source_maker):
        source = source_maker.all_data
        self._preprocessing(self.ts_matrix.as_matrix(), self.wa_matrices, self.max_t_lag, self.sample_size)

        # make time serie stationar
        diff = (1, 288)
        self.ts_matrix = np.log1p(self.ts_matrix)
        self.ts_matrix = util.set_stationary(self.ts_matrix, diff)

        # re-run preprocessing
        self._preprocessing(self.ts_matrix.as_matrix(), self.wa_matrices, self.max_t_lag, self.sample_size)

        self._model_identification(self.ts_matrix, self.wa_matrices, max_lag=25)

    def _preprocessing(self, ts_matrix, wa_matrices=0, max_lag=25, sample_size=0.1):
        stationary_test = self._preprocess(ts_matrix, max_lag, sample_size)

        adf_test = stationary_test['test_stat']
        print('Test Statistic:\t%s' % adf_test['Test_Statistic'])
        print('p-Value:\t%s' % adf_test['p-value'])
        print('#Lags_Used:\t%s' % adf_test['#Lags_Used'])
        print('Num_of_Obs:\t%s' % adf_test['Num_of_Obs'])
        print('Crit_Value_1:\t%s' % adf_test['Crit_Value_1%'])
        print('Crit_Value_5:\t%s' % adf_test['Crit_Value_5%'])
        print('Crit_Value_10:\t%s' % adf_test['Crit_Value_10%'])

        plt.subplot(121)
        plt.title('Autocorrelation Function')
        self._plot_acf(stationary_test['acf_mean'], max_lag, len(ts_matrix))

        plt.subplot(122)
        plt.title('Partial Autocorrelation Function')
        self._plot_acf(stationary_test['pacf_mean'], max_lag, len(ts_matrix))
        plt.show()

        if wa_matrices != 0:
            cv = []
            cvm = []
            for tlag in range(max_lag):
                cv.append(1.96 / np.sqrt(len(ts_matrix) - tlag))
                cvm.append(-1.96 / np.sqrt(len(ts_matrix) - tlag))

            stacf = stm.Stacf(ts_matrix, wa_matrices, max_lag).estimate()
            self._plot_stacf(stacf, max_lag, len(ts_matrix))
            plt.title('Space-Time Autocorrelation Function')
            plt.show()

    def _preprocess(self, ts_matrix, max_lags=10, sample_size=0.1):
        factor = int(len(ts_matrix.T) * (sample_size))
        acf_matrix = np.zeros((max_lags + 1, factor))
        pacf_matrix = np.zeros((max_lags + 1, factor))

        test_stat = {'Test_Statistic': 0.,
                     'p-value': 0.,
                     '#Lags_Used': 0.,
                     'Num_of_Obs': 0.,
                     'Crit_Value_1%': 0.,
                     'Crit_Value_5%': 0.,
                     'Crit_Value_10%': 0.}

        for i in range(0, len(ts_matrix.T), (len(ts_matrix.T) + 1) / factor):
            timeserie = ts_matrix[:, i]
            acf_matrix[:, i / factor], pacf_matrix[:, i / factor] = estimate_acf_pacf(timeserie, max_lags)
            dfoutput = test_stationarity(timeserie, max_lags)
            test_stat['Test_Statistic'] += dfoutput['Test_Statistic'] * (1. / factor)
            test_stat['p-value'] += dfoutput['p-value'] * (1. / factor)
            test_stat['#Lags_Used'] += dfoutput['#Lags_Used'] * (1. / factor)
            test_stat['Num_of_Obs'] += dfoutput['Num_of_Obs'] * (1. / factor)
            test_stat['Crit_Value_1%'] += dfoutput['Crit_Value_1%'] * (1. / factor)
            test_stat['Crit_Value_5%'] += dfoutput['Crit_Value_5%'] * (1. / factor)
            test_stat['Crit_Value_10%'] += dfoutput['Crit_Value_10%'] * (1. / factor)

        acf_std = acf_matrix.std(1)
        acf_mean = acf_matrix.mean(1)
        pacf_std = pacf_matrix.std(1)
        pacf_mean = pacf_matrix.mean(1)

        return {"test_stat": test_stat, 'acf_std': acf_std, 'acf_mean': acf_mean, 'pacf_std': pacf_std,
                'pacf_mean': pacf_mean}

    def _model_identification(self, ts_matrix, wa_matrices, max_lag=25):
        # Get critcal value for model identification
        cv = []
        cvm = []
        for tlag in range(max_lag):
            cv.append(1.96 / np.sqrt(len(ts_matrix) - tlag))
            cvm.append(-1.96 / np.sqrt(len(ts_matrix) - tlag))

        stacf = stm.Stacf(ts_matrix, wa_matrices, max_lag).estimate()
        plt.subplot(211)
        self.__plot_stacf(stacf, max_lag, len(ts_matrix))
        plt.title('Space-Time Autocorrelation Function')

        stpacf = stm.Stpacf(ts_matrix, wa_matrices, max_lag).estimate()
        plt.subplot(212)
        self._plot_stacf(stpacf, max_lag, len(ts_matrix))
        plt.title('Space-Time Partial Autocorrelation Function')
        plt.show()

    def _plot_acf(self, acf, max_lag, ts_len):
        cv = []
        cvm = []
        for lag in range(max_lag):
            cv.append(1.96 / np.sqrt(ts_len - lag))
            cvm.append(-1.96 / np.sqrt(ts_len - lag))

        plt.plot(acf)
        plt.plot(cv, linestyle='--', color='red')
        plt.plot(cvm, linestyle='--', color='red')
        plt.axhline(y=0, linestyle='--', color='gray')
        pass

    def _plot_stacf(self, stacf, max_lag, ts_len):
        cv = []
        cvm = []
        for lag in range(max_lag):
            cv.append(1.96 / np.sqrt(ts_len - lag))
            cvm.append(-1.96 / np.sqrt(ts_len - lag))

        for i, ts in enumerate(stacf.T):
            plt.plot(ts, label=str(i) + '. Ordnung')

        plt.plot(stacf)
        plt.plot(cv, linestyle='--', color='red')
        plt.plot(cvm, linestyle='--', color='red')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pass

    def create_pystarma_model_from_source_maker(self, source_maker):
        source = source_maker.all_data
        self.pystarma_model = self._create_pystarma_model(self.ts_matrix, self.wa_matrices, self.ar, self.ma, self.lags,
                                                     self.iterations)

    def _create_pystarma_model(self, ts_matrix, wa_matrices, ar=0, ma=0, lags='', iterations=2):
        if lags == '':
            pystarma_model = stm.STARMA(ar, ma, ts_matrix.copy(), wa_matrices, iterations)
        else:
            pystarma_model = stm.STARIMA(ar, ma, lags, ts_matrix.copy(), wa_matrices,
                                         iterations)
        return pystarma_model

    def fit_model(self):
        self.pystarma_model.fit()

        def _extract_sensors_currently_being_used_as_output_from_source_data(self):
            self.subset_starma_array = self._ts_matrix.df[self.input_target_maker.get_target_sensor_idxs_list()].values

        def _target_time_offset_at_index(self, index):
            return self.input_target_maker.get_target_time_offsets_list()[index]

        def _max_input_time_offset(self):
            return max(self.input_target_maker.get_input_time_offsets_list())

    def make_prediction_dataset_object(self):
        size = [self.num_target_rows, self.num_target_offsets * self.num_target_sensors]
        array = np.zeros(size)
        self._extract_sensors_currently_being_used_as_output_from_source_data()
        self._iterate_over_target_time_offsets_replicating_pystarma_model(array)
        self.prediction_dataset_object.set_numpy_array(array)
        return self.prediction_dataset_object

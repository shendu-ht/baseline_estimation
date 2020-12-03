#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : spot.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 2020/12/2 6:01 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 2020/12/2 6:01 下午 by shendu.ht  init
"""
from math import log

import numpy
import pandas
import tqdm
from matplotlib import pyplot
from scipy.optimize import minimize

deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'


class BiSpot:
    """
    This class allows to run biSPOT algorithm on univariate dataset (upper and lower bounds)

    Attributes
    -----------
    proba: Detection level (risk), chosen by the user
    extreme_quantile: current threshold (bound between normal and abnormal events)
    data: stream
    init_data: initial batch of observations (for the calibration/initialization step)
    init_threshold: initial threshold computed during the calibration step
    peaks: array of peaks (excesses above the initial threshold)
    n: number of observed values
    Nt: number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor
        :param q: Detection level (risk)
        """
        self.proba = q
        self.data = None
        self.init_data = None
        self.n = 0

        none_dict = {"up": None, "down": None}
        self.extreme_quantile = dict.copy(none_dict)
        self.init_threshold = dict.copy(none_dict)
        self.peaks = dict.copy(none_dict)
        self.gamma = dict.copy(none_dict)
        self.sigma = dict.copy(none_dict)
        self.Nt = {"up": 0, "down": 0}

        self.alarm = list()

    def set_data(self, init_data, data):
        """
        Import data to BiSPOT object

        :param init_data: initial batch to calibrate the algorithm. list, numpy.array or pandas.Series
        :param data: data for the run (list, np.array or pd.series)
        """

        if isinstance(data, list):
            self.data = numpy.array(data)
        elif isinstance(data, numpy.ndarray):
            self.data = data
        elif isinstance(data, pandas.Series):
            self.data = data.values
        else:
            raise ValueError("This data format (%s) is not supported" % type(data))

        if isinstance(init_data, list):
            self.data = numpy.array(data)
        elif isinstance(init_data, numpy.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pandas.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and 0 < init_data < 1:
            r = int(init_data * self.data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            raise ValueError("The initial data cannot be set")

    def add_data(self, data):
        """
        This function allows to append data to the already fitted data

        :param data: data to append. list, numpy.array, pandas.Series
        """
        if isinstance(data, list):
            data = numpy.array(data)
        elif isinstance(data, numpy.ndarray):
            data = data
        elif isinstance(data, pandas.Series):
            data = data.values
        else:
            raise ValueError("This data format (%s) is not supported" % type(data))

        self.data = numpy.append(self.data, data)

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step

        :param verbose: (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size

        sort_data = numpy.sort(self.init_data)
        self.init_threshold["up"] = sort_data[int(n_init * 0.98)]
        self.init_threshold["down"] = sort_data[int(n_init * 0.02)]

        # initial peaks
        self.peaks["up"] = self.init_data[self.init_data > self.init_threshold["up"]] - self.init_threshold["up"]
        self.peaks["down"] = self.init_threshold["down"] - self.init_data[self.init_data < self.init_threshold["down"]]
        self.Nt["up"] = self.peaks["up"].size
        self.Nt["down"] = self.peaks["down"].size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)

        for side in ["up", "down"]:
            g, s, _ = self._grim_shaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side], self.sigma[side] = g, s

    def _roots_finder(self, fun, jac, bounds, n_points, method):
        """
        Find possible roots of a scalar function

        :param fun: scalar function
        :param jac: first order derivative of the function
        :param bounds: (min,max) interval for the roots search
        :param n_points: maximum number of roots to output
        :param method: 'regular' : regular sample of the search interval,
                       'random' : uniform (distribution) sample of the search interval
        :return: possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (n_points + 1)
            x0 = numpy.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            x0 = numpy.random.uniform(bounds[0], bounds[1], n_points)
        else:
            raise ValueError("This method (%s) is not supported" % method)

        def obj_fun(x_back, f_back, jac_back):
            g, i, j = 0, 0, numpy.zeros(x_back.shape)
            for x_i in x_back:
                f_x = f_back(x_i)
                g += f_x ** 2
                j[i] = 2 * f_x * jac_back(x_i)
                i += 1
            return g, j

        opt = minimize(lambda x: obj_fun(x, fun, jac), x0, method="L-BFGS-B", jac=True, bounds=[bounds] * len(x0))
        x = opt.x
        numpy.round(x, decimals=5)
        return numpy.unique(x)

    def _log_likelihood(self, y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        :param y: observations
        :param gamma: GPD index parameter
        :param sigma: GPD scale parameter (>0)
        :return: log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = y.size
        if gamma != 0:
            tau = gamma / sigma
            l = -n * log(sigma) - (1 + (1 / gamma)) * (numpy.log(1 + tau * y)).sum()
        else:
            l = n * (1 + log(y.mean()))

        return l

    def _grim_shaw(self, side, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        :param side: "up" or "down"
        :param epsilon: numerical parameter to perform (default : 1e-8)
        :param n_points: maximum number of candidates for maximum likelihood (default : 10)
        :return: gamma_best,sigma_best,ll_best
                 gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + numpy.log(s).mean()

        def v(s):
            return numpy.mean(1 / s)

        def w(y, t):
            s = 1 + t * y
            us, vs = u(s), v(s)
            return us * vs - 1

        def jac_w(y, t):
            s = 1 + t * y
            us, vs = u(s), v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + numpy.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        y_min = self.peaks[side].min()
        y_max = self.peaks[side].max()
        y_mean = self.peaks[side].mean()

        a = -1 / y_max
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (y_mean - y_min) / (y_mean * y_min)
        c = 2 * (y_mean - y_min) / (y_min ** 2)

        # We look for possible roots
        left_zeros = self._roots_finder(lambda t: w(self.peaks[side], t),
                                        lambda t: jac_w(self.peaks[side], t),
                                        (a + epsilon, -epsilon),
                                        n_points, 'regular')

        right_zeros = self._roots_finder(lambda t: w(self.peaks[side], t),
                                         lambda t: jac_w(self.peaks[side], t),
                                         (b, c),
                                         n_points, 'regular')

        # all the possible roots
        zeros = numpy.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = y_mean
        ll_best = self._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = self._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, side, gamma, sigma):
        """
        Compute the quantile at level 1-q for a given side

        :param side: "up" or "down"
        :param gamma: GPD parameter
        :param sigma: GPD parameter
        :return: quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        if side == "up":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["up"] + (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["up"] - sigma * log(r)
        elif side == "down":
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold["down"] - (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold["down"] + sigma * log(r)
        raise ValueError("error : the side is not right")

    def run(self, with_alarm=True):
        """
        Run BiSPOT on the stream

        :param with_alarm: (default = True) If False, SPOT will adapt the threshold assuming
                           there is no abnormal values
        :return: dict, keys : "upper_thresholds", "lower_thresholds" and "alarms"
                              "***-thresholds" contains the extreme quantiles and "alarms" contains
        """
        if self.n > self.init_data.size:
            print("Warning : the algorithm seems to have already been run, you should initialize before running again")
            return {}

        # list of the thresholds
        threshold_up = []
        threshold_down = []
        alarm = []

        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            # If the observed value exceeds the current threshold (alarm case)
            if self.data[i] > self.extreme_quantile["up"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["up"] = numpy.append(self.peaks["up"], self.data[i] - self.init_threshold["up"])
                    self.Nt["up"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grim_shaw("up")
                    self.extreme_quantile["up"] = self._quantile("up", g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] > self.init_threshold["up"]:
                # we add it in the peaks
                self.peaks["up"] = numpy.append(self.peaks["up"], self.data[i] - self.init_threshold["up"])
                self.Nt["up"] += 1
                self.n += 1
                # and we update the thresholds
                g, s, l = self._grim_shaw("up")
                self.extreme_quantile["up"] = self._quantile("up", g, s)

            elif self.data[i] < self.extreme_quantile["down"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["down"] = numpy.append(self.peaks["down"], -(self.data[i] - self.init_threshold["down"]))
                    self.Nt["down"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grim_shaw("down")
                    self.extreme_quantile["down"] = self._quantile("down", g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] < self.init_threshold["down"]:
                # we add it in the peaks
                self.peaks["down"] = numpy.append(self.peaks["down"], -(self.data[i] - self.init_threshold["down"]))
                self.Nt["down"] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grim_shaw("down")
                self.extreme_quantile["down"] = self._quantile("down", g, s)
            else:
                self.n += 1

            threshold_up.append(self.extreme_quantile["up"])  # thresholds record
            threshold_down.append(self.extreme_quantile["down"])  # thresholds record

        self.alarm = alarm
        return {"upper_thresholds": threshold_up, "lower_thresholds": threshold_down, "alarms": alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results of given by the run

        :param run_results: results given by the "run" method
        :param with_alarm: (default = True) If True, alarms are plotted.
        :return: list of the plots
        """

        x = range(self.data.size)
        keys = run_results.keys()

        ts_fig, = pyplot.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if "upper_thresholds" in keys:
            threshold_up = run_results["upper_thresholds"]
            uth_fig, = pyplot.plot(x, threshold_up, color=deep_saffron, lw=2, ls="dashed")
            fig.append(uth_fig)

        if "lower_thresholds" in keys:
            threshold_down = run_results["lower_thresholds"]
            lth_fig, = pyplot.plot(x, threshold_down, color=deep_saffron, lw=2, ls="dashed")
            fig.append(lth_fig)

        if with_alarm and ("alarms" in keys):
            alarm = run_results["alarms"]
            al_fig = pyplot.scatter(alarm, self.data[alarm], color="red")
            fig.append(al_fig)

        pyplot.xlim((0, self.data.size))
        pyplot.show()

        return fig

    def __str__(self):
        s = ""

        s += "Streaming Peaks-Over-Threshold Object\n"
        s += "Detection level q = %s\n" % self.proba

        if self.data is not None:
            s += "Data imported : Yes\n"
            s += "\t initialization  : %s values\n" % self.init_data.size
            s += "\t stream : %s values\n" % self.data.size
        else:
            s += "Data imported : No\n"
            return s

        if self.n == 0:
            s += "Algorithm initialized : No\n"
        else:
            s += "Algorithm initialized : Yes\n"
            s += "\t initial threshold : %s\n" % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += "Algorithm run : Yes\n"
                s += "\t number of observations : %s (%.2f %%)\n" % (r, 100 * r / self.n)
                s += "\t triggered alarms : %s (%.2f %%)\n" % (len(self.alarm), 100 * len(self.alarm) / self.n)
            else:
                s += "\t number of peaks  : %s\n" % self.Nt
                s += "\t upper extreme quantile : %s\n" % self.extreme_quantile["up"]
                s += "\t lower extreme quantile : %s\n" % self.extreme_quantile["down"]
                s += "Algorithm run : No\n"
        return s

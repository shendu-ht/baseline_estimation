#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : bspot.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 2020/12/2 10:08 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 2020/12/2 10:08 下午 by shendu.ht  init
"""
import numpy
import tqdm

from extreme_value_theory.espot import back_mean
from extreme_value_theory.spot import BiSpot


class BidSPOT(BiSpot):
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper and lower bounds)

    Attributes
    -----------
    proba : Detection level (risk), chosen by the user
    depth : Number of observations to compute the moving average
    extreme_quantile : current threshold (bound between normal and abnormal events)
    data : stream
    init_data : initial batch of observations (for the calibration/initialization step)
    init_threshold : initial threshold computed during the calibration step
    peaks : array of peaks (excesses above the initial threshold)
    n : number of observed values
    Nt : number of observed peaks
    """

    def __init__(self, q=1e-4, depth=10):
        self.proba = q
        self.data = None
        self.init_data = None
        self.n = 0
        self.depth = depth

        none_dict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(none_dict)
        self.init_threshold = dict.copy(none_dict)
        self.peaks = dict.copy(none_dict)
        self.gamma = dict.copy(none_dict)
        self.sigma = dict.copy(none_dict)
        self.Nt = {"up": 0, "down": 0}

        self.alarm = list()

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step

        :param verbose : (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size - self.depth

        m = back_mean(self.init_data, self.depth)
        t = self.init_data[self.depth:] - m[:-1]  # new variable

        sort_t = numpy.sort(t)  # we sort T to get the empirical quantile
        self.init_threshold["up"] = sort_t[int(0.98 * n_init)]  # t is fixed for the whole algorithm
        self.init_threshold["down"] = sort_t[int(0.02 * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks["up"] = t[t > self.init_threshold["up"]] - self.init_threshold["up"]
        self.peaks["down"] = -(t[t < self.init_threshold["down"]] - self.init_threshold["down"])
        self.Nt["up"] = self.peaks["up"].size
        self.Nt["down"] = self.peaks["down"].size
        self.n = n_init

        if verbose:
            print("Initial threshold : %s" % self.init_threshold)
            print("Number of peaks : %s" % self.Nt)
            print("Grimshaw maximum log-likelihood estimation ... ")

        l = {"up": None, "down": None}
        for side in ["up", "down"]:
            g, s, l[side] = self._grim_shaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

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

            # actual normal window
        w = self.init_data[-self.depth:]

        # list of the thresholds
        threshold_up = []
        threshold_down = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):
            mi = w.mean()
            ni = self.data[i] - mi
            # If the observed value exceeds the current threshold (alarm case)
            if ni > self.extreme_quantile["up"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["up"] = numpy.append(self.peaks["up"], ni - self.init_threshold["up"])
                    self.Nt["up"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grim_shaw("up")
                    self.extreme_quantile["up"] = self._quantile("up", g, s)
                    w = numpy.append(w[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif ni > self.init_threshold["up"]:
                # we add it in the peaks
                self.peaks["up"] = numpy.append(self.peaks["up"], ni - self.init_threshold["up"])
                self.Nt["up"] += 1
                self.n += 1
                # and we update the thresholds
                g, s, l = self._grim_shaw("up")
                self.extreme_quantile["up"] = self._quantile("up", g, s)
                w = numpy.append(w[1:], self.data[i])

            elif ni < self.extreme_quantile["down"]:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks["down"] = numpy.append(self.peaks["down"], -(ni - self.init_threshold["down"]))
                    self.Nt["down"] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grim_shaw("down")
                    self.extreme_quantile["down"] = self._quantile("down", g, s)
                    w = numpy.append(w[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif ni < self.init_threshold["down"]:
                # we add it in the peaks
                self.peaks["down"] = numpy.append(self.peaks["down"], -(ni - self.init_threshold["down"]))
                self.Nt["down"] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grim_shaw("down")
                self.extreme_quantile["down"] = self._quantile("down", g, s)
                w = numpy.append(w[1:], self.data[i])
            else:
                self.n += 1
                w = numpy.append(w[1:], self.data[i])

            threshold_up.append(self.extreme_quantile["up"] + mi)  # upper thresholds record
            threshold_down.append(self.extreme_quantile["down"] + mi)  # lower thresholds record

        self.alarm = alarm
        return {"upper_thresholds": threshold_up, "lower_thresholds": threshold_down, "alarms": alarm}

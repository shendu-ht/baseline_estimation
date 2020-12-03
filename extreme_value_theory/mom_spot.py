#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : mom_spot.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 2020/12/2 10:47 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 2020/12/2 10:47 下午 by shendu.ht  init
"""
import numpy
import tqdm

from extreme_value_theory.spot import BiSpot


class MomSpot(BiSpot):

    def __init__(self, q=1e-4):
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

        self.alarm = []

    def _mom(self, side):
        yi = self.peaks[side]
        avg = numpy.mean(yi)
        var = numpy.var(yi)
        sigma = 0.5 * avg * (avg ** 2 / var + 1)
        gamma = 0.5 * (avg ** 2 / var - 1)
        return gamma, sigma, 100

    def run(self, with_alarm=True):
        
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

                    g, s, l = self._mom("up")
                    self.extreme_quantile["up"] = self._quantile("up", g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] > self.init_threshold["up"]:
                # we add it in the peaks
                self.peaks["up"] = numpy.append(self.peaks["up"], self.data[i] - self.init_threshold["up"])
                self.Nt["up"] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._mom("up")
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

                    g, s, l = self._mom("down")
                    self.extreme_quantile["down"] = self._quantile("down", g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] < self.init_threshold["down"]:
                # we add it in the peaks
                self.peaks["down"] = numpy.append(self.peaks["down"], -(self.data[i] - self.init_threshold["down"]))
                self.Nt["down"] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._mom("down")
                self.extreme_quantile["down"] = self._quantile("down", g, s)
            else:
                self.n += 1

            threshold_up.append(self.extreme_quantile["up"])  # thresholds record
            threshold_down.append(self.extreme_quantile["down"])  # thresholds record

        self.alarm = alarm
        return {"upper_thresholds": threshold_up, "lower_thresholds": threshold_down, "alarms": alarm}

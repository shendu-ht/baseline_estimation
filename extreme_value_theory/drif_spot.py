#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : drif_spot.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 2020/12/2 10:41 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 2020/12/2 10:41 下午 by shendu.ht  init
"""
import numpy
import pandas
import tqdm

from extreme_value_theory.spot import BiSpot


class DrSpot(BiSpot):

    def __init__(self, q=1e-4):
        self.proba = q
        self.data = None
        self.init_data = None
        self.update_number = 0
        self.n = 0
        none_dict = {"up": None, "down": None}

        self.extreme_quantile = dict.copy(none_dict)
        self.init_threshold = dict.copy(none_dict)
        self.peaks = dict.copy(none_dict)
        self.gamma = dict.copy(none_dict)
        self.sigma = dict.copy(none_dict)
        self.Nt = {"up": 0, "down": 0}

        self.alarm = []

    def set_data(self, init_data, data):
        if isinstance(data, list):
            self.data = numpy.array(data)
        elif isinstance(data, numpy.ndarray):
            self.data = data
        elif isinstance(data, pandas.Series):
            self.data = data.values
        else:
            raise ValueError("This data format (%s) is not supported" % type(data))

        if isinstance(init_data, list):
            self.init_data = numpy.array(init_data)
            self.update_number = len(self.init_data)
        elif isinstance(init_data, numpy.ndarray):
            self.init_data = init_data
            self.update_number = len(self.init_data)
        elif isinstance(init_data, pandas.Series):
            self.init_data = init_data.values
            self.update_number = len(self.init_data)
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
            self.update_number = init_data
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
            self.update_number = r
        else:
            raise ValueError("The initial data cannot be set")

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

            if self.n % self.update_number == 0:  # update 
                # update on time
                up_data = numpy.append(self.init_data, self.data[:i])
                print("update at ", i)
                print("updating using data: ", len(up_data))
                sort_data = numpy.sort(up_data)  # we sort X to get the empirical quantile
                # t is fixed for the whole algorithm
                self.init_threshold["up"] = sort_data[int(0.98 * len(sort_data))]
                # t is fixed for the whole algorithm
                self.init_threshold["down"] = sort_data[int(0.02 * len(sort_data))]

                self.peaks["up"] = up_data[up_data > self.init_threshold["up"]] - self.init_threshold["up"]
                self.peaks["down"] = -(up_data[up_data < self.init_threshold["down"]] - self.init_threshold["down"])
                self.Nt["up"] = self.peaks["up"].size
                self.Nt["down"] = self.peaks["down"].size

                for side in ["up", "down"]:
                    g, s, _ = self._grim_shaw(side)
                    self.extreme_quantile[side] = self._quantile(side, g, s)
                    self.gamma[side] = g
                    self.sigma[side] = s

            threshold_up.append(self.extreme_quantile["up"])  # thresholds record
            threshold_down.append(self.extreme_quantile["down"])  # thresholds record

        return {"upper_thresholds": threshold_up, "lower_thresholds": threshold_down, "alarms": alarm}


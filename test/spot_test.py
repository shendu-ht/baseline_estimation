#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : spot_test.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 2020/12/3 11:45 上午
    Description : description what the main function of this file
    Change Activity :
            version0 : 2020/12/3 11:45 上午 by shendu.ht  init
"""
import os
import time
import unittest

import numpy

from extreme_value_theory.drif_spot import DrSpot
from extreme_value_theory.spot import BiSpot
from test import FILE_PATH

SPOT_FILE_PATH = os.path.join(FILE_PATH, "extreme_value_theory")


class PhysicCheck(unittest.TestCase):
    """
        use Physics data to test
    """

    @staticmethod
    def testSpot():
        file = os.path.join(SPOT_FILE_PATH, "physics.dat")
        r = open(file, "r").read().split(",")
        x = numpy.array(list(map(float, r)))
        n_init = 2000
        init_data = x[:n_init]  # initial batch
        data = x[n_init:]  # stream

        q = 1e-3  # risk parameter
        # d = 450  # depth parameter
        start = time.time()
        spot = BiSpot(q)
        spot.set_data(init_data, data)
        spot.initialize()

        results = spot.run()
        end = time.time()
        print("Runtime is:", end - start)
        spot.plot(results)

    @staticmethod
    def testDriftSpot():
        file = os.path.join(SPOT_FILE_PATH, "physics.dat")
        r = open(file, "r").read().split(",")
        x = numpy.array(list(map(float, r)))
        n_init = 2000
        init_data = x[:n_init]  # initial batch
        data = x[n_init:]  # stream

        q = 1e-3  # risk parameter
        # d = 450  # depth parameter
        start = time.time()
        spot = DrSpot(q)
        spot.set_data(init_data, data)
        spot.initialize()

        results = spot.run()
        end = time.time()
        print("Runtime is:", end - start)
        spot.plot(results)

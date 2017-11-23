import os

import numpy as np
import unittest
from cityfinder.city_vector import CityVector, israel, match


class TestCityVector(unittest.TestCase):
    def test_azimuths(self):
        print(israel['tlv', 'j-m'])
        
        # azimuths = israel._calc_azimuths()
        # print(azimuths)

    def test_match(self):
        israel_with_outliers = israel.add_outliers(10)
        match(israel, israel_with_outliers)

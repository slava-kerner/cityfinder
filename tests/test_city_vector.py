import os

import numpy as np
import unittest
from cityfinder.city_vector import CityVector, israel


class TestCityVector(unittest.TestCase):
    def test_azimuths(self):
        azimuths = israel._calc_azimuths()
        print(azimuths)
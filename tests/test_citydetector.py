import cv2

import unittest
from cityfinder.citydetector import CityDetector


class TestCityDetector(unittest.TestCase):
    def test_find_circle_city(self):
        path = 'data/city/tulkarm.png'
        debug_path = 'out/debug'

        config = CityDetector.default_config()  # and override
        detector = CityDetector(config=config, debug_path=debug_path)
        detector.find_circle_city(path)
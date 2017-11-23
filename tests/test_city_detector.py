import os

import numpy as np
import unittest
from cityfinder.city_detector import CityDetector


class TestCityDetector(unittest.TestCase):
    in_folder = 'data/city'
    out_folder = 'out'
    debug_folder = 'out/debug'

    def _path(self, name):
        return os.path.join(self.in_folder, name + '.png')

    def test_optimize_params(self):
        name = 'tulkarm'

        config = CityDetector.default_config()
        for config['hough_circle']['dp'] in np.arange(.5, 2.5, .5):
            for config['hough_circle']['maxRadius'] in range(5, 40, 5):
                config_text = '_maxRadius=%0.2f_dp=%0.2f' % (config['hough_circle']['maxRadius'], config['hough_circle']['dp'])
                out_path = os.path.join(self.out_folder, name + config_text + '.png')
                detector = CityDetector(config=config)
                circles = detector.find_circle_city(self._path(name), out_path=out_path, imshow=False)
                print('config: %s. Detected %d circles' % (config_text, len(circles)))

    def test_radii_histogram(self):
        name = 'tulkarm'

        config = CityDetector.default_config()
        config['hough_circle']['dp'] = 2
        config['hough_circle']['maxRadius'] = 20
        detector = CityDetector(config=config)
        circles = detector.find_circle_city(self._path(name))
        radii = [c[1] for c in circles]
        print(sorted(radii))

    def test_find_pink_blob(self):
        name = 'tulkarm'

        config = CityDetector.default_config()
        detector = CityDetector(config=config)
        out_path = os.path.join(self.out_folder, name + '_hsv' + '.png')

        pink = detector.find_pink_blob(self._path(name), imshow=True, out_path=out_path)

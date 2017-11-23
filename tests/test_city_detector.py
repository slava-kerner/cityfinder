import os
import itertools

from PIL import Image
import numpy as np
import unittest
from cityfinder.city_detector import CityDetector
from cityfinder.city_vector import CityVector, israel, israel_utm, match


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
        name = 'big_downsampled'

        # downsamples:
        # path = self._path(name)
        # img = Image.open(path)
        # print(img.size)
        # new_size = (img.size[0] // 4, img.size[0] // 4)
        # out_path = path[:-4] + '_downsampled.png'
        # img_downsampled = img.resize(new_size)
        # print(img_downsampled.size)
        # img_downsampled.save(out_path)
        # return

        config = CityDetector.default_config()
        detector = CityDetector(config=config)

        cities = detector.find_pink_blob(self._path(name), imshow=False, out_folder=os.path.join(self.out_folder, name))
        print(cities)

        cities = CityVector(cities)
        match(cities, israel_utm)
        # print('\n\n\n')
        # relevant_cities = ['nablus', 'tlv', 'j-m', 'haifa', 'gaza', 'tiberias']
        # for city1, city2 in itertools.combinations(relevant_cities, r=2):
        #     pair = (city1, city2)
        #     az1, d1 = cities[pair]
        #     az2, d2 = israel_utm[pair]
        #     print(*pair, az1 - az2, 100 * d1 / d2)

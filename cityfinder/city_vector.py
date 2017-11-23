import itertools
from copy import copy, deepcopy
import operator

import cv2
from affine import Affine
from uuid import uuid4
import random
import numpy as np


class CityVector:
    """ {city -> (x, y)}, where x,y are ideally lon,lat. in future consider radius. """
    def __init__(self, other=None):
        if other is not None:
            if isinstance(other, dict):
                self.cities = other
            elif isinstance(other, CityVector):
                self.cities = other.cities
            else:
                raise NotImplementedError
        else:
            self.cities = {}

    def __iter__(self):
        return iter(self.cities)

    def __len__(self):
        return len(self.cities)

    def azimuth(self, city1, city2):
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return np.rad2deg(np.math.atan2(y2 - y1, x2 - x1))

    def dist(self, city1, city2):
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return np.linalg.norm([y2 - y1, x2 - x1])

    def __getitem__(self, cities):
        """
        :param cities: (city1, city2)
        :return: returns (azimuth, distance)
        """
        city1, city2 = cities
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        azimuth = np.rad2deg(np.math.atan2(y2 - y1, x2 - x1))
        distance = np.linalg.norm([y2 - y1, x2 - x1])
        return azimuth, distance

    def _calc_azimuths(self):
        azimuths = {(city1, city2): self.azimuth(city1, city2)
                    for city1 in self.cities for city2 in self.cities if city1 != city2}
        return azimuths

    def add_outliers(self, number):
        cities = copy(self.cities)
        for idx in range(number):
            cities[str(uuid4())] = (random.uniform(0, 90), random.uniform(0, 90))
        return CityVector(cities)


def match(cities1, cities2):
    # TODO move to config:
    THRESHOLD_AZIMUTH = 1  # [deg]
    BINS_NUM = 100
    RATIO_TEST_THRESHOLD = 2  # ratio_test above that is success (result is significant)
    RATIO_THRESHOLD = 1.01  # keep only those that agree with best ratio up to this rel_tol

    # find potential matches - those that agree on azimuth:
    potential_matches = []  # list of pairs, each pair being 2 cities: (('tlv', 'j-m'), ('tlv', 'j-m'))
    for pair1 in itertools.combinations(cities1, r=2):
        az1, dist1 = cities1[list(pair1)]
        # print(*pair1, az1, dist1)
        for pair2 in itertools.permutations(cities2, r=2):
            az2, dist2 = cities2[list(pair2)]
            if abs(az1 - az2) < THRESHOLD_AZIMUTH:
                potential_matches.append((pair1, pair2))
    print('\n\nfound %d potential matches:' % len(potential_matches))
    print_per_line(potential_matches)

    # collect distance ratios:
    print('\n\ndistance ratios:')
    ratios = [cities1.dist(*pair1) / cities2.dist(*pair2) for pair1, pair2 in potential_matches]
    print_per_line(sorted(ratios))

    # choose cluster. currently primitive - find largest bin, validate by comparing to 3rd largest bin
    print('\n\nfinding most dense cluster of ratios:')
    ratios = sorted(ratios)
    min, max = np.min(ratios), np.max(ratios)
    hist = np.histogram(ratios, bins=BINS_NUM, range=(min, max))
    hist = {hist[1][idx]: hist[0][idx] for idx in range(len(hist[0]))}  # convert to {value->bin_size}
    hist = sorted(hist.items(), key=operator.itemgetter(1), reverse=True)
    print_per_line(hist)
    ratio_test = hist[0][1] / hist[2][1]  # checking vs. 3rd and not 2nd, since peak could be split between 2 bins.
    ratio_test_success = ratio_test > RATIO_TEST_THRESHOLD
    print('\nratio_test: %f, success=%s' % (ratio_test, ratio_test_success))
    if not ratio_test_success:
        return []
    best_ratio = hist[0][0]
    print('best_ratio:', best_ratio)

    # filter potential matches, keeping only those with ratio close to best ratio:
    def ratio_good(pair1, pair2):
        ratio = cities1.dist(*pair1) / cities2.dist(*pair2)
        return best_ratio / RATIO_THRESHOLD < ratio < best_ratio * RATIO_THRESHOLD

    matches = [p for p in potential_matches if ratio_good(*p)]
    print('\n\nafter filtering by ratio, remained %d pairs (out of %d):' % (len(matches), len(potential_matches)))
    print_per_line(matches)

    # for each city, collect candidate matches:
    city_matches = {}  # city->[matching cities]
    for pair1, pair2 in matches:
        for idx in [0, 1]:
            city_matches.setdefault(pair1[idx], [])
            city_matches[pair1[idx]].append(pair2[idx])
    print('\n\ncity matches:')
    print_per_line(city_matches.items())

    # filter city matches - only leave those, where most candidate matches are identical:
    def is_city_match_robust(candidates):
        cities, counts = np.unique(candidates, return_counts=True)
        return cities[0] if counts[0] > 0.9 * len(candidates) else None  # robust

    city_matches = {city: is_city_match_robust(candidates) for city, candidates in city_matches.items()}
    city_matches = {city: match for city, match in city_matches.items() if match is not None}
    print('\n\nfinal matches:')
    print_per_line(city_matches.items())

    # find affine:
    src = np.array([[cities1.cities[m][0] for m in city_matches.keys()], [cities1.cities[m][1] for m in city_matches.keys()]])
    dst = np.array([[cities2.cities[m][0] for m in city_matches.values()], [cities2.cities[m][1] for m in city_matches.values()], [1] * len(city_matches.values())])
    print('\n\nsrc:', src)
    print('\n\ndst:', dst)
    affine = np.linalg.lstsq(src.transpose(), dst.transpose())
    affine = affine[0].transpose()
    affine = Affine(*np.ravel(affine))
    print('\n\naffine:', affine)

    # find homography:
    src = np.array([[*cities1.cities[m]] for m in city_matches.keys()])
    dst = np.array([[*cities2.cities[m]] for m in city_matches.values()])
    h, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    print('\n\nhomography:', h)

israel = CityVector({'tlv': (34.777303, 32.076025),
                     'j-m': (35.207415, 31.768136),
                     'haifa': (34.997550, 32.804523),
                     'tiberias': (35.533599, 32.793664),
                     'beersheba': (34.790372, 31.249257),
                     'eilat': (34.945825, 29.553369)})


def print_per_line(arr):
    print('\n'.join([str(m) for m in arr]))
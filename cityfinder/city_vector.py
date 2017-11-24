import itertools
from copy import copy, deepcopy
import operator
import pyproj
import shapely.geometry

import cv2
from affine import Affine
from uuid import uuid4
import random
import numpy as np

from cityfinder.geo import transform, p_geographic, p_utm, lonlat_to_utm, fit_affine, UTM
from cityfinder.gcp import GCP


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
    THRESHOLD_AZIMUTH = 5  # [deg]
    BINS_NUM = 20
    RATIO_TEST_THRESHOLD = 1.5  # ratio_test above that is success (result is significant)
    RATIO_THRESHOLD = 1.1  # keep only those that agree with best ratio up to this rel_tol
    CANDIDATES_RATIO = 0.5

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
    ratios = np.log(ratios)
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
    best_ratio = np.exp(hist[0][0])
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
        argsort = np.argsort(-counts)  # minus in order to argsort reverse (descending)
        cities = cities[argsort]
        counts = counts[argsort]
        return cities[0] if counts[0] > CANDIDATES_RATIO * len(candidates) else None  # robust

    city_matches = {city: is_city_match_robust(candidates) for city, candidates in city_matches.items()}
    city_matches = {city: match for city, match in city_matches.items() if match is not None}
    print('\n\nfinal matches:')
    print_per_line(city_matches.items())

    def correct_x_back(x, width):
        factor = 1 / 0.66
        return width / 2 + factor * (x - width / 2)
    # find affine:
    # print('before hack:', city_matches)
    city_matches = {k: v for idx, (k, v) in enumerate(list(city_matches.items())) if idx in [0, 1, 2, 3, 4]}
    # print('after hack:', city_matches)
    src = np.array([[cities2.cities[m][0] for m in city_matches.values()], [cities2.cities[m][1] for m in city_matches.values()]])
    dst = np.array([[correct_x_back(cities1.cities[m][0], 4147) for m in city_matches.keys()], [cities1.cities[m][1] for m in city_matches.keys()], [1] * len(city_matches.values())])

    width, height = 4147, 4147  # TODO HACK

    gcps = []
    for city1, city2 in city_matches.items():
        pix = cities1.cities[city1]
        lonlat = cities2.cities[city2]
        gcps.append(GCP(*lonlat, None, '', correct_x_back(pix[0], width=width), height - pix[1]))

    # affine, _ = fit_affine(gcps, p_geographic)
    # print('geographic:', affine)

    affine, _ = fit_affine(gcps, p_utm)
    print('utm:', affine)

    return affine, UTM


israel = CityVector({
                     'tlv': (34.777303, 32.076025),
                     'j-m': (35.207415, 31.768136),
                     'haifa': (34.997550, 32.804523),
                     'tiberias': (35.533599, 32.793664),
                     # 'beersheba': (34.790372, 31.249257),
                     'nazareth': (35.297864, 32.701897),
                     'gaza': (34.462414, 31.506374),
                     # 'hebron': (35.101843, 31.532422),
                     'aman': (35.927025, 31.962766),
                     'nablus': (35.251627, 32.226124),
                     'kiryat shmona': (35.570337, 33.210001),
                     'kiryat gat': (34.771532, 31.609439),
                     'netania': (34.850431, 32.311543),
                     # 'eilat': (34.945825, 29.553369)
                     })

israel_utm = CityVector({city: lonlat_to_utm(*coords) for city, coords in israel.cities.items()})


def print_per_line(arr):
    print('\n'.join([str(m) for m in arr]))
    

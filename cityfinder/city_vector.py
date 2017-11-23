import itertools
from copy import copy, deepcopy

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
    print('\n'.join([str(m) for m in potential_matches]))

    # collect distance ratios:
    print('\n\ndistance ratios:')
    ratios = [cities1.dist(*pair1) / cities2.dist(*pair2) for pair1, pair2 in potential_matches]
    print(sorted(ratios))




israel = CityVector({'tlv': (34.777303, 32.076025),
                     'j-m': (35.207415, 31.768136),
                     'haifa': (34.997550, 32.804523),
                     'tiberias': (35.533599, 32.793664),
                     'beersheba': (34.790372, 31.249257),
                     'eilat': (34.945825, 29.553369)})

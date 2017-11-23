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

    def _azimuth(self, city1, city2):
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return np.rad2deg(np.math.atan2(y2 - y1, x2 - x1))

    def _dist(self, city1, city2):
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
        azimuths = {(city1, city2): self._azimuth(city1, city2)
                    for city1 in self.cities for city2 in self.cities if city1 != city2}
        return azimuths


israel = CityVector({'tlv': (34.777303, 32.076025),
                     'j-m': (35.207415, 31.768136),
                     'haifa': (34.997550, 32.804523),
                     'tiberias': (35.533599, 32.793664),
                     'beersheba': (34.790372, 31.249257),
                     'eilat': (34.945825, 29.553369)})
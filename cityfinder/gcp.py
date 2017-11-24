import numpy as np
import json

from affine import Affine

import shapely.geometry


class GCPError(Exception):
    pass


class GCP(object):
    """ single ground control point. """
    def __init__(self, lon, lat, alt, image_id, x, y):
        """
        :param lon: [deg]
        :param lat: [deg]
        :param alt: [m], optional
        :param image_id: any string, optional
        :param x: pixel in image
        :param y: pixel in image
        """
        self.lon = lon
        self.lat = lat
        self.alt = alt
        self.image_id = image_id
        self.x = x
        self.y = y

    def to_dict(self):
        return {
            'lon': self.lon,
            'lat': self.lat,
            'alt': self.alt,
            'image_id': self.image_id,
            'x': self.x,
            'y': self.y,
        }

    @classmethod
    def from_dict(cls, gcp_dict):
        alt = gcp_dict.get('alt', None)
        return GCP(float(gcp_dict['lon']),
                   float(gcp_dict['lat']),
                   float(alt) if alt is not None else None,
                   gcp_dict.get('image_id', None),
                   float(gcp_dict['x']),
                   float(gcp_dict['y']))

    def __repr__(self):
        return self.to_dict().__repr__()

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def copy_with(self, **kwargs):
        """Get a copy of self with some attributes changed."""
        init_args = {
            'lat': self.lat, 'lon': self.lon, 'alt': self.alt, 'image_id': self.image_id, 'x': self.x, 'y': self.y
        }
        init_args.update(kwargs)
        return GCP(**init_args)
#
#     def transform(self, tf):
#         """ transforms pixel by tf (affine or homography). lat,lon remain intact. """
#         if isinstance(tf, Affine):
#             transformed = tf * [self.x, self.y]
#         elif isinstance(tf, Homography):
#             transformed = tf([self.x, self.y])
#         else:
#             raise GCPError('%s transformation not supported' % type(tf))
#         return self.copy_with(x=transformed[0], y=transformed[1])
#
#     def error_wrt_transformation(self, tf):
#         """ returns error[pix] of reprojection of lat,lon to image by tf. """
#         if isinstance(tf, Affine):
#             orig = np.array([self.x, self.y])
#             projected = (~tf) * [self.lon, self.lat]
#         elif isinstance(tf, Homography):
#             orig = np.array([self.x, self.y, 1])
#             projected = (~tf).apply(self.lon, self.lat)
#         else:
#             raise GCPError('%s transformation not supported' % type(tf))
#         error = np.linalg.norm(orig - projected)
#         return error
#
#     @classmethod
#     def from_hugin_line(cls, line, map, first_image_is_raster=True):
#         """
#         :param line:
#         :param map: reference to which gcps were compared
#         :param first_image_is_raster: true if hugin specified img_0 as raster for gcps, img_1 as map (reference)
#         :return: GCP
#         """
#         fields = line.split(' ')
#         raster_x, raster_y = float(fields[3][1:]), float(fields[4][1:])  # x, y in hugin notations
#         map_x, map_y = float(fields[5][1:]), float(fields[6][1:])  # X, Y in hugin notations
#         if not first_image_is_raster:
#             raster_x, raster_y, map_x, map_y = map_x, map_y, raster_x, raster_y
#
#         pt_world = map.to_world(shapely.geometry.Point(map_x, map_y))
#         pt_lonlat = rastile.utils.geography.transform(pt_world, map.crs, CRS_GEOGRAPHIC)
#         return GCP(pt_lonlat.x, pt_lonlat.y, None, '', raster_x, raster_y)
#
#
# def gcps_from_dict(gcps_dict):
#     """ Returns list of GCPs. """
#     return [GCP.from_dict(gcp_dict) for gcp_dict in gcps_dict]
#
#
# def read_gcps(filepath):
#     with open(filepath, 'r') as gcps_file:
#         gcps_json = json.loads(gcps_file.read())
#         return gcps_from_dict(gcps_json['GCP'])
#
#
# def gcps_to_dict(gcps):
#     return {'GCP': [gcp.to_dict() for gcp in gcps]}
#
#
# def write_gcps(gcps, filepath):
#     rastile.utils.file.assert_dir_exists(filepath)
#     with open(filepath, 'w') as gcps_file:
#         gcps_dict = gcps_to_dict(gcps)
#         json.dump(gcps_dict, gcps_file, indent=4, sort_keys=True)
#
#
# def gcps_geojson(gcps):
#     points = [shapely.geometry.Point(gcp.lon, gcp.lat) for gcp in gcps]
#     multipoint = shapely.geometry.MultiPoint(points)
#     geojson = json.dumps(shapely.geometry.mapping(multipoint), indent=4, sort_keys=True)
#     return geojson
#
#
# def write_gcps_geojson(gcps, filepath):
#     rastile.utils.file.assert_dir_exists(filepath)
#     with open(filepath, 'w') as gcps_file:
#         geojson = gcps_geojson(gcps)
#         gcps_file.write(geojson)

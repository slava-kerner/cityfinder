import shapely
import shapely.ops, shapely.geometry
import pyproj
from affine import Affine

import numpy as np


p_geographic = pyproj.Proj(proj='latlong', datum='WGS84')
p_utm = pyproj.Proj(proj='utm', zone=36, datum='WGS84')
GEOGRAPHIC = {'init': 'EPSG:4326'}
UTM = {'init': 'EPSG:32636'}


def transform(geom, src_crs, dest_crs, src_affine=None, dst_affine=None):
    """
    Transform a shapely geometry from src_crs to dest_crs.
    If affine is not None, also move to pixel coordinates.
    """
    if src_affine is not None:
        geom = shapely.ops.transform(lambda r, q: ~src_affine * (r, q), geom)

    if src_crs != dest_crs:
        original = src_crs if isinstance(src_crs, pyproj.Proj) else pyproj.Proj(src_crs)
        destination = dest_crs if isinstance(dest_crs, pyproj.Proj) else pyproj.Proj(dest_crs)

        projected = shapely.ops.transform(lambda r, q: pyproj.transform(original, destination, r, q), geom)
    else:
        projected = geom

    if dst_affine is not None:
        if projected.type == 'Point':  # for some reason shapely.ops.transform below doesn't work for Point
            p = dst_affine * (projected.x, projected.y)
            projected = shapely.geometry.Point(p[0], p[1])
        else:
            projected = shapely.ops.transform(lambda r, q: dst_affine * (r, q), projected)

    return projected


def fit_affine(gcps, crs=p_geographic):
    if len(gcps) < 3:
        raise np.linalg.linalg.LinAlgError('Too few gcps, eq.system underdetermined. ')

    pts = [transform(shapely.geometry.Point(gcp.lon, gcp.lat), p_geographic, crs) for gcp in gcps]

    points_world = np.array([[pt.x for pt in pts], [pt.y for pt in pts]])
    points_image = np.array([[gcp.x for gcp in gcps], [gcp.y for gcp in gcps], [1] * len(gcps)])
    affine = np.linalg.lstsq(points_image.transpose(), points_world.transpose())
    error = np.linalg.norm(affine[1]) if affine[1] is not None else 0
    affine = affine[0].transpose()
    affine = Affine(*np.ravel(affine))
    return affine, error


def lonlat_to_utm(lon, lat):
    x, y = pyproj.transform(p_geographic, p_utm, lon, lat)
    return x, y


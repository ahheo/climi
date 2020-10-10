from .ffff import flt_l
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.path import Path


__all__ = ['gpd_read',
           'poly_to_path_']


gpd_read = gpd.GeoDataFrame.from_file


def _i_poly_to_path_(poly):                                                        
    if hasattr(poly, 'exterior'):
        return Path(list(poly.exterior.coords))
    else:
        if not hasattr(poly, 'geoms'):
            return None
        else:
            tmp = list(poly.geoms)
            return [_i_poly_to_path_(i) for i in tmp]


def poly_to_path_(poly):
    return flt_l([_i_poly_to_path_(i) for i in poly])

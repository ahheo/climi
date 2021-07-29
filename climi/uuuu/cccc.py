"""
>--#########################################################################--<
>-------------------------functions operating on cube-------------------------<
>--#########################################################################--<
* alng_axis_            : apply along axis
* area_weights_         : modified area_weights from iris
* ax_fn_mp_             : apply along axis mp
* axT_cube              : time axis of cube
* concat_cube_          : robust cube concater
* cubesv_               : save cube to nc with dim_t unlimitted
* cut_as_cube           : cut into the domain of another cube
* en_iqr_               : ensemble interquartile range
* en_max_               : ensemble max
* en_mean_              : ensemble mean
* en_min_               : ensemble min
* en_mxn_               : ensemble spread
* en_mm_cubeL_          : make ensemble cube for multimodels
* en_rip_               : ensemble (rxixpx) cube
* extract_byAxes_       : extraction with help of inds_ss_
* extract_month_cube    : extraction cube of month
* extract_period_cube   : extraction cube within [y0, y1]
* extract_season_cube   : extraction cube of season
* extract_win_cube      : extraction within a window (daily)
* f_allD_cube           : iris analysis func over all dims of cube(L) (each)
* getGridAL_cube        : grid_land_area
* getGridA_cube         : grid_area from file or calc with basic assumption
* get_gwl_y0_           : first year of 30-year window of global warming level
* get_loa_              : longitude/latitude coords of cube
* get_loa_dim_          : modified _get_lon_lat_coords from iris
* get_loa_pts_2d_       : 2d longitude/latitude points (from coord or meshed)
* get_xy_dim_           : horizontal spatial dim coords
* get_xyd_cube          : cube axes of xy dims
* guessBnds_cube        : bounds of dims points
* half_grid_            : points between grids
* initAnnualCube_       : initiate annual cube
* inpolygons_cube       : points if inside polygons
* intersection_         : cube intersection with lon/lat range
* isMyIter_             : Iterable with items as cube/ndarray
* kde_cube              : kernal distribution estimation over all cube data
* lccs_m2km_            : change LambfortComfort unit
* maskLS_cube           : mask land or sea area
* maskNaN_cube          : mask nan points in a cube
* maskPOLY_cube         : mask area in respect of polygons
* max_cube              : max of cube(L) data (each)
* max_cube_             : max of cube(L) data (all)
* merge_cube_           : robust cube merger
* min_cube              : min of cube(L) data (each)
* min_cube_             : min of cube(L) data (all)
* minmax_cube           : minmax of cube(L) data (each)
* minmax_cube_          : minmax of cube(L) data (all)
* nTslice_cube          : slices along a no-time axis
* nine_points_cube      : extract nine points cube centered at a given point
* pSTAT_cube            : period statistic (month, season, year)
* pst_                  : post-rename/reunits cube(L)
* pp_cube               : pth and 100-pth of cube(L) data (each)
* purefy_cubeL_         : prepare for concat or merge
* repair_cs_            : bug fix for save cube to nc
* repair_lccs_          : bug fix for save cube to nc (LamgfortComfort)
* replace_coord_        : replace coord data according to coord.name()
* rgCount_cube          : regional count
* rgCount_poly_cube     : regional count over in polygon only
* rgF_cube              : regional function
* rgF_poly_cube         : regional function over in polygon only
* rgMean_cube           : regional mean
* rgMean_poly_cube      : regional mean over in polygon only
* ri_cube               : recurrence period for each grid space
* rm_sc_cube            : remove scalar coords
* rm_t_aux_cube         : remove time-related aux_coords
* rm_yr_doy_cube        : opposite action of yr_doy_cube
* seasonyr_cube         : season_year auxcoord
* slice_back_           : slice back to parent (1D)
* unique_yrs_of_cube    : unique year points of cube
* y0y1_of_cube          : starting and ending year of cube
* yr_doy_cube           : year and day-of-year auxcoord
...

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 06.09.2019
Date last modified: 11.11.2020
           comment: add function half_grid_, move remaping functions to rgd.py
"""

import numpy as np
import iris
import iris.coord_categorisation as cat
from iris.experimental.equalise_cubes import equalise_attributes

import cf_units
import warnings
from datetime import datetime

from .ffff import *


__all__ = ['alng_axis_',
           'area_weights_',
           'ax_fn_mp_',
           'axT_cube',
           'concat_cube_',
           'cubesv_',
           'cut_as_cube',
           'en_iqr_',
           'en_max_',
           'en_mean_',
           'en_min_',
           'en_mxn_',
           'en_mm_cubeL_',
           'en_rip_',
           'extract_byAxes_',
           'extract_month_cube',
           'extract_period_cube',
           'extract_season_cube',
           'extract_win_cube',
           'f_allD_cube',
           'getGridAL_cube',
           'getGridA_cube',
           'get_gwl_y0_',
           'get_loa_',
           'get_loa_dim_',
           'get_loa_pts_2d_',
           'get_xy_dim_',
           'get_xyd_cube',
           'guessBnds_cube',
           'half_grid_',
           'initAnnualCube_',
           'inpolygons_cube',
           'intersection_',
           'isMyIter_',
           'kde_cube',
           'lccs_m2km_',
           'maskLS_cube',
           'maskNaN_cube',
           'maskPOLY_cube',
           'max_cube',
           'max_cube_',
           'merge_cube_',
           'min_cube',
           'min_cube_',
           'minmax_cube',
           'minmax_cube_',
           'nTslice_cube',
           'nine_points_cube',
           'pSTAT_cube',
           'pst_',
           'pp_cube',
           'purefy_cubeL_',
           'repair_cs_',
           'repair_lccs_',
           'replace_coord_',
           'rgCount_cube',
           'rgCount_poly_cube',
           'rgF_cube',
           'rgF_poly_cube',
           'rgMean_cube',
           'rgMean_poly_cube',
           'ri_cube',
           'rm_sc_cube',
           'rm_t_aux_cube',
           'rm_yr_doy_cube',
           'seasonyr_cube',
           'slice_back_',
           'unique_yrs_of_cube',
           'y0y1_of_cube',
           'yr_doy_cube']


def slice_back_(cnd, c1d, ii, axis):
    """
    ... put 1D slice back to its parent CUBE/ARRAY ...

    Parsed arguments:
         cnd: parent CUBE/ARRAY that has multiple dimensions
         c1d: CUBE/ARRAY slice
          ii: slice # of c1d in iteration
        axis: axis of cnd to place c1d
    Returns:
        revised cnd
    """

    if isinstance(c1d, iris.cube.Cube):
        c1d = c1d.data
    if not isinstance(c1d, np.ndarray):
        c1d = np.asarray(c1d)
    if ((isinstance(cnd, iris.cube.Cube) or np.ma.isMaskedArray(cnd))
        and not np.ma.isMaskedArray(c1d)):
        c1d = np.ma.masked_array(c1d, np.isnan(c1d))
    emsg = "slice NOT matched its parent along axis({})."
    if axis is None:
        if c1d.size != 1:
            raise Exception(emsg.format(axis))
    else:
        axis = cyl_(axis, cnd.ndim)
        axis = sorted(axis) if isIter_(axis, xi=(int, np.integer)) else axis
        if not np.all(np.asarray(cnd.shape)[axis] == np.asarray(c1d.shape)):
            raise Exception(emsg.format(axis))
    ind = ind_shape_i_(cnd.shape, ii, axis)
    if isinstance(cnd, iris.cube.Cube):
        cnd.data[ind] = c1d
    elif np.ma.isMaskedArray(cnd):
        cnd[ind] = c1d
    else:
        cnd[ind_shape_i_(cnd.shape, ii, axis)] = c1d


def extract_byAxes_(cnd, axis, sl_i, *vArg):
    """
    ... extract CUBE/ARRAY by providing selection along axis/axes ...

    Parsed arguments:
         cnd: parent CUBE/ARRAY
        axis: along which for the extraction; axis name acceptable
        sl_i: slice, list, or 1d array of selected indices along axis
        vArg: any pairs of (axis, sl_i)
    Returns:
        revised cnd
    """

    if len(vArg)%2 != 0:
        raise Exception("arguments {!r} not interpretable!".format(vArg))

    if len(vArg) > 0:
        ax, sl = list(vArg[::2]), list(vArg[1::2])
        ax.insert(0, axis)
        sl.insert(0, sl_i)
    else:
        ax = [axis]
        sl = [sl_i]

    if isinstance(cnd, iris.cube.Cube):
        ax = [cnd.coord_dims(i)[0]
              if isinstance(i, (str, iris.coords.DimCoord)) else i
              for i in ax]

    nArg = [(i, j) for i, j in zip(ax, sl)]
    nArg = tuple(j for i in nArg for j in i)
    inds = inds_ss_(cnd.ndim, *nArg)

    return cnd[inds]


def isMyIter_(x):
    return isIter_(x,
                   xi=(np.ndarray, iris.cube.Cube),
                   XI=(np.ndarray, iris.cube.Cube, str, bytes))


def pst_(cube, name=None, units=None, var_name=None, attrU=None):
    if isinstance(cube, iris.cube.Cube):
        if name:
            cube.rename(name)
        if units:
            cube.units = units
        if var_name:
            cube.var_name = var_name
        if attrU:
            cube.attributes.update(attrU)
    elif isMyIter_(cube):
        for i in cube:
            pst_(i, name=name, units=units, var_name=var_name, attrU=attrU)


def axT_cube(cube):
    try:
        tc = cube.coord(axis='T', dim_coords=True)
        return cube.coord_dims(tc)[0]
    except:
        return None


def nTslice_cube(cube, n):
    nd, shp = cube.ndim, cube.shape
    ax_nT = [i for i in range(nd) if i not in cube.coord_dims('time')]
    if all([shp[i] < n for i in ax_nT]):
        return [cube]
    else:
        for i in reversed(ax_nT):
            if shp[i] > n:
                step = int(np.ceil(shp[i] / n))
                return [extract_byAxes_(cube, i, np.s_[ii:(ii + step)])
                        for ii in range(0, shp[i], step)]


def unique_yrs_of_cube(cube, ccsn='year', mmm=None):
    if isinstance(cube, iris.cube.Cube):
        c = cube.copy()
        ccs = [i.name() for i in c.coords()]
        if ccsn not in ccs:
            if 'season' in ccsn:
                if mmm:
                    seasonyr_cube(c, mmm, name=ccsn)
                else:
                    emsg = "'mmm' must not be None for adding coord {!r}!"
                    raise ValueError(emsg.format(ccsn))
            else:
                cat.add_year(c, 'time', name=ccsn)
        return np.unique(c.coord(ccsn).points)
    elif isIter_(cube, xi=iris.cube.Cube):
        return [unique_yrs_of_cube(i) for i in cube]
    else:
        raise TypeError("unknown type for the first argument!")


def y0y1_of_cube(cube, ccsn='year', mmm=None):
    if isinstance(cube, iris.cube.Cube):
        c = cube.copy()
        ccs = [i.name() for i in c.coords()]
        if ccsn not in ccs:
            if 'season' in ccsn:
                if mmm:
                    seasonyr_cube(c, mmm, name=ccsn)
                else:
                    emsg = "'mmm' must not be None for adding coord {!r}!"
                    raise ValueError(emsg.format(ccsn))
            else:
                cat.add_year(c, 'time', name=ccsn)
        elif mmm and 'season' in ccsn:
            c.remove_coord(ccsn)
            seasonyr_cube(c, mmm, name=ccsn)
        return list(c.coord(ccsn).points[[0, -1]])
    elif isIter_(cube, xi=iris.cube.Cube):
        yy = np.array([y0y1_of_cube(i) for i in cube])
        return [np.max(yy[:, 0]), np.min(yy[:, 1])]
    else:
        raise TypeError("unknown type for the first argument!")


def extract_period_cube(cube, y0, y1, yy=False, ccsn='year', mmm=None):
    """
    ... extract a cube within the period from year y0 to year y1 ...

    Parsed arguments:
        cube: a cube containing at least coord('time')
          y0: starting year
          y1: end year
          yy: True forcing output strictly starting/end at y0/y1
    Returns:
         c_y: a cube within the period
    """
    c = cube.copy()
    ccs = [i.name() for i in c.coords()]
    if ccsn not in ccs:
        if 'season' in ccsn:
            if mmm:
                seasonyr_cube(c, mmm, name=ccsn)
            else:
                emsg = "'mmm' must not be None for adding coord {!r}!"
                raise ValueError(emsg.format(ccsn))
        else:
            cat.add_year(c, 'time', name=ccsn)
    elif mmm and 'season' in ccsn:
        c.remove_coord(ccsn)
        seasonyr_cube(c, mmm, name=ccsn)
    cstrD = {ccsn: lambda x: y0 <= x <= y1}
    cstr = iris.Constraint(**cstrD)
    c_y = c.extract(cstr)
    if not (yy and (y0y1_of_cube(c_y) != [y0, y1] or
               not np.all(np.diff(unique_yrs_of_cube(c_y)) == 1))):
        return c_y


def extract_win_cube(cube, d, r=15):
    """
    ... extract a cube within a 2*r-day (2*r+1 days exactly) window centered
        at day-of-year = d ...

    Parsed arguments:
        cube: a cube containing at least coord('time')
           d: center of the 30-day window, day-of-year
           r: half window width (defaul 15)
    Returns:
        c_30: a cube within a 30-day window
    """
    c = cube.copy()
    try:
        cat.add_day_of_year(c, 'time', name='doy')
    except ValueError:
        pass
    x1, x2 = cyl_(d - r, 365), cyl_(d + r, 365)
    c_30 = c.extract(
               iris.Constraint(doy=lambda x:
                   x1 <= x <= x2 if x1 < x2 else not (x2 < x < x1)
                   )
               )
    #c_30.remove_coord('doy')
    return c_30


def extract_season_cube(cube, mmm, valid_season=True):
    """
    ... extract a cube of season named with continuous-months' 1st letters ...

    Parsed arguments:
         cube: cube/cubelist containing at least coord('time')
          mmm: continuous-months' 1st letters (for example, 'djf')
    Returns:
        ncube: cube/cubelist after extraction
    """
    if isinstance(cube, iris.cube.Cube):
        if 'season' in (i.name() for i in cube.coords()):
            ncube = cube.extract(iris.Constraint(season=mmm))
        else:
            c = cube.copy() # to avoid changing metadata of original cube
            try:
                cat.add_season_membership(c, 'time', mmm, name=mmm)
            except ValueError:
                c.remove_coord(mmm)
                cat.add_season_membership(c, 'time', mmm, name=mmm)
            c.coord(mmm).points = c.coord(mmm).points.astype(np.int)
            ncube = c.extract(iris.Constraint(**{mmm: True}))
        if valid_season and not ismono_(mmmN_(mmm)):
            y0, y1 = y0y1_of_cube(ncube, ccsn='seasonyr', mmm=mmm)
            ncube = extract_period_cube(ncube, y0 + 1, y1 - 1,
                                        ccsn='seasonyr', mmm=mmm)
        return ncube
    elif isMyIter_(cube):
        cl = [extract_season_cube(i, mmm) for i in cubeL]
        return iris.cube.CubeList(cl)
    else:
        raise TypeError("unknown type for the first argument!")


def extract_month_cube(cube, Mmm):
    c = cube.copy()
    try:
        cat.add_month(c, 'time', name='month')
    except ValueError:
        pass
    ncube = c.extract(iris.Constraint(month=Mmm.capitalize()))
    return ncube


def f_allD_cube(cube, rg=None, f='MAX', **f_opts):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if isinstance(cube, iris.cube.Cube):
        if rg:
            cube = intersection_(cube, **rg)
        f_ = eval('iris.analysis.{}'.format(f.upper()))
        c = cube.collapsed(cube.dim_coords, f_, **f_opts)
        return c.data
    elif isMyIter_(cube):
        return np.asarray([f_allD_cube(i, rg=rg, f=f, **f_opts)
                           for i in cube])
    else:
        return np.nan


def min_cube(cube, rg=None):
    return f_allD_cube(cube, rg=rg, f='MIN')


def max_cube(cube, rg=None):
    return f_allD_cube(cube, rg=rg)


def minmax_cube(cube, rg=None):
    return np.asarray([min_cube(cube, rg=rg), max_cube(cube, rg=rg)])


def pp_cube(cube, rg=None, p=10):
    return np.asarray(
        [f_allD_cube(cube, rg=rg, f='PERCENTILE', percent=p),
         f_allD_cube(cube, rg=rg, f='PERCENTILE', percent=100 - p)]
        )


def min_cube_(cube, rg=None):
    return np.nanmin(min_cube(cube, rg=rg))


def max_cube_(cube, rg=None):
    return np.nanmax(max_cube(cube, rg=rg))


def minmax_cube_(cube, rg=None, p=None):
    if p:
        mms = pp_cube(cube, rg=rg, p=p)
    else:
        mms = minmax_cube(cube, rg=rg)
    return (np.nanmin(mms), np.nanmax(mms))


def get_xyd_cube(cube, guess_lst2=True):
    xc, yc = get_xy_dim_(cube)
    if xc is None:
        if guess_lst2:
            warnings.warn("missing 'x' or 'y' dimcoord in input cube; "
                          "guess last two as xyd.")
            return tuple(cyl_([-2, -1], cube.ndim))
        else:
            raise Exception("missing 'x' or 'y' dimcoord in input cube!")
    else:
        xyd = list(cube.coord_dims(yc) + cube.coord_dims(xc))
        xyd.sort()
        return tuple(xyd)


def _get_xy_lim(cube, longitude=None, latitude=None):
    xc, yc = get_xy_dim_(cube)
    if xc is None or yc is None:
        raise Exception("missing 'x' or 'y' dimcoord in input cube!")
    lo, la = cube.coord('longitude'), cube.coord('latitude')
    if longitude is None:
        longitude = [lo.points.min(), lo.points.max()]
    if latitude is None:
        latitude = [la.points.min(), la.points.max()]
    if xc == lo and yc == la:
        xyl = {xc.name(): longitude, yc.name(): latitude}
    else:
        xd = cube.coord_dims(lo).index(np.intersect1d(cube.coord_dims(lo),
                                                      cube.coord_dims(xc)))
        yd = cube.coord_dims(lo).index(np.intersect1d(cube.coord_dims(lo),
                                                      cube.coord_dims(yc)))
        a_ = ind_inRange_(lo.points, *longitude, r_=360)
        b_ = ind_inRange_(la.points, *latitude)
        c_ = np.logical_and(a_, b_)
        xi, yi = np.where(np.any(c_, axis=yd)), np.where(np.any(c_, axis=xd))
        xv, yv = xc.points[xi], yc.points[yi]
        if np.any(np.diff(xi[0]) != 1):
            if xc.circular and xc.units.modulus:
                rb = xc.points[np.where(np.diff(xi[0]) != 1)[0] + 1]
                xv = cyl_(xv, rb, rb - xc.units.modulus)
            else:
                raise Exception("limits given outside data!")
        xll = [xv.min(), xv.max()]
        yll = [yv.min(), yv.max()]
        xyl = {xc.name(): xll, yc.name(): yll}
    if xc.units.modulus:
        return xyl
    else:
        return (xc.name(), ind_inRange_(xc.points, *xll),
                yc.name(), ind_inRange_(yc.points, *yll))


def _get_ind_lolalim(cube, longitude=None, latitude=None):
    xyd = get_xyd_cube(cube)
    lo, la = get_loa_pts_2d_(cube)
    if longitude is None:
        longitude = [lo.min(), lo.max()]
    if latitude is None:
        latitude = [la.min(), la.max()]
    a_ = ind_inRange_(lo, *longitude, r_=360)
    b_ = ind_inRange_(la, *latitude)
    c_ = np.logical_and(a_, b_)
    return robust_bc2_(c_, cube.shape, xyd)


def intersection_(cube, longitude=None, latitude=None):
    """
    ... intersection by range of longitude/latitude ...
    """
    kwArgs=dict()
    if longitude is not None:
        kwArgs.update(dict(longitude=longitude))
    if latitude is not None:
        kwArgs.update(dict(latitude=latitude))
    if cube.coord('latitude').ndim == 1:
        return cube.intersection(**kwArgs)
    else:
        xyl = _get_xy_lim(cube, **kwArgs)
        if isinstance(xyl, dict):
            return cube.intersection(**xyl)
        else:
            return extract_byAxes_(cube, *xyl)


def seasonyr_cube(cube, mmm, name='seasonyr'):
    """
    ... add season_year auxcoords to a cube especially regarding
        specified season ...
    """
    if isinstance(cube, iris.cube.Cube):
        if isinstance(mmm, str):
            seasons = (mmm, rest_mns_(mmm))
        elif (isinstance(mmm, (list, tuple)) and
              sorted(''.join(mmm)) == sorted('djfmamjjason')):
            seasons = mmm
        else:
            raise Exception("unknown seasons {!r}!".format(mmm))
        try:
            cat.add_season_year(cube, 'time', name=name, seasons=seasons)
        except ValueError:
            cube.remove_coord(name)
            cat.add_season_year(cube, 'time', name=name, seasons=seasons)
    elif isIter_(cube, xi=(iris.cube.Cube, iris.cube.CubeList, tuple, list)):
        for c in cube:
            seasonyr_cube(c, mmm, name=name)


def yr_doy_cube(cube):
    """
    ... add year, day-of-year auxcoords to a cube ...
    """
    if isinstance(cube, iris.cube.Cube):
        try:
            cat.add_year(cube, 'time', name='year')
        except ValueError:
            pass
        else:
            cube.coord('year').attributes = {}
        try:
            cat.add_day_of_year(cube, 'time', name='doy')
        except ValueError:
            pass
        else:
            cube.coord('doy').attributes = {}
    elif isIter_(cube, xi=(iris.cube.Cube, iris.cube.CubeList, tuple, list)):
        for c in cube:
            yr_doy_cube(c)


def rm_yr_doy_cube(cube):
    """
    ... remove year, day-of-year auxcoords from a cube ...
    """
    try:
        cube.remove_coord('year')
    except iris.exceptions.CoordinateNotFoundError:
        pass
    try:
        cube.remove_coord('doy')
    except iris.exceptions.CoordinateNotFoundError:
        pass


def rm_t_aux_cube(cube, keep=None):
    """
    ... remove time-related auxcoords from a cube or a list of cubes ...
    """
    tauxL = ['year', 'month', 'season', 'day', 'doy', 'hour', 'yr']
    def _isTCoord(x):
        return any((i in x.name() for i in tauxL)) or isSeason_(x.name())
    if isinstance(cube, iris.cube.Cube):
        for i in cube.aux_coords:
            if keep is None:
                isTaux = _isTCoord(i)
            elif isIter_(keep):
                isTaux = _isTCoord(i) and i.name() not in keep
            else:
                isTaux = _isTCoord(i) and i.name() != keep
            if isTaux:
                cube.remove_coord(i)
    elif isMyIter_(cube):
        for c in cube:
            rm_t_aux_cube(c)
    else:
        raise TypeError('Input should be CUBE or iterable CUBEs!')


def rm_sc_cube(cube):
    if isinstance(cube, iris.cube.Cube):
        for i in cube.coords():
            if len(cube.coord_dims(i)) == 0:
                cube.remove_coord(i)
    elif isMyIter_(cube):
        for c in cube:
            rm_sc_cube(c)
    else:
        raise TypeError('Input should be CUBE or Iterable CUBEs!')


def guessBnds_cube(cube):
    """
    ... guess bounds of dims of cube if not exist ...
    """
    for i in cube.dim_coords:
        try:
            i.guess_bounds()
        except ValueError:
            pass


def get_loa_dim_(cube):
    """
    ... get lon and lat coords (dimcoords only) ...
    """
    lat_coords = [coord for coord in cube.dim_coords
                  if "latitude" in coord.name()]
    lon_coords = [coord for coord in cube.dim_coords
                  if "longitude" in coord.name()]
    if len(lat_coords) > 1 or len(lon_coords) > 1:
        raise ValueError(
            "Calling `get_loa_dim_` with multiple lat or lon coords"
            " is currently disallowed")
    lat_coord = lat_coords[0]
    lon_coord = lon_coords[0]
    return (lon_coord, lat_coord)


def get_xy_dim_(cube, guess_lst2=True):
    try:
        return (cube.coord(axis='X', dim_coords=True),
                cube.coord(axis='Y', dim_coords=True))
    except:
        if guess_lst2 and cube.ndim > 1:
            return(cube.coord(dimensions=cyl_(-1, cube.ndim), dim_coords=True),
                   cube.coord(dimensions=cyl_(-2, cube.ndim), dim_coords=True))
        else:
            return (None, None)


def get_loa_(cube):
    try:
        lo, la = cube.coord('longitude'), cube.coord('latitude')
        lo.convert_units('degrees')
        la.convert_units('degrees')
        return (lo, la)
    except:
        return (None, None)


def area_weights_(cube, normalize=False):
    """
    ... revised iris.analysis.cartography.area_weights to ignore lon/lat in
        auxcoords ...
    """
    from iris.analysis.cartography import DEFAULT_SPHERICAL_EARTH_RADIUS, \
                                     DEFAULT_SPHERICAL_EARTH_RADIUS_UNIT, \
                                                          _quadrant_area
    # Get the radius of the earth
    cs = cube.coord_system("CoordSystem")
    if isinstance(cs, iris.coord_systems.GeogCS):
        if cs.inverse_flattening != 0.0:
            warnings.warn("Assuming spherical earth from ellipsoid.")
        radius_of_earth = cs.semi_major_axis
    elif (isinstance(cs, iris.coord_systems.RotatedGeogCS) and
            (cs.ellipsoid is not None)):
        if cs.ellipsoid.inverse_flattening != 0.0:
            warnings.warn("Assuming spherical earth from ellipsoid.")
        radius_of_earth = cs.ellipsoid.semi_major_axis
    else:
        warnings.warn("Using DEFAULT_SPHERICAL_EARTH_RADIUS.")
        radius_of_earth = DEFAULT_SPHERICAL_EARTH_RADIUS

    # Get the lon and lat coords and axes
    try:
        lon, lat = get_loa_dim_(cube)
    except IndexError:
        raise ValueError('Cannot get latitude/longitude '
                         'coordinates from cube {!r}.'.format(cube.name()))

    if lat.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lat)
    if lon.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lon)

    lat_dim = cube.coord_dims(lat)
    lat_dim = lat_dim[0] if lat_dim else None

    lon_dim = cube.coord_dims(lon)
    lon_dim = lon_dim[0] if lon_dim else None

    if not (lat.has_bounds() and lon.has_bounds()):
        msg = ("Coordinates {!r} and {!r} must have bounds to determine "
               "the area weights.".format(lat.name(), lon.name()))
        raise ValueError(msg)

    # Convert from degrees to radians
    lat = lat.copy()
    lon = lon.copy()

    for coord in (lat, lon):
        if coord.units in (cf_units.Unit('degrees'),
                           cf_units.Unit('radians')):
            coord.convert_units('radians')
        else:
            msg = ("Units of degrees or radians required, coordinate "
                   "{!r} has units: {!r}".format(coord.name(),
                                                 coord.units.name))
            raise ValueError(msg)
    # Create 2D weights from bounds.
    # Use the geographical area as the weight for each cell
    ll_weights = _quadrant_area(lat.bounds,
                                lon.bounds, radius_of_earth)

    # Normalize the weights if necessary.
    if normalize:
        ll_weights /= ll_weights.sum()

    # Now we create an array of weights for each cell. This process will
    # handle adding the required extra dimensions and also take care of
    # the order of dimensions.
    broadcast_dims = [x for x in (lat_dim, lon_dim) if x is not None]
    wshape = []
    for idim, dim in zip((0, 1), (lat_dim, lon_dim)):
        if dim is not None:
            wshape.append(ll_weights.shape[idim])
    ll_weights = ll_weights.reshape(wshape)
    broad_weights = iris.util.broadcast_to_shape(ll_weights,
                                                 cube.shape,
                                                 broadcast_dims)

    return broad_weights


def cut_as_cube(cube0, cube1):
    """
    ... cut cube1 with the domain of cube0 ...
    """
    xc1, yc1 = get_xy_dim_(cube1)
    xc0, yc0 = get_xy_dim_(cube0)
    xn, yn = xc1.name(), yc1.name()
    xe = np.min(np.abs(np.diff(xc1.points))) / 2
    ye = np.min(np.abs(np.diff(yc1.points))) / 2
    x0, x1 = np.min(xc0.points), np.max(xc0.points)
    y0, y1 = np.min(yc0.points), np.max(yc0.points)
    return extract_byAxes_(cube1,
                           xn, ind_inRange_(xc1.points,
                                            x0 - xe, x1 + xe, side=0),
                           yn, ind_inRange_(yc1.points,
                                            y0 - ye, y1 + ye, side=0))


def maskLS_cube(cubeD, sftlf, LorS='S', thr=0):
    """
    ... mask sea/land area ...

    Parsed arguments:
        cubeD: DATA cube to be masked
        sftlf: land area fraction; at least covering entire cubeD
         LorS: 'land' or 'sea' to be masked (default 'sea')
          thr: sftlf value <= thr as not land area (default 0)
    """
    LList = ['L', 'LAND']
    SList = ['S', 'O', 'W', 'SEA', 'OCEAN', 'WATER']
    if LorS.upper() not in (LList + SList):
        raise ValueError("Variable 'LorS' not interpretable!")
    sftlf_ = cut_as_cube(cubeD, sftlf)
    ma_0 = sftlf_.data <= thr
    if LorS.upper() in LList:
        ma_0 = ~ma_0
    ma_ = np.broadcast_to(ma_0, cubeD.shape)
    cubeD = iris.util.mask_cube(cubeD, ma_)


def getGridA_cube(cubeD, areacella=None):
    """
    ... get grid_area of cube ...
    """
    if areacella:
        ga_ = iris.util.squeeze(areacella)
        if ga_.ndim != 2:
            return getGridA_cube(cubeD)
        ga = cut_as_cube(cubeD, ga_).data
        try:
            ga = robust_bc2_(ga, cubeD.shape, get_xyd_cube(cubeD))
            return ga
        except:
            return getGridA_cube(cubeD)
    else:
        try:
            guessBnds_cube(cubeD)
            ga = area_weights_(cubeD)
        except:
            ga = None
        return ga


def getGridAL_cube(cubeD, sftlf=None, areacella=None):
    """
    ... return grid_land_area of cube if sftlf provided, else return
        grid_area of cube ...
    """
    ga = getGridA_cube(cubeD, areacella)
    if sftlf is not None:
        sf_ = iris.util.squeeze(sftlf)
        if sf_.ndim != 2:
            raise Exception('NOT 2D area-cube!')
        sf = cut_as_cube(cubeD, sf_).data
        sf = robust_bc2_(sf, cubeD.shape, get_xyd_cube(cubeD))
        if ga is None:
            return np.ones(cubeD.shape) * sf / 100
        else:
            return ga * sf / 100.
    else:
        return ga


def rgF_cube(cubeD, function, rgD=None, **functionD):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if rgD:
        ind = _get_ind_lolalim(cubeD, **rgD)
        tmp = iris.util.mask_cube(cubeD.copy(), ~ind)
    else:
        tmp = cubeD
    xc, yc = get_xy_dim_(tmp)
    return tmp.collapsed([xc, yc], function, **functionD)


def rgF_poly_cube(cubeD, poly, function, **functionD):
    #warnings.filterwarnings("ignore", category=UserWarning)
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    tmp = iris.util.mask_cube(cubeD.copy(), ~ind)
    xc, yc = get_xy_dim_(tmp)
    return tmp.collapsed([xc, yc], function, **functionD)


def rgCount_cube(cubeD, sftlf=None, areacella=None, rgD=None, function=None):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if function is None:
        function = lambda values: values > 0
        warnings.warn("'function' not provided; count values greater than 0.")
    xyd = get_xyd_cube(cubeD)
    ga0 = getGridA_cube(cubeD, areacella)
    ga0 = np.ones(cubeD.shape) if ga0 is None else ga0
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    ga = np.ones(cubeD.shape) if ga is None else ga
    if rgD:
        ind = _get_ind_lolalim(cubeD, **rgD)
        ga0 *= ind
        ga *= ind
    umsk = ~cubeD.data.mask if (np.ma.isMaskedArray(cubeD.data) and
                                np.ma.is_masked(cubeD.data)) else 1
    sum0 = np.sum(ga0 * umsk, axis=xyd)
    if np.any(sum0 == 0):
        raise Exception("empty slice encountered.")
    data = np.sum(function(cubeD.data) * ga, axis=xyd) * 100 / sum0
    xc, yc = get_xy_dim_(cubeD)
    tmp = cubeD.collapsed([xc, yc], iris.analysis.MEAN)
    return tmp.copy(data)


def rgCount_poly_cube(cubeD, poly, sftlf=None, areacella=None, function=None,
                      **kwArgs):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if function is None:
        function = lambda values: values > 0
        warnings.warn("'function' not provided; count values greater than 0.")
    xyd = get_xyd_cube(cubeD)
    ga0 = getGridA_cube(cubeD, areacella)
    ga0 = np.ones(cubeD.shape) if ga0 is None else ga0
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    ga = np.ones(cubeD.shape) if ga is None else ga
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    ga0 *= ind
    ga *= ind
    sum0 = np.sum(ga0, axis=xyd)
    if np.any(sum0 == 0):
        raise Exception("empty slice encountered.")
    data = np.sum(function(cubeD.data) * ga, axis=xyd) * 100 / sum0
    xc, yc = get_xy_dim_(cubeD)
    tmp = cubeD.collapsed([xc, yc], iris.analysis.MEAN)
    return tmp.copy(data)


def rgMean_cube(cubeD, sftlf=None, areacella=None, rgD=None):
    """
    ... regional mean; try weighted if available ...
    """
    #warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    if rgD:
        ind = _get_ind_lolalim(cubeD, **rgD)
        if ga is None:
            ga = ind * np.ones(ind.shape)
        else:
            ga *= ind
    xc, yc = get_xy_dim_(cubeD)
    if ga is None:
        return cubeD.collapsed([xc, yc], iris.analysis.MEAN)
    else:
        return cubeD.collapsed([xc, yc], iris.analysis.MEAN, weights=ga)


def get_gwl_y0_(cube, gwl, pref=[1861, 1890]):
    c = pSTAT_cube(cube if cube.ndim == 1 else rgMean_cube(cube),
                  'MEAN', 'year')
    tref = extract_period_cube(c, *pref)
    tref = tref.collapsed('time', iris.analysis.MEAN).data
    def _G_tR(G, tR):
        if not isIter_(G):
            ind = np.where(rMEAN1d_(c.data, 30) >= G + tR)[0][0]
            return c.coord('year').points[ind]
        else:
            return [_G_tR(i, tR) for i in G]
    if c.ndim == 1:
        return _G_tR(gwl, tref)
    else:
        o = np.empty(tref.shape + np.array(gwl).shape)
        ax = c.coord_dims('year')[0]
        for i in range(nSlice_(c.shape, ax)):
            ind = ind_shape_i_(c.shape, i, ax)
            ind_ = ind_shape_i_(tref.shape, i, axis=None)
            ind__ = ind_shape_i_(o.shape, i,
                                 axis=-1 if np.array(gwl).shape else None)
            o[ind__] = np.array(_G_tR(gwl, tref[ind]))
        return o


def _inpolygons(poly, points, **kwArgs):
    if not isIter_(poly):
        ind = poly.contains_points(points, **kwArgs)
    elif len(poly) < 2:
        ind = poly[0].contains_points(points, **kwArgs)
    else:
        inds = [i.contains_points(points, **kwArgs) for i in poly]
        ind = np.logical_or.reduce(inds)
    return ind


def _isyx(cube):
    xc, yc = get_xy_dim_(cube)
    if xc is None:
        raise Exception("cube missing 'x' or 'y' coord")
    xcD, ycD = cube.coord_dims(xc)[0], cube.coord_dims(yc)[0]
    return ycD < xcD


def get_loa_pts_2d_(cube):
    lo_, la_ = get_loa_(cube)
    if lo_ is None or la_ is None:
        raise Exception("input cube(s) must have "
                        "longitude/latidute coords!")
    yx_ = _isyx(cube)
    if lo_.ndim != 2:
        if yx_:
            x, y = np.meshgrid(lo_.points, la_.points)
        else:
            y, x = np.meshgrid(la_.points, lo_.points)
    else:
        if yx_:
            x, y = lo_.points, la_.points
        else:
            x, y = lo_.points.T, la_.points.T
    return (x, y)


def inpolygons_cube(poly, cube, **kwArgs):
    x, y = get_loa_pts_2d_(cube)
    ind = _inpolygons(poly, np.vstack((x.ravel(), y.ravel())).T, **kwArgs)
    ind = robust_bc2_(ind.reshape(x.shape), cube.shape, get_xyd_cube(cube))
    return ind


def maskNaN_cube(cube):
    ind = np.isnan(cube.data)
    cube = iris.util.mask_cube(cube, ind)


def maskPOLY_cube(poly, cube, masked_out=True, **kwArgs):
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    ind = ~ind if masked_out else ind
    cube = iris.util.mask_cube(cube, ind)


def rgMean_poly_cube(cubeD, poly, sftlf=None, areacella=None, **kwArgs):
    #warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    xc, yc = get_xy_dim_(cubeD)
    if ga is None:
        ga = ind * np.ones(ind.shape)
    else:
        ga *= ind
    return cubeD.collapsed([xc, yc],
                           iris.analysis.MEAN,
                           weights=ga)


def _rm_extra_coords_cubeL(cubeL):
    l0 = [[ii.name() for ii in i.aux_coords] for i in cubeL]
    l1 = ouniqL_(flt_l(l0))
    l2 = [i for i in l1 if sum(np.array(flt_l(l0))==i) < len(cubeL)]
    if len(l2) != 0:
        for i, ii in zip(cubeL, l0):
            for iii in l2:
                if iii in ii:
                    i.remove_coord(iii)


def _get_xycoords(cube):
    """
    ... get xy (spatial) coords ...
    """
    xycn = ['lon', 'x_coord', 'x-coord', 'x coord',
            'lat', 'y_coord', 'y-coord', 'y coord']
    xycoords = [coord for coord in cube.coords()
                if any([i in coord.name() for i in xycn])]
    return xycoords


def _unify_1coord_points(cubeL, coord_name, thr=1e-10):
    epochs = {}
    emsg = "COORD {!r} can't be unified!".format(coord_name)
    emsg_ = "Bounds of COORD {!r} can't be unified!".format(coord_name)
    for c in cubeL:
        cc = c.coord(coord_name)
        d0 = epochs.setdefault('points', cc.points)
        dp = np.max(np.abs(cc.points - d0))
        if 0 < dp < thr:
            cc.points = d0
        elif dp > thr:
            raise Exception(emsg)
        if cc.has_bounds():
            d1 = epochs.setdefault('bounds', cc.bounds)
            db = np.max(np.abs(cc.bounds - d1))
            if 0 < db < thr:
                cc.bounds = d1
            elif db > thr:
                raise Exception(emsg_)


def _unify_xycoord_points(cubeL, thr=1e-10):
    ll_('cccc: _unify_xycoord_points() called')
    if len(cubeL) > 1:
        coord_names = [i.name() for i in _get_xycoords(cubeL[0])]
        for coord_name in coord_names:
            _unify_1coord_points(cubeL, coord_name, thr=thr)


def _unify_1coord_attrs(cubeL, coord_name):
    attrs = ['long_name', 'var_name', 'attributes', 'coord_system']
    epochs = {}
    for c in cubeL:
        cc = c.coord(coord_name)
        tp = cc.points.dtype
        tp = np.dtype(tp.str.replace('>', '<')) if '>' in tp.str else tp
        tmp = epochs.setdefault('dtype', tp)
        if tp != tmp:
            cc.points = cc.points.astype(tmp)
        if cc.has_bounds() and cc.bounds.dtype != tmp:
            cc.bounds = cc.bounds.astype(tmp)
        for i in attrs:
            tmp = epochs.setdefault(i, cc.__getattribute__(i))
            cc.__setattr__(i, tmp)


def _unify_coord_attrs(cubeL):
    ll_('cccc: _unify_coord_attrs() called')
    if len(cubeL) > 1:
        coord_names = [i.name() for i in cubeL[0].coords()]
        for coord_name in coord_names:
            _unify_1coord_attrs(cubeL, coord_name)


def _unify_time_units(cubeL):
    CLD0 = 'proleptic_gregorian'
    CLD = 'gregorian'
    clds = [c.coord('time').units.calendar for c in cubeL]
    if len(ouniqL_(clds)) > 1:
        for c in cubeL:
            ctu = c.coord('time').units
            if ctu.calendar == CLD0:
                ctu = cf_units.Unit(c.coord('time').units.origin, CLD)
    iris.util.unify_time_units(cubeL)


def _unify_dtype(cubeL, fst=False):
    ll_('cccc: _unify_dtype() called')
    tps = [c.dtype for c in cubeL]
    if fst:
        tp = tps[0]
    else:
        utps = np.unique(tps)
        tpi = [np.sum(np.asarray(tps) == i) for i in utps]
        tp = utps[np.argmax(tpi)]
    for c in cubeL:
        if c.dtype != tp:
            c.data = c.data.astype(tp)


def _unify_cellmethods(cubeL, fst=True):
    ll_('cccc: _unify_cellmethods() called')
    cms = [c.cell_methods for c in cubeL]
    if fst:
        cm = cms[0]
    else:
        ucms = np.unique(cms)
        cmi = [np.sum(np.asarray(cms) == i) for i in ucms]
        cm = utps[np.argmax(cmi)]
    for c in cubeL:
        if c.cell_methods != cm:
            c.cell_methods = cm


def purefy_cubeL_(cubeL):
    _rm_extra_coords_cubeL(cubeL)
    equalise_attributes(cubeL)
    _unify_time_units(cubeL)


def concat_cube_(cubeL, thr=1e-10):
    purefy_cubeL_(cubeL)
    try:
        o = cubeL.concatenate_cube()
    except iris.exceptions.ConcatenateError as ce_:
        if any(['Data types' in i for i in ce_.args[0]]):
            _unify_dtype(cubeL)
        if any(['Cube metadata' in i for i in ce_.args[0]]):
            _unify_cellmethods(cubeL)
        if any(['coordinates metadata differ' in i for i in ce_.args[0]]):
            if any(['height' in i for i in ce_.args[0]]):
                ll_("cccc: set COORD 'height' points to those of cubeL[0]")
                _unify_1coord_points(cubeL, 'height', thr=10)
            else:
                _unify_coord_attrs(cubeL)
        try:
            o = cubeL.concatenate_cube()
        except iris.exceptions.ConcatenateError as ce_:
            if any(['Expected only a single cube' in i for i in ce_.args[0]]):
                _unify_xycoord_points(cubeL, thr=thr)
            o = cubeL.concatenate_cube()
    return o


def merge_cube_(cubeL, thr=1e-10):
    purefy_cubeL_(cubeL)
    try:
        o = cubeL.merge_cube()
    except iris.exceptions.ConcatenateError as ce_:
        if any(['Data types' in i for i in ce_.args[0]]):
            _unify_dtype(cubeL)
        if any(['Cube metadata' in i for i in ce_.args[0]]):
            _unify_cellmethods(cubeL)
        if any(['coordinates metadata differ' in i for i in ce_.args[0]]):
            if any(['height' in i for i in ce_.args[0]]):
                ll_("cccc: set COORD 'height' points to those of cubeL[0]")
                _unify_1coord_points(cubeL, 'height', thr=10)
            else:
                _unify_coord_attrs(cubeL)
        try:
            o = cubeL.merge_cube()
        except iris.exceptions.ConcatenateError as ce_:
            if any(['Expected only a single cube' in i for i in ce_.args[0]]):
                _unify_xycoord_points(cubeL, thr=thr)
            o = cubeL.merge_cube()
    return o


def en_mxn_(eCube):
    """
    ... ensemble max of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        a = en_max_(eCube)
        b = en_min_(eCube)
        o = a - b
        o.rename(a.name())
        return o


def en_min_(eCube):
    """
    ... ensemble max of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        return eCube.collapsed('realization', iris.analysis.MIN)


def en_max_(eCube):
    """
    ... ensemble max of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        return eCube.collapsed('realization', iris.analysis.MAX)


def en_mean_(eCube, **kwArgs):
    """
    ... ensemble mean of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        return eCube.collapsed('realization', iris.analysis.MEAN, **kwArgs)


def en_iqr_(eCube):
    """
    ... ensemble interquartile range (IQR) of a cube (along dimcoord
        'realization') ...
    """
    if eCube.coord_dims('realization'):
        a = eCube.collapsed('realization', iris.analysis.PERCENTILE,
                            percent=75)
        b = eCube.collapsed('realization', iris.analysis.PERCENTILE,
                            percent=25)
        o = a - b
        o.rename(a.name())
        return o


def kde_cube(cube, **kde_opts):
    """
    ... kernal distribution estimate over all nomasked data ...
    """
    data = nanMask_(cube.data).flatten()
    data = data[~np.isnan(data)]
    data = data.astype(np.float64)
    return kde_(data, **kde_opts)


def _rip(cube):
    """
    ... get rxixpx from cube metadata ...
    """
    if 'parent_experiment_rip' in cube.attributes:
        return cube.attributes['parent_experiment_rip']
    elif 'driving_model_ensemble_member' in cube.attributes:
        return cube.attributes['driving_model_ensemble_member']
    else:
        return None


def en_rip_(cubeL):
    """
    ... ensemble cube over rxixpxs (along dimcoord 'realization') ...
    """
    for i, c in enumerate(cubeL):
        rip = _rip(c)
        rip = str(i) if rip is None else rip
        new_coord = iris.coords.AuxCoord(rip,
                                         long_name='realization',
                                         units='no_unit')
        c.add_aux_coord(new_coord)
        c.attributes = {}
    return cubeL.merge_cube()


def en_mm_cubeL_(cubeL, opt=0, cref=None):
    from .rgd import rgd_scipy_, rgd_iris_, rgd_li_opt0_
    tmpD = {}
    cl = []
    for i, c in enumerate(cubeL):
        c.attributes = {}
        if cref is None:
            cref = tmpD.setdefault('ref', c.copy())
        else:
            cref.attributes = {}
        if opt == 0:
            tmp = rgd_li_opt0_(c, cref)
        elif opt == 1:
            tmp = rgd_iris_(c, cref)
        elif opt == 2:
            tmp = rgd_scipy_(c, cref)
        else:
            raise ValueError('opt should be one of (0, 1, 2)!')
        a0 = tmpD.setdefault('a0', tmp)
        a = a0.copy(tmp.data)
        a.add_aux_coord(iris.coords.AuxCoord(np.int32(i),
                                             long_name='realization',
                                             units='no_unit'))
        cl.append(a)
    cl = iris.cube.CubeList(cl)
    eCube = cl.merge_cube()
    return eCube


def _func(func, ak_):
    arr, o0, args, kwargs = ak_
    try:
        if o0.data.mask or o0.mask:
            return None
    except AttributeError:
        pass
    return func(*arr, *args, **kwargs)


def ax_fn_ray_(arr, ax, func, out, *args, npr=32, **kwargs):
    import ray
    import psutil
    nproc = min(psutil.cpu_count(logical=False), npr)
    ray.init(num_cpus=nproc)

    if isMyIter_(out):
        o0 = out[0]
    else:
        o0 = out
    if not isinstance(o0, iris.cube.Cube):
        raise Exception("type of 'out' should be CUBE!")
    if not isinstance(arr, (tuple, list)):
        arr = (arr,)

    @ray.remote
    def f(i, arr):
        ind = ind_shape_i_(arr[0].shape, i, ax)
        try:
            if o0[ind][0].data.mask:
                return None
        except AttributeError:
            pass
        aaa = tuple([ii[ind] for ii in arr])
        return func(*aaa, *args, **kwargs)

    arr_id = ray.put(arr)
    tmp = [f.remote(i, arr_id) for i in range(nSlice_(arr[0].shape, ax))]
    XX = ray.get(tmp)

    _sb(XX, out, ax)


def ax_fn_mp_(arr, ax, func, out, *args, npr=32, **kwargs):
    import multiprocessing as mp
    nproc = min(mp.cpu_count(), npr)

    if isMyIter_(out):
        o0 = out[0]
    else:
        o0 = out
    if not isinstance(o0, (np.ndarray, iris.cube.Cube)):
        raise Exception("type of 'out' should be NDARRAY or CUBE!")
    if not isinstance(arr, (tuple, list)):
        arr = (arr,)

    P = mp.Pool(nproc)
    def _i(i, sl_=np.s_[:]):
        return ind_shape_i_(arr[0].shape, i, ax, sl_)
    X = P.starmap_async(_func, [(func, (
                                        tuple(ii[_i(i)] for ii in arr),
                                        arr[0][_i(i, sl_=0)],
                                        args,
                                        kwargs
                                       )
                                )
                                for i in range(nSlice_(arr[0].shape, ax))])
    XX = X.get()
    P.close()

    _sb(XX, out, ax)


def _sb(XX, out, ax):
    for i, o in enumerate(XX):
        if o is not None:
            _isb(i, o, out, ax)


def _isb(i, o, out, ax):
    def _get_nax(x):
        if x.ndim == 0:
            nax = None
        else:
            ax_ = tuple(ax) if isIter_(ax) else (ax,)
            nax = tuple(np.arange(ax_[0], ax_[0] + x.ndim))
        return nax
    if isMyIter_(out):
        for j, k in zip(out, o):
            if not isinstance(k, np.ndarray):
                k = np.asarray(k)
            slice_back_(j, k, i, _get_nax(k))
    else:
        if not isinstance(o, np.ndarray):
            o = np.asarray(o)
        slice_back_(out, o, i, _get_nax(o))


def alng_axis_(arrs, ax, func, out, *args, **kwargs):
    if isMyIter_(out):
        o0 = out[0]
    else:
        o0 = out
    if not isinstance(o0, (np.ndarray, iris.cube.Cube)):
        raise Exception("type of 'out' should be NDARRAY or CUBE!")
    if not isinstance(arrs, (list, tuple)):
        arrs = (arrs,)
    for i in range(nSlice_(arrs[0].shape, ax)):
        ind = ind_shape_i_(arrs[0].shape, i, ax)
        try:
            if o0[ind][0].data.mask:
                continue
        except AttributeError:
            pass
        aaa = [xxx[ind] for xxx in arrs]
        tmp = func(*aaa, *args, **kwargs)
        _isb(i, tmp, out, ax)


def initAnnualCube_(c0, y0y1, name=None, units=None, var_name=None,
                    long_name=None, attrU=None, mmm='j-d'):
    mmm = 'jfmamjjasond' if mmm == 'j-d' else mmm
    y0, y1 = y0y1
    ny = y1 - y0 + 1
    c = extract_byAxes_(c0, 'time', np.s_[:ny])
    rm_t_aux_cube(c)

    def _mm01():
        if isMonth_(mmm):
            m0 = mnN_(mmm)
            m1 = cyl_(m0 + 1, 13, 1)
            y0_ = y0
            y0__ = y0 + 1 if m1 < m0 else y0
        else:
            tmp = mmmN_(mmm)
            m0, m1 = tmp[0], cyl_(tmp[-1] + 1, 13, 1)
            y0_ = y0 if ismono_(tmp) else y0 - 1
            y0__ = y0_ + 1 if m1 <= m0 else y0
        return (m0, m1, y0_, y0__)

    ##data and mask
    if isinstance(c.data, np.ma.MaskedArray):
        if ~np.ma.is_masked(c.data):
                c.data.data[:] = 0.
        #if c.data.mask.ndim == 0:
        #    if ~c.data.mask:
        #        c.data.data[:] = 0.
        else:
            c.data.data[~c.data.mask] = 0.
    else:
        c.data = np.zeros(c.shape)
    ##coord('time')
    c.coord('time').units = cf_units.Unit('days since 1850-1-1',
                                           calendar='gregorian')
    m0, m1, y0_, y0__ = _mm01()
    y0_h = [datetime(i, m0, 1) for i in range(y0_, y0_ + ny)]
    y1_h = [datetime(i, m1, 1) for i in range(y0__, y0__ + ny)]
    tbnds = np.empty((ny, 2))
    tbnds[:, 0] = cf_units.date2num(y0_h,
                                    c.coord('time').units.origin,
                                    c.coord('time').units.calendar)
    tbnds[:, 1] = cf_units.date2num(y1_h,
                                    c.coord('time').units.origin,
                                    c.coord('time').units.calendar)
    tdata = np.mean(tbnds, axis=-1)
    c.coord('time').points = tdata
    c.coord('time').bounds = tbnds
    ##var_name ...
    if name:
        c.rename(name)
    if units:
        c.units = units
    if var_name:
        c.var_name = var_name
    if long_name:
        c.long_name = long_name
    if attrU:
        c.attributes.update(attrU)
    return c


def pSTAT_cube(cube, method, *freq, valid_season=True, **method_opts):
    ef0 = "method {!r} unreconnigsed!"
    ef1 = "freq {!r} unreconigsed!"
    method = method.upper()
    if method not in ('MEAN', 'MAX', 'MIN', 'MEDIAN', 'SUM', 'PERCENTILE',
                      'PROPORTION', 'STD_DEV', 'RMS', 'VARIANCE', 'HMEAN',
                      'COUNT', 'PEAK'):
        raise Exception(ef0.format(method))

    s4 = ('djf', 'mam', 'jja', 'son')

    d = dict(year=('year',),
             season=('season', 'seasonyr'),
             month=('month', 'year'),
             day=('doy', 'year'),
             hour=('hour', 'year'))

    dd = dict(hour=(cat.add_hour, ('time',), dict(name='hour')),
              day=(cat.add_day_of_year, ('time',), dict(name='doy')),
              month=(cat.add_month, ('time',), dict(name='month')),
              year=(cat.add_year, ('time',), dict(name='year')),
              season=(cat.add_season, ('time',), dict(name='season',
                                                      seasons=s4)),
              seasonyr=(seasonyr_cube, (s4,), dict(name='seasonyr')))

    def _x(f0):
        if f0 in d.keys():
            return d[f0]
        elif isSeason_(f0):
            return (f0, 'seasonyr')
        elif isMonth_(f0):
            return d['month']
        else:
            raise Exception(ef1.format(f0))

    def _xx(x):
        if isinstance(x, str):
            if x in d.keys():
                return (dd.copy(), None)
            elif isSeason_(x):
                tmp = {x: (cat.add_season_membership, ('time', x),
                           dict(name=x)),
                       'seasonyr': (seasonyr_cube, (x,),
                                    dict(name='seasonyr'))}
                dd_ = dd.copy()
                dd_.update(tmp)
                return (dd_, x)
            elif isMonth_(x):
                return (dd.copy(), x.capitalize())
            else:
                raise Exception(ef1.format(x))
        else:
            if all((f_ in d.keys() for f_ in x)):
                return (dd.copy(), None)
            else:
                f_ = [f_ for f_ in x if f_ not in d.keys()]
                #if len(f_) > 1 or not isSeason_(f_[0]) or 'season' in d.keys():
                #    raise("one or more specified 'freqs' unreconigsed; "
                #          "check input!")
                if len(f_) == 1 and isSeason_(f_[0]):
                    tmp = {f_[0]: (cat.add_season_membership, ('time', f_[0]),
                                   dict(name=f_[0])),
                           'seasonyr': (seasonyr_cube, (f_[0],),
                                        dict(name='seasonyr'))}
                    dd_ = dd.copy()
                    dd_.update(tmp)
                    return (dd_, f_[0])
                elif len(f_) == 1 and isMonth_(f_[0]):
                    return (dd.copy(), x.capitalize())
                else:
                    raise Exception(ef1.format('-'.join(x)))

    def _xxx(c, x):
        dd_, mmm = _xx(x)
        if isinstance(x, str):
            dff = _x(x)
        else:
            dff = uniqL_(flt_l([_x(i) for i in x]))
            if 'seasonyr' in dff:
                dff.remove('year')
        #dff = (dff, ) if isinstance(dff, str) else dff
        for i in dff:
            _f, fA, fK = dd_[i]
            try:
                _f(c, *fA, **fK)
            except ValueError:
                c.remove_coord(fK['name'])
                _f(c, *fA, **fK)
        if mmm:
            if isSeason_(mmm):
                c.coord(mmm).points = c.coord(mmm).points.astype(np.int)
                cstr = iris.Constraint(**{mmm: True})
            elif isMonth_(mmm):
                cstr = iris.Constraint(**{'month': mmm})
            c = c.extract(cstr)
            tmp = c.aggregated_by(dff, eval('iris.analysis.' + method),
                                   **method_opts)
            if isSeason_(mmm) and not ismono_(mmmN_(mmm)):
                tmp = extract_byAxes_(tmp, 'time', np.s_[1:-1])
        else:
            tmp = c.aggregated_by(dff, eval('iris.analysis.' + method),
                                  **method_opts)
            if x == 'season' and valid_season:
                tmp = extract_byAxes_(tmp, 'time', np.s_[1:-1])
        rm_t_aux_cube(tmp, keep=dff)
        return tmp

    freqs = ('year',) if len(freq) == 0 else freq
    o = ()
    for ff in [i.split('-') if '-' in i else i for i in freqs]:
        tmp = _xxx(cube.copy(), ff)
        o += (tmp,)
    return o[0] if len(o) == 1 else o

    #freqs = uniqL_(flt_l([i.split('-') if '-' in i else i for i in freq]))
    #if (len(freqs) == 0 or
    #    not any([x in ['hour', 'day', 'month', 'season', 'year']
    #             for x in freqs])):
    #    freq = ('year',)
    #if any(i in freqs for i in ('hour', 'day', 'month', 'year')):
    #    try:
    #        cat.add_year(cube, 'time', name='year')
    #    except ValueError:
    #        pass
    #if 'hour' in freqs:
    #    try:
    #        cat.add_hour(cube, 'time', name='hour')
    #    except ValueError:
    #        pass
    #if 'day' in freqs:
    #    try:
    #        cat.add_day_of_year(cube, 'time', name='doy')
    #    except ValueError:
    #        pass
    #if 'month' in freqs:
    #    try:
    #        cat.add_month(cube, 'time', name='month')
    #    except ValueError:
    #        pass
    #if 'season' in freqs:
    #    try:
    #        cat.add_season(cube, 'time', name='season',
    #                       seasons=('djf', 'mam', 'jja', 'son'))
    #    except ValueError:
    #        pass
    #    try:
    #        cat.add_season_year(cube, 'time', name='seasonyr',
    #                            seasons=('djf', 'mam', 'jja', 'son'))
    #    except ValueError:
    #        pass
    #o = ()
    #for ff in [i.split('-') if '-' in i else i for i in freq]:
    #    if isinstance(ff, str):
    #        dff = d[ff]
    #    else:
    #        dff = uniqL_(flt_l([d[i] for i in ff]))
    #        if 'seasonyr' in dff:
    #            dff.remove('year')
    #    tmp = cube.aggregated_by(dff, eval('iris.analysis.' + method),
    #                             **method_opts)
    #    if ff == 'season' and valid_season:
    #        tmp = extract_byAxes_(tmp, 'time', np.s_[1:-1])
    #    rm_t_aux_cube(tmp, keep=dff)
    #    o += (tmp,)
    #return o[0] if len(o) == 1 else o


def repair_cs_(cube):
    def _repair_cs_cube(c):
        cs = c.coord_system('CoordSystem')
        if cs is not None:
            for k in cs.__dict__.keys():
                if eval('cs.' + k) is None:
                    exec('cs.' + k + ' = ""')
        for coord in c.coords():
            if coord.coord_system is not None:
                coord.coord_system = cs
    if isinstance(cube, iris.cube.Cube):
        _repair_cs_cube(cube)
    elif isMyIter_(cube):
        for i in cube:
            if isinstance(i, iris.cube.Cube):
                _repair_cs_cube(i)


def _repair_lccs_cube(c, out=False):
    cs = c.coord_system('CoordSystem')
    o_ = 0
    if isinstance(cs, iris.coord_systems.LambertConformal):
        if (cs.false_easting is None or
            isinstance(cs.false_easting, np.ndarray)):
            cs.false_easting = ''
            o_ += 1
        if (cs.false_northing is None or
            isinstance(cs.false_northing, np.ndarray)):
            cs.false_northing = ''
            o_ += 1
        for coord in c.coords():
            if coord.coord_system is not None:
                coord.coord_system = cs
                coord.convert_units('m')
    if out:
        return o_


def repair_lccs_(cube):
    if isinstance(cube, iris.cube.Cube):
        _repair_lccs_cube(cube)
    elif isMyIter_(cube):
        for i in cube:
            _repair_lccs_cube(i)


def lccs_m2km_(cube):
    if not isMyIter_(cube):
        if isinstance(cube, iris.cube.Cube):
            cs = cube.coord_system('CoordSystem')
            if isinstance(cs, iris.coord_systems.LambertConformal):
                for coord in cube.coords():
                    if coord.coord_system is not None:
                        coord.convert_units('km')
    else:
        for i in cube:
            lccs_m2km_(i)


def cubesv_(cube, filename, netcdf_format='NETCDF4', local_keys=None,
            zlib=True, complevel=4, shuffle=True, fletcher32=False,
            contiguous=False, chunksizes=None, endian='native',
            least_significant_digit=None, packing=None, fill_value=None):
    if isinstance(cube, iris.cube.Cube):
        repair_lccs_(cube)
        dms = [i.name() for i in cube.dim_coords]
        udm = ('time',) if 'time' in dms else None
        iris.save(cube, filename, netcdf_format=netcdf_format,
                  local_keys=local_keys, zlib=zlib, complevel=complevel,
                  shuffle=shuffle, fletcher32=fletcher32,
                  contiguous=contiguous, chunksizes=chunksizes, endian=endian,
                  least_significant_digit=least_significant_digit,
                  packing=packing, fill_value=fill_value,
                  unlimited_dimensions=udm)
    elif isMyIter_(cube):
        for i, ii in enumerate(cube):
            cubesv_(ii, filename.replace(ext_(filename),
                                         '_{}{}'.format(i, ext_(filename))))


def half_grid_(x, side='i', axis=-1, loa=None, rb=360):
    dx = np.diff(x, axis=axis)
    if loa == 'lo':
        lb = rb - 360
        dx = cyl_(dx, 180, -180)
    tmp = extract_byAxes_(x, axis, np.s_[:-1]) + dx * .5
    if side in (0, 'i', 'inner'):
        o = tmp
    elif side in (-1, 'l', 'left'):
        o = np.concatenate((extract_byAxes_(x, axis, np.s_[:1]) -
                            extract_byAxes_(dx, axis, np.s_[:1]) * .5,
                            tmp),
                           axis=axis)
    elif side in (1, 'r', 'right'):
        o = np.concatenate((tmp,
                            extract_byAxes_(x, axis, np.s_[-1:]) +
                            extract_byAxes_(dx, axis, np.s_[-1:]) * .5),
                           axis=axis)
    elif side in (2, 'b', 'both'):
        o = np.concatenate((extract_byAxes_(x, axis, np.s_[:1]) -
                            extract_byAxes_(dx, axis, np.s_[:1]) * .5,
                            tmp,
                            extract_byAxes_(x, axis, np.s_[-1:]) +
                            extract_byAxes_(dx, axis, np.s_[-1:]) * .5),
                           axis=axis)
    else:
        raise ValueError("unknow value of side!")
    if loa == 'lo':
        o = cyl_(o, rb, lb)
    if loa == 'la':
        o = np.where(o > 90, 90, o)
        o = np.where(o < -90, -90, o)
    return o


def _ri1d(c1d, v):
    from skextremes.models.classic import GEV
    data = c1d.data
    data = data.compressed() if np.ma.isMaskedArray(data) else data
    if data.size:
        _gev = GEV(data)
        return _gev.return_periods(v)
    else:
        return np.nan


def ri_cube(cube, v, nmin=10):
    c = extract_byAxes_(cube, 'time', 0)
    rm_sc_cube(c)
    pst_(c, 'recurrence interval', units='year')
    ax = axT_cube(cube)
    if ax is None or cube.shape[ax] < nmin:
        emsg = "too few data for estimation!"
        raise Exception(emsg)
    ax_fn_mp_(cube, ax, _ri1d, c, v)
    return c


def nine_points_cube(cube, longitude, latitude):
    x, y = get_loa_pts_2d_(cube)
    d_ = ind_shape_i_(x.shape,
                      np.argmin(haversine_(longitude, latitude, x, y)),
                      axis=None)
    xyd = get_xyd_cube(cube)
    ind_ = np.arange(-1, 2, dtype=np.int)
    ind = list(np.s_[:,] * cube.ndim)
    wmsg = ("Causious that center point may be given outside (or at the "
            "boundary of) the geo domain of the input CUBE!")
    for i, ii in zip(xyd, d_):
        if ii == cube.shape[i] - 1:
            warnings.warn(wmsg)
            ii -= 1
        elif ii == 0:
            warnings.warn(wmsg)
            ii += 1
        ind[i] = ind_ + ii
    return cube[tuple(ind)]


def replace_coord_(cube, new_coord):
    """
    Replace the coordinate whose metadata matches the given coordinate.

    """
    old_coord = cube.coord(new_coord.name())
    dims = cube.coord_dims(old_coord)
    was_dimensioned = old_coord in cube.dim_coords
    cube._remove_coord(old_coord)
    if was_dimensioned and isinstance(new_coord, iris.coords.DimCoord):
        cube.add_dim_coord(new_coord, dims[0])
    else:
        cube.add_aux_coord(new_coord, dims)

    for factory in cube.aux_factories:
        factory.update(old_coord, new_coord)

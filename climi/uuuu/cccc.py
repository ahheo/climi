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
* en_mean_              : ensemble mean
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
* get_lon_lat_dimcoords_: modified _get_lon_lat_coords from iris
* get_x_y_dimcoords_    : horizontal spatial dim coords
* get_xyd_cube          : cube axes of xy dims
* guessBnds_cube        : bounds of dims points
* initAnnualCube_       : initiate annual cube
* inpolygons_cube       : points if inside polygons
* intersection_         : cube intersection with lon/lat range
* iris_regrid_          : using iris.cube.Cube.regrid but keeping aux_coords
* isGI_                 : if Iterator
* isIter__              : if Iterable but not str or bytes
* isMyIter_             : if Iterable but not cube ndarray str or bytes
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
* pSTAT_cube            : period statistic (month, season, year)
* pst_                  : post-rename/reunits cube(L) 
* pp_cube               : pth and 100-pth of cube(L) data (each)
* purefy_cubeL_         : prepare for concat or merge
* remap_ll_             : remap cube
* repair_cs_            : bug fix for save cube to nc
* repair_lccs_          : bug fix for save cube to nc (LamgfortComfort)
* rgMean_cube           : regional mean
* rgMean_poly_cube      : regional mean over in polygon only
* rm_sc_cube            : remove scalar coords
* rm_t_aux_cube         : remove time-related aux_coords
* rm_yr_doy_cube        : opposite action of yr_doy_cube
* seasonyr_cube         : season_year auxcoord
* slice_back_           : slice back to parent (1D)
* y0y1_of_cube          : starting and ending year of cube
* yr_doy_cube           : year and day-of-year auxcoord
...

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 06.09.2019
Date last modified: 24.09.2020
           comment: add func nTslice_cube for slicing cube along a no-Time axis
"""

import numpy as np
import iris
import iris.coord_categorisation as cat
from iris.experimental.equalise_cubes import equalise_attributes

import cf_units
import warnings
from datetime import datetime
from typing import Iterable, Iterator

from .ffff import *


__all__ = ['alng_axis_',
           'area_weights_',
           'ax_fn_mp_',
           'axT_cube',
           'concat_cube_',
           'cubesv_',
           'cut_as_cube',
           'en_iqr_',
           'en_mean_',
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
           'get_lon_lat_dimcoords_',
           'get_x_y_dimcoords_',
           'get_xyd_cube',
           'guessBnds_cube',
           'initAnnualCube_',
           'inpolygons_cube',
           'intersection_',
           'iris_regrid_',
           'isGI_',
           'isIter_',
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
           'pSTAT_cube',
           'pst_',
           'pp_cube',
           'purefy_cubeL_',
           'remap_ll_',
           'repair_cs_',
           'repair_lccs_',
           'rgMean_cube',
           'rgMean_poly_cube',
           'rm_sc_cube',
           'rm_t_aux_cube',
           'rm_yr_doy_cube',
           'seasonyr_cube',
           'slice_back_',
           'y0y1_of_cube',
           'yr_doy_cube']


def slice_back_(cnd, c1d, ii, axis):
    """
    ... put 1D slice back to its parent CUBE/ARRAY ...

    Parsed arguments:
         cnd: parent CUBE/ARRAY that has multiple dimensions
         c1d: 1D CUBE/ARRAY slice
          ii: slice # of c1d in iteration
        axis: axis of cnd to place c1d
    Returns:
        revised cnd
    """

    if isinstance(c1d, iris.cube.Cube):
        c1d = c1d.data
    if not isinstance(c1d, np.ndarray):
        c1d = np.asarray(c1d)
    if axis is None:
        if c1d.size != 1:
            raise Exception('slice NOT matched its parent '
                            'along axis {}'.format(axis))
    else:
        axis = cyl_(axis, len(cnd.shape))
        if isinstance(axis, Iterable) and not np.all(np.diff(axis) == 1):
            raise Exception("unknown axis!")
        if not np.all(np.asarray(cnd.shape)[axis] == np.asarray(c1d.shape)):
            raise Exception('slice NOT matched its parent '
                            'along axis {}'.format(axis))

    if isinstance(cnd, iris.cube.Cube):
        cnd.data[ind_shape_i_(cnd.shape, ii, axis)] =\
            np.ma.masked_array(c1d, np.isnan(c1d))
    elif isinstance(cnd, np.ma.masked_array):
        cnd[ind_shape_i_(cnd.shape, ii, axis)] =\
            np.ma.masked_array(c1d, np.isnan(c1d))
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
        raise Exception('arguments not interpretable!')

    if len(vArg) > 0:
        ax, sl = list(vArg[::2]), list(vArg[1::2])
        ax.insert(0, axis)
        sl.insert(0, sl_i)
    else:
        ax = [axis]
        sl = [sl_i]

    if isinstance(cnd, iris.cube.Cube):
        ax = [cnd.coord_dims(ii)[0] if isinstance(ii, str) else ii
              for ii in ax]

    nArg = [(ii, jj) for ii, jj in zip(ax, sl)]
    nArg = tuple(jj for ii in nArg for jj in ii)
    inds = inds_ss_(cnd.ndim, *nArg)

    return cnd[inds]


def isGI_(x):
    return isinstance(x, Iterator)


def isIter_(x, xi=None, XI=(str, bytes)):
    o = isinstance(x, Iterable) and not isinstance(x, XI)
    if xi and o:
        if not isGI_(x):
            o = o and all([isinstance(i, xi) or i is None for i in x])
        else:
            warnings.warn("xi ignored for Iterator or Generator!")
    return o


def isMyIter_(x):
    return isIter_(x,
                   xi=(np.ndarray, iris.cube.Cube),
                   XI=(np.ndarray, iris.cube.Cube, str, bytes))


def pst_(cube, name=None, units=None, var_name=None):                            
    if isinstance(cube, iris.cube.Cube):                                                        
        if name:                                                                
            cube.rename(name)                                                    
        if units:                                                               
            cube.units = units                                                   
        if var_name:                                                            
            cube.var_name = var_name                                             
    elif isMyIter_(cube):                                                                       
        for i in cube:                                                           
            pst_(i, name=name, units=units, var_name=var_name)


def axT_cube(cube):
    tmp = cube.coord_dims('time')
    return None if len(tmp) == 0 else tmp[0]


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


def y0y1_of_cube(cube):
    if isinstance(cube, iris.cube.Cube):
        ccs = [i.name() for i in cube.coords()]
        if 'year' in ccs:
            return list(cube.coord('year').points[[0, -1]])
        elif 'seasonyr' in ccs:
            return list(cube.coord('seasonyr').points[[0, -1]])
        else:
            cat.add_year(cube, 'time', name='year')
            return list(cube.coord('year').points[[0, -1]])
    elif isMyIter_(cube):
        yy = np.array([y0y1_of_cube(i) for i in cube])
        return [np.max(yy[:, 0]), np.min(yy[:, 1])]
    else:
        raise TypeError("unknown type!")


def extract_period_cube(cube, y0, y1, yy=False):
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
    ccs = [i.name() for i in cube.coords()]
    if 'year' in ccs:
        cstr = iris.Constraint(year=lambda cell: y0 <= cell <= y1)
    elif 'seasonyr' in ccs:
        cstr = iris.Constraint(seasonyr=lambda cell: y0 <= cell <= y1)
    else:
        cat.add_year(cube, 'time', name='year')
        cstr = iris.Constraint(year=lambda cell: y0 <= cell <= y1)
    c_y = cube.extract(cstr)
    #c_y.remove_coord('year')
    return None if yy and y0y1_of_cube(c_y) != [y0, y1] else c_y


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
    try:
        cat.add_day_of_year(cube, 'time', name='doy')
    except ValueError:
        pass
    else:
        cube.coord('doy').attributes = {}
    x1, x2 = cyl_(d - r, 365), cyl_(d + r, 365)
    c_30 = cube.extract(iris.Constraint(doy = lambda cell:
                                        x1 <= cell <= x2 if x1 < x2
                                        else not (x2 < cell < x1)))
    #c_30.remove_coord('doy')
    return c_30


def _name_not_in(nm, nmL):
    while nm in nmL:
        nm += '_'
    return nm


def extract_season_cube(cube, mmm):
    """
    ... extract a cube of season named with continuous-months' 1st letters ...

    Parsed arguments:
         cube: cube/cubelist containing at least coord('time')
          mmm: continuous-months' 1st letters (for example, 'djf')
    Returns:
        ncube: cube/cubelist after extraction
    """
    if isinstance(cube, iris.cube.Cube):
        ss_auxs = [i.name() for i in cube.aux_coords if 'season' in i.name()]
        if len(ss_auxs) == 0:
            cat.add_season_membership(cube, 'time', season=mmm,
                                      name='season_ms')
            ncube = cube.extract(iris.Constraint(season_ms=True))
            ncube.remove_coord('season_ms')
        else:
            auxn = None
            for i in ss_auxs:
                if (isinstance(cube.coord(i).points[0], str) and
                    mmm in cube.coord(i).points):
                    auxn = i
                    break
            if auxn is not None:
                ncube = cube.extract(iris.Constraint(**{auxn: mmm}))
            else:
                auxn = _name_not_in('season_ms', ss_auxs)
                cat.add_season_membership(cube, 'time', season=mmm, name=auxn)
                ncube = cube.extract(iris.Constraint(**{auxn: True}))
                ncube.remove_coord(auxn)
        return ncube
    elif isMyIter_(cube):
        cl = [extract_season_cube(i, mmm) for i in cubeL]
        return iris.cube.CubeList(cl)
    else:
        raise TypeError("unknown type!")


def extract_month_cube(cube, Mmm):
    try:
        cat.add_month(cube, 'time', name='month')
    except ValueError:
        pass
    ncube = cube.extract(iris.Constraint(month=Mmm))
    ncube.remove_coord('month')
    return ncube


def f_allD_cube(cube, rg=None, f='MAX', **f_opts):
    warnings.filterwarnings("ignore", category=UserWarning)
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
    return np.asarray([f_allD_cube(cube, rg=rg, f='PERCENTILE',
                                                percent=p),
                       f_allD_cube(cube, rg=rg, f='PERCENTILE',
                                                percent=100 - p)])


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


def get_xyd_cube(cube):
    xc, yc = get_x_y_dimcoords_(cube)
    xyd = list(cube.coord_dims(xc) + cube.coord_dims(yc))
    xyd.sort()
    return xyd


def _get_xy_lim(cube, lol=None, lal=None):
    xc, yc = get_x_y_dimcoords_(cube)
    lo, la = cube.coord('longitude'), cube.coord('latitude')
    if lol is None:
        lol = [lo.points.min(), lo.points.max()]
    if lal is None:
        lal = [la.points.min(), la.points.max()]
    if xc == lo and yc == la:
        xyl = {xc.name(): lol, yc.name(): lal}
    else:
        xd = cube.coord_dims(lo).index(
                 np.intersect1d(cube.coord_dims(lo), cube.coord_dims(xc)))
        yd = cube.coord_dims(lo).index(
                 np.intersect1d(cube.coord_dims(lo), cube.coord_dims(yc)))
        if np.all(lo.points >= 0) and np.any(np.asarray(lol) < 0):
            a_ = ind_inRange_(cyl_(lo.points, 180, -180), *lol)
        elif np.all(lo.points <= 180) and np.any(np.asarray(lol) > 180):
            a_ = ind_inRange_(cyl_(lo.points, 360, 0), *lol)
        else:
            a_ = ind_inRange_(lo.points, *lol)
        b_ = ind_inRange_(la.points, *lal)
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


def _get_ind_lolalim(cube, lol=None, lal=None):
    loc, lac = cube.coord('longitude'), cube.coord('latitude')
    xyd = set(cube.coord_dims(loc) + cube.coord_dims(lac))
    lo, la = loc.points, lac.points
    if lol is None:
        lol = [lo.min(), lo.max()]
    if lal is None:
        lal = [la.min(), la.max()]
    if lo.ndim == 1:
        lo, la = np.meshgrid(lo, la)
    if np.all(lo >= 0) and np.any(np.asarray(lol) < 0):
        a_ = ind_inRange_(cyl_(lo, 180, -180), *lol)
    elif np.all(lo <= 180) and np.any(np.asarray(lol) > 180):
        a_ = ind_inRange_(cyl_(lo, 360, 0), *lol)
    else:
        a_ = ind_inRange_(lo, *lol)
    b_ = ind_inRange_(la, *lal)
    c_ = np.logical_and(a_, b_)
    return robust_bc2_(c_, cube.shape, xyd)


def intersection_(cube, **kwArgs):
    """
    ... intersection by range of longitude/latitude ...
    """
    if cube.coord('latitude').ndim == 1:
        return cube.intersection(**kwArgs)
    else:
        if 'longitude' not in kwArgs or 'latitude' not in kwArgs:
            raise Exception("both 'longitude' and 'latitude'"
                            "reqired when projection != lat/lon")
        lol, lal = kwArgs['longitude'], kwArgs['latitude']
        xyl = _get_xy_lim(cube, lol, lal)
        if isinstance(xyl, dict):
            return cube.intersection(**xyl)
        else:
            return  extract_byAxes_(cube, *xyl)
        #else:
        #    longitude = kwArgs['longitude']
        #    latitude = kwArgs['latitude']
        #    lo = ind_inRange_(cube.coord('longitude').points.data,
        #                      longitude[0], longitude[-1])
        #    la = ind_inRange_(cube.coord('latitude').points.data,
        #                      latitude[0], latitude[-1])
        #ii = np.logical_and(lo, la)
        #minii = np.min(np.argwhere(ii), axis=0)
        #maxii = np.max(np.argwhere(ii), axis=0)
        #cube_i = extract_byAxes_(cube, -2, np.s_[minii[0]:maxii[0]+1],
        #                         -1, np.s_[minii[-1]:maxii[-1]+1])
        #return cube_i


def seasonyr_cube(cube, mmm):
    """
    ... add season_year auxcoords to a cube especially regarding
        specified season ...
    """
    if isinstance(mmm, str):
        seasons = (mmm, rest_mns_(mmm))
    elif isinstance(mmm, (list, tuple)) and len(''.join(mmm)) == 12:
        seasons = mmm
    else:
        raise Exception("unknown season '{}'!".format(mmm))
    try:
        cat.add_season_year(cube, 'time', name='seasonyr', seasons=seasons)
    except ValueError:
        cube.remove_coord('seasonyr')
        cat.add_season_year(cube, 'time', name='seasonyr', seasons=seasons)


def yr_doy_cube(cube):
    """
    ... add year, day-of-year auxcoords to a cube ...
    """
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
    if isinstance(cube, iris.cube.Cube):
        for i in cube.aux_coords:
            if keep is None:
                isTaux = any([ii in i.name() for ii in tauxL])
            elif isIter_(keep):
                isTaux = any([ii in i.name() for ii in tauxL])\
                         and i.name() not in keep
            else:
                isTaux = any([ii in i.name() for ii in tauxL])\
                         and i.name() != keep
            if isTaux:
                cube.remove_coord(i)
    elif isMyIter_(cube):
        for c in cube:
            rm_t_aux_cube(c)
    else:
        raise TypeError('Input should be Cube or iterable Cubes!')


def rm_sc_cube(cube):
    if isinstance(cube, iris.cube.Cube):
        for i in cube.coords():
            if len(cube.coord_dims(i)) == 0:
                cube.remove_coord(i)
    elif isMyIter_(cube):
        for c in cube:
            rm_sc_cube(c)
    else:
        raise TypeError('Input should be Cube or Iterable Cubes!')


def guessBnds_cube(cube):
    """
    ... guess bounds of dims of cube if not exist ...
    """
    for i in cube.dim_coords:
        try:
            i.guess_bounds()
        except ValueError:
            pass


def get_lon_lat_dimcoords_(cube):
    """
    ... get lon and lat coords (dimcoords only) ...
    """
    lat_coords = [coord for coord in cube.dim_coords
                  if "latitude" in coord.name()]
    lon_coords = [coord for coord in cube.dim_coords
                  if "longitude" in coord.name()]
    if len(lat_coords) > 1 or len(lon_coords) > 1:
        raise ValueError(
            "Calling `_get_lon_lat_coords` with multiple lat or lon coords"
            " is currently disallowed")
    lat_coord = lat_coords[0]
    lon_coord = lon_coords[0]
    return (lon_coord, lat_coord)


def get_x_y_dimcoords_(cube):
    """
    ... get x and y coords (dimcoords only) ...
    """
    ycn = ['latitude', 'y_coord', 'y-coord', 'y coord', 'second']
    xcn = ['longitude', 'x_coord', 'x-coord', 'x coord', 'first']
    y_coords = [coord for coord in cube.dim_coords
                if any([i in coord.name() for i in ycn])]
    x_coords = [coord for coord in cube.dim_coords
                if any([i in coord.name() for i in xcn])]
    if len(y_coords) > 1 or len(x_coords) > 1:
        raise ValueError(
            "Calling `get_x_y_dimcoords_` with multiple x or y coords"
            " is currently disallowed")
    y_coord = y_coords[0]
    x_coord = x_coords[0]
    return (x_coord, y_coord)


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
        lon, lat = get_lon_lat_dimcoords_(cube)
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
    xc, yc = get_x_y_dimcoords_(cube1)
    xn = xc.name()
    yn = yc.name()
    xe = np.min(np.abs(np.diff(cube1.coord(xn).points)))/2
    ye = np.min(np.abs(np.diff(cube1.coord(yn).points)))/2
    x0 = np.min(cube0.coord(xn).points)
    x1 = np.max(cube0.coord(xn).points)
    y0 = np.min(cube0.coord(yn).points)
    y1 = np.max(cube0.coord(yn).points)
    cube_ = extract_byAxes_(cube1,
                            xn, ind_inRange_(cube1.coord(xn).points,
                                             x0 - xe, x1 + xe, False),
                            yn, ind_inRange_(cube1.coord(yn).points,
                                             y0 - ye, y1 + ye, False))
    return cube_


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
    if (sftlf is not None) and (ga is not None):
        sf_ = iris.util.squeeze(sftlf)
        if sf_.ndim != 2:
            raise Exception('NOT 2D area-cube!')
        sf = cut_as_cube(cubeD, sf_).data
        sf = robust_bc2_(sf, cubeD.shape, get_xyd_cube(cubeD))
        return ga * sf / 100.
    else:
        return ga


def rgMean_cube(cubeD, sftlf=None, areacella=None, rgD=None):
    """
    ... regional mean; try weighted if available ...
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    if rgD:
        lol = rgD['longitude'] if 'longitude' in rgD else None
        lal = rgD['latitude'] if 'latitude' in rgD else None
        ind = _get_ind_lolalim(cubeD, lol, lal)
        if ga is None:
            ga = ind * np.ones(ind.shape)
        else:
            ga *= ind
    xc, yc = get_x_y_dimcoords_(cubeD)
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
    if not isinstance(poly, (list, tuple, set)):
        ind = poly.contains_points(points, **kwArgs)
    elif len(poly) < 2:
        ind = poly[0].contains_points(points, **kwArgs)
    else:
        inds = [i.contains_points(points, **kwArgs) for i in poly]
        ind = np.logical_or.reduce(inds)
    return ind


def inpolygons_cube(poly, cube, **kwArgs):
    try:
        lo_ = cube.coord('longitude')
        la_ = cube.coord('latitude')
    except:
        raise Exception("input cube(s) must have "
                        "longitude/latidute coords!")
    dims = set(cube.coord_dims(lo_) + cube.coord_dims(la_))
    if lo_.ndim != 2:
        x, y = np.meshgrid(lo_.points, la_.points)
    else:
        x, y = lo_.points, la_.points
    ind = _inpolygons(poly, np.vstack((x.ravel(), y.ravel())).T, **kwArgs)
    ind = robust_bc2_(ind.reshape(x.shape), cube.shape, dims)
    return ind


def maskNaN_cube(cube):
    ind = np.isnan(cube.data)
    cube = iris.util.mask_cube(cube, ind)


def maskPOLY_cube(poly, cube, out=True, **kwArgs):
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    ind = ~ind if out else ind
    cube = iris.util.mask_cube(cube, ind)


def rgMean_poly_cube(cubeD, poly, sftlf=None, areacella=None, **kwArgs):
    warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    xc, yc = get_x_y_dimcoords_(cubeD)
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


def en_mean_(eCube, **kwArgs):
    """
    ... ensemble mean of a cube (along dimcoord 'realization') ...
    """
    return eCube.collapsed('realization', iris.analysis.MEAN, **kwArgs)


def en_iqr_(eCube):
    """
    ... ensemble interquartile range (IQR) of a cube (along dimcoord
        'realization') ...
    """
    a = eCube.collapsed('realization', iris.analysis.PERCENTILE, percent=75)
    b = eCube.collapsed('realization', iris.analysis.PERCENTILE, percent=25)
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
    tmpD = {}
    cl = []
    for i, c in enumerate(cubeL):
        c.attributes = {}
        if cref is None:
            cref = tmpD.setdefault('ref', c.copy())
        else:
            cref.attributes = {}
        if opt == 0:
            try:
                tmp = iris_regrid_(cref, c, rmpm)
            except:
                tmp = remap_ll_(cref, c)
        elif opt == 1:
            tmp = iris_regrid_(cref, c, rmpm)
        elif opt == 2:
            tmp = remap_ll_(cref, c)
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
    i, arr, o0, args, kwargs = ak_
    try:
        if o0.data.mask or o0.mask:
            return None
    except AttributeError:
        pass
    return (i, func(*arr, *args, **kwargs))


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
        raise Exception('type of out SHOULD be Cube')
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
        return (i, func(*aaa, *args, **kwargs))

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
        raise Exception('type of out SHOULD be NDARRAY or CUBE')
    if not isinstance(arr, (tuple, list)):
        arr = (arr,)

    P = mp.Pool(nproc)
    def _i(i, sl_=np.s_[:,]):
        return ind_shape_i_(arr[0].shape, i, ax, sl_)
    X = P.starmap_async(_func, [(func, (i, tuple([ii[_i(i)] for ii in arr]),
                                o0[_i(i, sl_=np.s_[0,])], args, kwargs))
                                for i in range(nSlice_(arr[0].shape, ax))])
    XX = X.get()
    P.close()

    _sb(XX, out, ax)


def _sb(out, c, ax):
    for sl in out:
        if sl is not None:
            i, o = sl[0], sl[1]
            if o is not None:
                _isb(i, o, c, ax)


def _isb(i, o, c, ax):
    def _get_nax(x):
        if x.ndim == 0:
            nax = None
        else:
            if isinstance(ax, int):
                nax = np.arange(ax, ax + x.ndim)
            else:
                nax = np.arange(ax[0], ax[0] + x.ndim)
        return nax
    if isinstance(c, (list, tuple, iris.cube.CubeList)):
        for j, k in zip(c, o):
            if not isinstance(k, np.ndarray):
                k = np.asarray(k)
            slice_back_(j, k, i, _get_nax(k))
    else:
        if not isinstance(o, np.ndarray):
            o = np.asarray(o)
        slice_back_(c, o, i, _get_nax(o))


def alng_axis_(arrs, ax, func, out, *args, **kwargs):
    if isMyIter_(out):
        o0 = out[0]
    else:
        o0 = out
    if not isinstance(o0, (np.ndarray, iris.cube.Cube)):
        raise Exception('type of out SHOULD be NDARRAY of CUBE')
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
                    long_name=None, attrU=None):
    y0, y1 = y0y1
    c = extract_byAxes_(c0, 'time', np.s_[:(y1 - y0 + 1)])
    rm_t_aux_cube(c)
    ##data and mask
    if isinstance(c.data, np.ma.core.MaskedArray):
        if c.data.mask.ndim == 0:
            if ~c.data.mask:
                c.data.data[:] = 0.
        else:
            c.data.data[~c.data.mask] = 0.
    else:
        c.data = np.zeros(c.shape)
    ##coord('time')
    c.coord('time').units = cf_units.Unit('days since 1850-1-1',
                                           calendar='gregorian')
    y_ = [datetime(i, 7, 1) for i in np.arange(y0, y1 + 1, dtype='int')]
    tdata = cf_units.date2num(y_,
                              c.coord('time').units.origin,
                              c.coord('time').units.calendar)
    y0_h = [datetime(i, 1, 1) for i in np.arange(y0, y1 + 1, dtype='int')]
    y1_h = [datetime(i, 12, 31) for i in np.arange(y0, y1 + 1, dtype='int')]
    tbnds = np.empty(tdata.shape + (2,))
    tbnds[:, 0] = cf_units.date2num(y0_h,
                                    c.coord('time').units.origin,
                                    c.coord('time').units.calendar)
    tbnds[:, 1] = cf_units.date2num(y1_h,
                                    c.coord('time').units.origin,
                                    c.coord('time').units.calendar)
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


def pSTAT_cube(cube, method, *freq, **method_otps):
    method = method.upper()
    if method not in ['MEAN', 'MAX', 'MIN', 'MEDIAN', 'SUM', 'PERCENTILE',
                      'PROPORTION', 'STD_DEV', 'RMS', 'VARIANCE', 'HMEAN',
                      'COUNT', 'PEAK']:
        raise Exception('method {} unknown!'.format(method))
    if (len(freq) == 0 or
        not any([x in ['day', 'month', 'season', 'year'] for x in freq])):
        freq = ('year',)
    if 'year' in freq or 'month' in freq or 'day' in freq:
        try:
            cat.add_year(cube, 'time', name='year')
        except ValueError:
            pass
    if 'day' in freq:
        try:
            cat.add_day_of_year(cube, 'time', name='doy')
        except ValueError:
            pass
    if 'month' in freq:
        try:
            cat.add_month(cube, 'time', name='month')
        except ValueError:
            pass
    if 'season' in freq:
        try:
            cat.add_season(cube, 'time', name='season',
                           seasons=('djf', 'mam', 'jja', 'son'))
        except ValueError:
            pass
        try:
            cat.add_season_year(cube, 'time', name='seasonyr',
                                seasons=('djf', 'mam', 'jja', 'son'))
        except ValueError:
            pass
    d = dict(year='year', season=('season', 'seasonyr'),
             month=('month', 'year'), day=('doy', 'year'))
    o = ()
    for ff in freq:
        tmp = cube.aggregated_by(d[ff], eval('iris.analysis.' + method),
                                 **method_otps)
        rm_t_aux_cube(tmp, keep=d[ff])
        o += (tmp,)
    return o[0] if len(o) == 1 else o


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


def _repair_lccs_cube(c):
    cs = c.coord_system('CoordSystem')
    if isinstance(cs, iris.coord_systems.LambertConformal):
        if (cs.false_easting is None or
            isinstance(cs.false_easting, np.ndarray)):
            cs.false_easting = ''
        if (cs.false_northing is None or
            isinstance(cs.false_northing, np.ndarray)):
            cs.false_northing = ''
        for coord in c.coords():
            if coord.coord_system is not None:
                coord.coord_system = cs
                coord.convert_units('m')


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


def remap_ll_(cref, cdata, method='linear', fill_value=None, rescale=False):
    #from scipy.interpolate import griddata

    lo0, la0 = cref.coord('longitude'), cref.coord('latitude')
    lo1, la1 = cdata.coord('longitude'), cdata.coord('latitude')
    for i in (lo0, la0, lo1, la1):
        i.convert_units('degrees')
    if lo0.ndim == 1:
        x0, y0 = np.meshgrid(lo0.points, la0.points)
    else:
        x0, y0 = lo0.points, la0.points
    if lo1.ndim == 1:
        x1, y1 = np.meshgrid(lo1.points, la1.points)
    else:
        x1, y1 = lo1.points, la1.points
    if np.all(x0 >= 0) and np.any(x1 < 0):
        x1 = cyl_(x1, 360)
    if np.all(x0 <= 180) and np.any(x1 > 180):
        x1 = cyl_(x1, 180, -180)

    sh0, sh1 = np.asarray(cref.shape), np.asarray(cdata.shape)
    xydim0 = cref.coord_dims(lo0) + cref.coord_dims(la0)
    xydim1 = cdata.coord_dims(lo1) + cdata.coord_dims(la1)

    nsh = sh1.copy()
    dmap = {}
    for i, ii in zip(xydim0, xydim1):
         nsh[ii] = sh0[i]
         dmap.update({i: ii})
    nsh = tuple(nsh)
    data = np.empty(nsh)

    data1 = nanMask_(cdata.data)
    data0 = nanMask_(cref.data)
    ind__ = np.isnan(data0[ind_shape_i_(cref.shape, 0, xydim0)])
    
    #for i in range(nSlice_(cdata.shape, xydim1)):
    #    ind = ind_shape_i_(cdata.shape, i, xydim1)
    #    dd = data1[ind]
    #    ind_ = ~np.isnan(dd)
    #    tmp = griddata((x1[ind_], y1[ind_]), dd[ind_],
    #                   (x0.ravel(), y0.ravel()), method=method,
    #                   fill_value=fill_value, rescale=rescale)
    #    tmp = tmp.reshape(x0.shape)
    #    tmp[ind__] = np.nan
    #    slice_back_(data, tmp, i, xydim1)
    ax_fn_mp_(data1, xydim1, _regrid_slice, data, x1, y1, x0, y0, ind__,
              method, np.nan, rescale)

    nmsk = np.isnan(data)
    fill_value = fill_value if fill_value\
                 else (cdata.data.fill_value
                       if hasattr(cdata.data, 'fill_value') else 1e+20)
    data[nmsk] = fill_value
    data = np.ma.MaskedArray(data, nmsk)

    dimc_dim = []
    for c in cdata.dim_coords:
        dim = cdata.coord_dims(c)[0]
        if dim not in xydim1:
            dimc_dim.append((c, dim))
    xc0, yc0 = get_x_y_dimcoords_(cref)
    xc1, yc1 = get_x_y_dimcoords_(cdata)
    dimc_dim.append((xc0, cdata.coord_dims(xc1)[0]))
    dimc_dim.append((yc0, cdata.coord_dims(yc1)[0]))

    auxc_dim = []
    for c in cdata.aux_coords:
        dim = cdata.coord_dims(c)
        if not any([dim_ in xydim1 for dim_ in dim]):
            auxc_dim.append((c, dim))
    for c in cref.aux_coords:
        dim = cref.coord_dims(c)
        if dim and all([dim_ in xydim0 for dim_ in dim]):
            auxc_dim.append((c, tuple(dmap[dim_] for dim_ in dim)))

    return iris.cube.Cube(data, standard_name=cdata.standard_name,
                          long_name=cdata.long_name,
                          var_name=cdata.var_name, units=cdata.units,
                          attributes=cdata.attributes,
                          cell_methods=cdata.cell_methods,
                          dim_coords_and_dims=dimc_dim,
                          aux_coords_and_dims=auxc_dim,
                          aux_factories=None,
                          cell_measures_and_dims=None)


def _regrid_slice(dd, x1, y1, x0, y0, ind__, method, fill_value, rescale): 
    from scipy.interpolate import griddata
    ind_ = ~np.isnan(dd)
    tmp = griddata((x1[ind_], y1[ind_]), dd[ind_],
                   (x0.ravel(), y0.ravel()), method=method,
                   fill_value=fill_value, rescale=rescale)
    tmp = tmp.reshape(x0.shape)
    tmp[ind__] = np.nan
    return tmp


def _dmap(cref, cdata):
    x0, y0 = get_x_y_dimcoords_(cref)
    x1, y1 = get_x_y_dimcoords_(cdata)
    return {cref.coord_dims(x0)[0]: cdata.coord_dims(x1)[0],
            cref.coord_dims(y0)[0]: cdata.coord_dims(y1)[0]}


def iris_regrid_(cref, cdata, scheme=None):
    scheme = scheme if scheme else\
             iris.analysis.Linear(extrapolation_mode='mask')
    tmp = cdata.regrid(cref, scheme)
    dmap = _dmap(cref, cdata)
    for c in cref.aux_coords:
        dim = cref.coord_dims(c)
        if dim and all([dim_ in dmap.keys() for dim_ in dim]):
            tmp.add_aux_coord(c, tuple(dmap[dim_] for dim_ in dim))
    return tmp

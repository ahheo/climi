"""
>--#########################################################################--<
>-------------------------------------rgd-------------------------------------<
>--#########################################################################--<
...

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 11.11.2019
Date last modified: 11.11.2020
           comment: efficiency to be improved
"""


import iris
import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon
from scipy.sparse import csc_matrix, diags

from .ffff import *
from .cccc import ax_fn_mp_, extract_byAxes_, get_loa_, get_xy_dim_, half_grid_


__all__ = ['rgd_scipy_',
           'rgd_iris_',
           'rgd_li_opt0_',
           'rgd_poly_',
           'POLYrgd']


def _lo_rb(lo):
    return 180 if np.any(lo < 0) else 360


def _get_ll_pnts(lo, la, isyx):
    if lo.ndim == 1:
        if isyx:
            x, y = np.meshgrid(lo.points, la.points)
        else:
            y, x = np.meshgrid(la.points, lo.points)
    if lo.ndim == 2:
        if isyx:
            x, y = lo.points, la.points
        else:
            x, y = lo.points.T, la.points.T
    return (x, y)


def _isyx(lod, lad):
    return lad[0] < lod[0] if len(lod) == 1 else lod[0] < lod[1]


def _lr(x2d, isyx, rb=180):
    def _crcl(xx):
        if np.any(np.abs(np.diff(xx)) > 180):
            return cyl_(xx, xx.max() - 180, xx.max() - 540)
        else:
            return xx
    if isyx:
        l, r = _crcl(x2d[:, 0]).min(), _crcl(x2d[:, -1]).max()
    else:
        l, r = _crcl(x2d[0, :]).min(), _crcl(x2d[-1, 0]).max()
    return tuple(cyl_(i, rb, rb - 360) for i in (l, r))


def _lld(x2d):
    return max(np.max(np.abs(cyl_(np.diff(x2d, axis=1), 180, -180))),
               np.max(np.abs(cyl_(np.diff(x2d, axis=0), 180, -180))))


def _is_crcl(x2d, isyx):
    return np.diff(_lr(x2d, isyx)) < _lld(x2d) * 1.5


def rgd_scipy_(src_cube, target_cube,
              method='linear', fill_value=None, rescale=False):
    #from scipy.interpolate import griddata
    dmap = _dmap(src_cube, target_cube)
    loT, laT = get_loa_(target_cube)
    loS, laS = get_loa_(src_cube)
    if loT is None or loS is None:
        raise Exception("missing longitude/latitude coords.")
    isyxT = _isyx(target_cube.coord_dims(loT), target_cube.coord_dims(laT))
    isyxS = _isyx(target_cube.coord_dims(loS), target_cube.coord_dims(laS))
    #2d longitude/latitude points
    xT, yT = _get_ll_pnts(loT, laT, isyxT)
    xS, yS = _get_ll_pnts(loS, laS, isyxS)
    lT, rT = _lr(xT, isyxT)
    xT = cyl_(xT, lT + 360, lT)
    xS = cyl_(xS, lT + 360, lT)
    if _is_crcl(xS, isyxS):
        rb = lT + 180
        xS = (cyl_(xS, rb, rb -360), cyl_(xS, rb + 360, rb))

    shT, shS = np.asarray(target_cube.shape), np.asarray(src_cube.shape)
    xydimT = tuple(dmap.keys())
    xydimS = tuple(dmap.values())

    nsh = shS.copy()
    for i in dmap.keys():
         nsh[dmap[i]] = shT[i]
    nsh = tuple(nsh)
    data = np.empty(nsh)
    dataS = nanMask_(src_cube.data)
    if nSlice_(shS, xydimS) > 20:
        ax_fn_mp_(dataS, xydimS, _regrid_slice, data, xS, yS, xT, yT,
                  method, np.nan, rescale)
    else:
        for i in range(nSlice_(shS, xydimS)):
            ind = ind_shape_i_(shS, i, axis=xydimS)
            data[ind] = _regrid_slice(dataS[ind], xS, yS, xT, yT,
                                      method, np.nan, rescale)
    #masking
    nmsk = np.isnan(data)
    fill_value = fill_value if fill_value\
                 else (src_cube.data.fill_value
                       if hasattr(src_cube.data, 'fill_value') else 1e+20)
    data[nmsk] = fill_value
    data = np.ma.MaskedArray(data, nmsk)
    #dims for new cube
    dimc_dim = _get_dimc_dim(src_cube, target_cube)
    auxc_dim = _get_auxc_dim(src_cube, target_cube, dmap)

    return iris.cube.Cube(data, standard_name=src_cube.standard_name,
                          long_name=src_cube.long_name,
                          var_name=src_cube.var_name, units=src_cube.units,
                          attributes=src_cube.attributes,
                          cell_methods=src_cube.cell_methods,
                          dim_coords_and_dims=dimc_dim,
                          aux_coords_and_dims=auxc_dim,
                          aux_factories=None,
                          cell_measures_and_dims=None)


def _get_dimc_dim(src_cube, target_cube):
    dimc_dim = []
    xcT, ycT = get_xy_dim_(target_cube)
    xcS, ycS = get_xy_dim_(src_cube)
    xydimS = src_cube.coord_dims(xcS) + src_cube.coord_dims(ycS)
    for c in src_cube.dim_coords:
        dim = src_cube.coord_dims(c)[0]
        if dim not in xydimS:
            dimc_dim.append((c, (dim,)))
    dimc_dim.append((xcT, src_cube.coord_dims(xcS)))
    dimc_dim.append((ycT, src_cube.coord_dims(ycS)))
    return dimc_dim


def _get_auxc_dim(src_cube, target_cube, dmap):
    auxc_dim = []
    xydimS = tuple(dmap.values())
    xydimT = tuple(dmap.keys())
    for c in src_cube.aux_coords:
        dim = src_cube.coord_dims(c)
        if dim:
            if not any([dim_ in xydimS for dim_ in dim]):
                auxc_dim.append((c, dim))
        else:
            if isinstance(c, iris.coords.AuxCoord):
                auxc_dim.append((c, ()))
    for c in target_cube.aux_coords:
        dim = target_cube.coord_dims(c)
        if dim and all([dim_ in xydimT for dim_ in dim]):
            auxc_dim.append((c, tuple(dmap[dim_] for dim_ in dim)))
    return auxc_dim


def _get_scac(cube):
    return [c for c in cube.coords(dimensions=())]


def _regrid_slice(data, xS, yS, xT, yT, method, fill_value, rescale):
    from scipy.interpolate import griddata
    ind_ = ~np.isnan(data)
    if isinstance(xS, tuple) and len(xS) == 2:
        xx = np.hstack((xS[0][ind_], xS[1][ind_]))
        yy = np.hstack((yS[ind_], yS[ind_]))
        dd = np.hstack((data[ind_], data[ind_]))
    else:
        xx, yy, dd = xS[ind_], yS[ind_], data[ind_]
    tmp = griddata((xx, yy), dd,
                   (xT.ravel(), yT.ravel()), method=method,
                   fill_value=fill_value, rescale=rescale)
    tmp = tmp.reshape(xT.shape)
    return tmp


def _dmap(src_cube, target_cube):
    xT, yT = get_xy_dim_(target_cube)
    xS, yS = get_xy_dim_(src_cube)
    if xT is None or xS is None:
        raise Exception("missing 'x'/'y' dimcoord")
    return {target_cube.coord_dims(yT)[0]: src_cube.coord_dims(yS)[0],
            target_cube.coord_dims(xT)[0]: src_cube.coord_dims(xS)[0]}


def rgd_iris_(src_cube, target_cube, scheme=None):
    scheme = scheme if scheme else\
             iris.analysis.Linear(extrapolation_mode='mask')
    tmp = src_cube.regrid(target_cube, scheme)
    dmap = _dmap(src_cube, target_cube)
    for c in target_cube.aux_coords:
        dim = target_cube.coord_dims(c)
        if dim and all([dim_ in dmap.keys() for dim_ in dim]):
            tmp.add_aux_coord(c, tuple(dmap[dim_] for dim_ in dim))
    return tmp


def rgd_li_opt0_(src_cube, target_cube):
    try:
        tmp = rgd_iris_(src_cube, target_cube)
    except:
        tmp = rgd_scipy_(src_cube, target_cube)
    return tmp


def _cnp(points, cn='lb', **hgKA):
    if cn not in ('lb', 'lu', 'ru', 'rb'):
        raise ValueError("unknow conner")
    s0 = 'l' if 'l' in cn else 'r'
    s1 = 'l' if 'u' in cn else 'r'
    return half_grid_(half_grid_(points, side=s0, axis=1, **hgKA),
                      side=s1, axis=0, **hgKA)


def _bnds_2d_3d(bounds, shp, ax):
    return np.stack([robust_bc2_(bounds[:, i], shp, axes=ax)
                     for i in range(bounds.shape[-1])], axis=-1)


def _bnds_2p_4p(bounds, isX_=True):
    p4_ = (0, 0, 1, 1) if isX_ else (0, 1, 1, 0)
    return np.stack([extract_byAxes_(bounds, -1, i) for i in p4_],
                    axis=-1)


def _slice_ll_bnds(coord, shp, ax):
    isX_ = True if coord.name() == 'longitude' else False
    if coord.ndim == 1:
        if not coord.has_bounds():
            coord.guess_bounds()
        bounds = _bnds_2d_3d(coord.bounds, shp, ax)
        if bounds.shape[-1] == 2:
            bounds = _bnds_2p_4p(bounds, isX_)
        if isX_:
            rb = _lo_rb(coord.points)
            bounds = cyl_(bounds, rb=rb, lb=rb-360)
        return bounds
    else:
        if coord.has_bounds():
            bounds = coord.bounds
            if ax == (1, 0):
                bounds = np.moveaxis(bounds, 0, 1)
            if isX_:
                rb = _lo_rb(coord.points)
                bounds = cyl_(bounds, rb=rb, lb=rb-360)
            return bounds
        else:
            if isX_:
                hgKA = dict(loa='lo', rb=_lo_rb(coord.points))
            else:
                hgKA = dict(loa='la')
            points = _slice_ll_pnts(coord, shp, ax)
            bounds = [_cnp(points, cn=i, **hgKA)
                      for i in ('lb', 'lu', 'ru', 'rb')]
            return np.stack(bounds, axis=-1)


def _slice_ll_pnts(coord, shp, ax, df_=False):
    points = coord.points
    if ax == (1, 0):
        points = points.T
    if coord.ndim == 1:
        points = robust_bc2_(coord.points, shp, ax)
    if df_:
        return (points, _lld(points))
    else:
        return points


def _slice_ll_bpd(cube_slice):
    lo, la = get_loa_(cube_slice)
    shp = cube_slice.shape
    lo_d = cube_slice.coord_dims(lo)
    la_d = cube_slice.coord_dims(la)
    po, do = _slice_ll_pnts(lo, shp, lo_d, df_=True)
    pa, da = _slice_ll_pnts(la, shp, la_d, df_=True)
    return dict(lop=po.flatten(),
                lap=pa.flatten(),
                lob=_slice_ll_bnds(lo, shp, lo_d).reshape(-1, 4),
                lab=_slice_ll_bnds(la, shp, la_d).reshape(-1, 4),
                lod=do,
                lad=da)


_g = Geod(ellps="WGS84")


def _area_p(p, g=True):
    return abs(_g.geometry_area_perimeter(p)[0]) if g else p.area


def _iarea_ps(p0, p1, g=True):
    p01 = p0.intersection(p1)
    return _area_p(p01, g) if p01.area else 0.0


def _iwght(i, bpdT, bpdS, loR, laR):
    wght, rows, cols = [], [], []
    X = ind_inRange_(bpdS['lop'],
                     bpdT['lop'][i] - loR, bpdT['lop'][i] + loR,
                     r_=360)
    Y = ind_inRange_(bpdS['lap'],
                     bpdT['lap'][i] - laR, bpdT['lap'][i] + laR,
                     r_=360)
    ind = np.where(np.logical_and(X, Y))[0]
    if ind.size:
        pT = Polygon([(o_, a_)
                      for o_, a_ in zip(bpdT['lob'][i, :],
                                        bpdT['lab'][i, :])])
        if pT.area:
            for j in ind:
                pS = Polygon([(o_, a_)
                              for o_, a_ in zip(bpdS['lob'][j, :],
                                                bpdS['lab'][j, :])])
                #print(i, pT.wkt, j, pS.wkt)
                ia_ = _iarea_ps(pT, pS) / _area_p(pT)
                if ia_:
                    wght.append(ia_)
                    rows.append(i)
                    cols.append(j)
    return (wght, rows, cols)


def _weights(bpdT, bpdS, thr):
    import multiprocessing as mp
    nproc = min(mp.cpu_count(), 32)

    wght, rows, cols = [], [], []
    loR = (bpdT['lod'] + bpdS['lod']) / 2 
    laR = (bpdT['lad'] + bpdS['lad']) / 2
    if bpdT['lop'].size > 1e6:
        with mp.Pool(nproc) as P:
            tmp = P.starmap_async(_iwght, [(i, bpdT, bpdS, loR, laR)
                                           for i in range(bpdT['lop'].size)])
            out = tmp.get()
    else:
        out = (_iwght(i, bpdT, bpdS, loR, laR)
               for i in range(bpdT['lop'].size))
    for i, ii, iii in out:
        wght.extend(i)
        rows.extend(ii)
        cols.extend(iii)
    sparse_matrix = csc_matrix((wght, (rows, cols)),
                               shape=(bpdT['lop'].size, bpdS['lop'].size))
    sum_weights = sparse_matrix.sum(axis=1).getA()
    rows = np.where(sum_weights > thr)
    return (sparse_matrix, sum_weights, rows)


def _rgd_poly_info(src_cube, target_cube, thr=.5):
    bpdT = _slice_ll_bpd(target_cube)
    bpdS = _slice_ll_bpd(src_cube)
    rbT = _lo_rb(bpdT['lop'])
    rbS = _lo_rb(bpdS['lop'])
    if rbS != rbT:
        bpdS['lop'] = cyl_(bpdS['lop'], rbT, rbT - 360)
        bpdS['lob'] = cyl_(bpdS['lob'], rbT, rbT - 360)
    return _weights(bpdT, bpdS, thr) + (target_cube.shape,)


def _cn_rgd(src_cube, rgd_info, thr):
    sparse_matrix, sum_weights, rows, shp = rgd_info
    is_masked = np.ma.isMaskedArray(src_cube.data)
    if not is_masked:
        data = src_cube.data
    else:
        # Use raw data array
        data = src_cube.data.data
        # Check if there are any masked source points to take account of.
        is_masked = np.ma.is_masked(src_cube.data)
        if is_masked:
            # Zero any masked source points so they add nothing in output sums.
            mask = src_cube.data.mask
            data[mask] = 0.0
            # Calculate a new 'sum_weights' to allow for missing source points.
            # N.B. it is more efficient to use the original once-calculated
            # sparse matrix, but in this case we can't.
            # Hopefully, this post-multiplying by the validities is less costly
            # than repeating the whole sparse calculation.
            vcS = ~mask.flat[:]
            vfS = diags(np.array(vcS, dtype=int), 0)
            valid_weights = sparse_matrix * vfS
            sum_weights = valid_weights.sum(axis=1).getA()
            # Work out where output cells are missing all contributions.
            # This allows for where 'rows' contains output cells that have no
            # data because of missing input points.
            zero_sums = sum_weights <= thr
            # Make sure we can still divide by sum_weights[rows].
            sum_weights[zero_sums] = 1.0

    # Calculate sum in each target cell, over contributions from each source
    # cell.
    numerator = sparse_matrix * data.reshape(-1, 1)

    # Create a template for the weighted mean result.
    weighted_mean = np.ma.masked_all(numerator.shape, dtype=numerator.dtype)

    # Calculate final results in all relevant places.
    weighted_mean[rows] = numerator[rows] / sum_weights[rows]
    if is_masked:
        # Ensure masked points where relevant source cells were all missing.
        if np.any(zero_sums):
            # Make masked if it wasn't.
            weighted_mean = np.ma.asarray(weighted_mean)
            # Mask where contributing sums were zero.
            weighted_mean[zero_sums] = np.ma.masked
    return weighted_mean.reshape(shp)


class POLYrgd:
    def __init__(self, src_cube, target_cube, thr=.5):
        # Validity checks.
        if not isinstance(src_cube, iris.cube.Cube):
            raise TypeError("'src_cube' must be a Cube")
        if not isinstance(target_cube, iris.cube.Cube):
            raise TypeError("'target_cube' must be a Cube")
        # Snapshot the state of the cubes to ensure that the regridder
        # is impervious to external changes to the original source cubes.
        self._src_cube = src_cube.copy()
        xT, yT = get_xy_dim_(target_cube)
        xydT = target_cube.coord_dims(xT) + target_cube.coord_dims(yT)
        self._target_cube = target_cube[ind_shape_i_(target_cube.shape,
                                                     0,
                                                     axis=xydT)]
        self._thr = thr
        self._regrid_info = None

    def _info(self, out=False):
        if self._regrid_info is None:
            xS, yS = get_xy_dim_(self._src_cube)
            _S = self._src_cube.coord_dims(xS) + self._src_cube.coord_dims(yS)
            ind = ind_shape_i_(self._src_cube.shape, 0, axis=_S)
            self._regrid_info = _rgd_poly_info(self._src_cube[ind],
                                               self._target_cube,
                                               self._thr)
        if out:
            return self._regrid_info

    def __call__(self, src):
        # Validity checks.
        if not isinstance(src, iris.cube.Cube):
            raise TypeError("'src' must be a Cube")
        loG, laG = get_loa_(self._src_cube)
        src_grid = (loG.copy(), laG.copy())
        loS, laS = get_loa_(src)
        if (loS, laS) != src_grid:
            raise ValueError("The given cube is not defined on the same "
                             "source grid as this regridder.")

        dmap = _dmap(src, self._target_cube)
        xydT = tuple(dmap.keys())
        xydS = tuple(dmap.values())
        shT, shS = np.asarray(self._target_cube.shape), np.asarray(src.shape)

        nsh = shS.copy()
        for i in dmap.keys():
             nsh[dmap[i]] = shT[i]
        nsh = tuple(nsh)
        data = np.ma.empty(nsh, dtype=src.dtype)
        #dims for new cube
        dimc_dim = _get_dimc_dim(src, self._target_cube)
        auxc_dim = _get_auxc_dim(src, self._target_cube, dmap)

        cube = iris.cube.Cube(data, standard_name=src.standard_name,
                              long_name=src.long_name,
                              var_name=src.var_name, units=src.units,
                              attributes=src.attributes,
                              cell_methods=src.cell_methods,
                              dim_coords_and_dims=dimc_dim,
                              aux_coords_and_dims=auxc_dim,
                              aux_factories=None,
                              cell_measures_and_dims=None)

        self._info()
        for i in range(nSlice_(src.shape, xydS)):
            ind = ind_shape_i_(src.shape, i, axis=xydS)
            cube.data[ind] = _cn_rgd(src[ind], self._regrid_info, self._thr)
        return cube


def rgd_poly_(src_cube, target_cube, thr=.5):
    regrider = POLYrgd(src_cube, target_cube, thr)
    return regrider(src_cube)

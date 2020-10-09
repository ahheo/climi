"""
>--#¤&#¤%/%&(¤%¤#%"¤#"#¤%&¤#!"#%#%&()/=?=()/&%¤%&/()/£$€¥{[}]()=?=)(/&¤%&/()--<
>---------------You don't need to know what for this script is----------------<
>-----------------------------I don't know too--------------------------------<
>--#¤&&()/=?=()/&#¤%&/(=)(/&$€¥{[]}]\±$¡$£€@"#%¤#¤&/("#¤!&/(%&"#¤"¤%&#%/)/*#--<
* p25_75_           : 25th and 75th percentile      --> pre_hwmid_
* md_               : daily magnitude               --> pre_hwmid_
* tE                : class of heat/cold extreme    --> tE_
* tE_               : heat/cold extreme             --> get_hwmid_slice
* thr_              : threshold                     --> get_thr_cube
* dt_thr_           : difference to threshold: nD   --> pre_hwmi_
* subH              : class of sub-heat-wave        --> subH_
* subH_             : sub-heat-wave                 --> get_hwmi_slice,
                                                        data4kde_
* get_t_period_     : data of specified period      --> get_thr_,
                                                        get_kde_,
                                                        pre_hwmi_
* init_c_thr_       : initial cube of threshold     --> get_thr_
* get_thr_          : thresold: calculate|read      --> pre_hwmi_
* data4kde_         : annual max sub-heat-wave      --> get_kde_
* get_kde_          : kde (cdf and support values)  --> pre_hwmi_

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 04.09.2019
Date last modified: 08.04.2020
"""


import numpy as np
import iris
import os
import math
import warnings
import logging
import cf_units
from datetime import datetime
from dataclasses import dataclass, field
from typing import List
from uuuu.ffff import *
from uuuu.cccc import *


__all__ = ['p25_75_', 'md_', 'tE', 'tE_', 'thr_', 'dt_thr_', 'subH', 'subH_',
           'get_t_period_', 'init_c_thr_', 'get_thr_', 'data4kde_',
           'get_kde_']


def p25_75_(t_ref, ax_t):
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    p25 = tmp = np.nanpercentile(t_ref, 25, ax_t, keepdims=True)
    p75 = tmp = np.nanpercentile(t_ref, 75, ax_t, keepdims=True)
    return (p25, p75)


def thr_(t_ref, doy_ref, ax_t, pctl=90):
    """
    ... percentile-based threshold ...
    Parsed arguments:
          t_ref: array of t_max of reference period
        doy_ref: 1D array of day-or-year indices
           ax_t: axis of coord('time')
           pctl: percentile as threshold (default 90)
    Returns:
          t_thr: array of t_max threshold with shape[ax_t] = 366
    """

    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    doy_a = np.unique(doy_ref)
    if len(doy_a) < 360:
        raise Exception('doy less than 360!')
    doy = np.arange(1, 367, dtype='int')
    t_thr = extract_byAxes_(t_ref, ax_t, doy)

    t0 = l__('running thr_')
    _mp = np.product(t_thr.shape) > 5e6
    if _mp:
        ax_fn_mp_(t_ref, ax_t, _pctl, t_thr, pctl, doy_ref, doy)
    else:
        for i in doy:
            ind = ind_s_(t_ref.ndim, ax_t, ind_win_(doy_ref, i, 15))
            tmp = np.nanpercentile(t_ref[ind], pctl, ax_t, keepdims=True)
            t_thr[ind_s_(t_ref.ndim, ax_t, doy == i)] = tmp
    ll_('running thr_', t0)

    return t_thr


def _pctl(t_ref_sl, pctl, doy_ref, doy):
    out = np.full(doy.shape, np.nan)
    for i in doy:
        ind = ind_win_(doy_ref, i, 15)
        out[doy == i] = np.nanpercentile(t_ref_sl[ind], pctl)
    return out


def dt_thr_(y, doy_y, ax_t, t_thr, hw=True):
    """
    ... difference to threshold;
        y minus t_thr if hw else t_thr minus y ...

    Parsed arguments:
            y: array of origin
        doy_y: day-of-year indices of y
         ax_t: axis associated with coord('time')
        t_thr: array of t_max threshold
           hw: True not to converse the sign of results (default True)
    Returns:
           dy: array of difference to threshold;
               has the same shape of y
    """

    #adding day-of-year if not added
    dy = y.copy()
    doy_thr = np.arange(1, 367, dtype='int')

    t0 = l__('running dt_thr_')
    #subtraction with respect to day-of-year
    for idoy in np.unique(doy_y):
        dy[ind_s_(y.ndim, ax_t, doy_y == idoy)] -= \
            t_thr[ind_s_(t_thr.ndim, ax_t, doy_thr == idoy)]

    ll_('running dt_thr_', t0)

    return dy if hw else -dy


def md_(t, dt, p25, p75, hw=True):
    y = (t - p25) / (p75 - p25) if hw else (p75 - t) / (p75 - p25)
    y[np.isnan(dt)] = np.nan
    y[np.less(dt, 0, where=~np.isnan(dt))] = 0.
    return y


@dataclass
class tE:
    """
    _____heat/cold wave extremes_____
    class object includes:

        data:
            lCell (List[int]): lengh of each event
            uMagn (List[float]; can be nested List): unscaled magnitude

        methods:
            maxL(): maximum length of events
            nCell(): count of events
            sumL(): accumulated length of events
            meanL(): mean length of events
            max_uMagn(): max uMagn of events
            min_uMagn(): min uMagn of events
            mean_uMagn(): mean uMagn of events
    """

    lCell: List[int] = field(default_factory=list)
    uMagn: List[float] = field(default_factory=list)

    def nCell(self):
        return len(self.lCell)

    def maxL(self):
        return 0 if self.nCell() == 0 else max(self.lCell)

    def sumL(self):
        return 0 if self.nCell() == 0 else sum(self.lCell)

    def meanL(self):
        return 0 if self.nCell() == 0 else self.sumL() / self.nCell()

    def max_uMagn(self):
        return np.nan if self.nCell() == 0 else max(self.uMagn)

    def min_uMagn(self):
        return np.nan if self.nCell() == 0 else min(self.uMagn)

    def mean_uMagn(self):
        return np.nan if self.nCell() == 0 else sum(self.uMagn) / self.nCell()


@dataclass
class subH:
    """
    _____sub-heat/cold-wave_____
    class object includes:

        data:
            lCell (List[int]): lengh of each event
            uMagn (List[float]; can be nested List): unscaled magnitude
                                                     of each sub event;
                                                     grouped by events

        methods:
            maxL(): maximum length of events
            nCell(): count of events
            sumL(): accumulated length of events
            meanL(): mean length of events
            flt_uMagn(): ungroup uMagn
            max_uMagn(): max uMagn of sub events
            min_uMagn(): min uMagn of sub events
    """

    lCell: List[int] = field(default_factory=list)
    uMagn: List[float] = field(default_factory=list)

    def nCell(self):
        return len(self.lCell)

    def maxL(self):
        return 0 if self.nCell() == 0 else max(self.lCell)

    def sumL(self):
        return 0 if self.nCell() == 0 else sum(self.lCell)

    def meanL(self):
        return 0 if self.nCell() == 0 else self.sumL() / self.nCell()

    def flt_uMagn(self):
        return np.array(flt_l(self.uMagn))

    def max_uMagn(self):
        return np.nan if self.nCell() == 0 else max(self.flt_uMagn())

    def min_uMagn(self):
        return np.nan if self.nCell() == 0 else min(self.flt_uMagn())


def tE_(c1d_md, minL=3):
    """
    ... heat/cold wave extremes ...

    Parsed arguments:
        c1d_md: 1D array/cube of t_max minus threshold
          minL: minimum days to define a heat wave event (default 3)
    Returns:
        object of class tE:
            .lCell: length of each event;
            .uMagn: unscaled magnitude of each event;
    """

    #checking parsed arguments
    y = c1d_md.ravel()

    if minL < 2:
        raise ValueError("'minL' should be greater than 1!")

    #adding nan to the head of record
    y = np.concatenate([[np.nan,], y])

    #splitting extreme events using MarkValue
    h = np.split(y, np.where(np.isnan(y))[0])
    h = [x[1:] for x in h if len(x) > minL]

    #length of each event
    out1 = [len(x) for x in h]
    #magnitude of each event
    out2 = [sum(x) for x in h]

    return tE(lCell=out1, uMagn=out2)


def subH_(c1d_y, c1d_dy, minL=3):
    """
    ... sub heat-wave extremes ...

    Parsed arguments:
         c1d_y: 1D array/cube of t_max/t_min
        c1d_dy: 1D array/cube of t_max/tmin deviation to threshold
          minL: minimum days to define a heat wave event (default 3)
    Returns:
        object of class subH:
            .lCell: length of each event;
            .uMagn: unscaled magnitude of each sub-extreme event;
                    grouped by extreme cell
    """

    #checking parsed arguments
    if isinstance(c1d_dy, iris.cube.Cube):
        if c1d_dy.ndim > 1:
            raise Exception("Both Cubes should be 1D ('time')!")
        else:
            c1d_dy = nanMask_(c1d_dy.data)
    if isinstance(c1d_y, iris.cube.Cube):
        if c1d_y.ndim > 1:
            raise Exception("Both Cubes should be 1D ('time')!")
        else:
            c1d_y = nanMask_(c1d_y.data)

    c1d_y, c1d_dy = c1d_y.ravel(), c1d_dy.ravel()

    if len(c1d_y) != len(c1d_dy):
        raise Exception('first two arguments should have identical length!')

    if minL < 2:
        raise ValueError("'minL' should be greater than 1!")

    #creating index of data
    ind_c = np.arange(len(c1d_y))

    #marking no-extreme days
    c1d_dy[np.isnan(c1d_dy)] = -9999.
    ind_c[c1d_dy < 0] = -9999
    ind_c = np.concatenate([[-9999,], ind_c])

    #splitting extreme events using MarkValue
    h = np.split(ind_c, np.where(ind_c == -9999)[0])
    h = [x[1:] for x in h if len(x) > minL]

    #length of each events
    out1 = [len(x) for x in h]

    #deriving sub-extreme events (n days)
    h = [np.arange(x[0], x[0] + int(np.ceil(len(x) / minL) * minL)
                  ).reshape([-1, minL]) for x in h]
    def rpl_excee_(x, nn):
        x[x >= nn] = nn - 1
        return x
    h = [rpl_excee_(x, len(c1d_y)) for x in h]

    #magnitude of each sub-extreme event; grouped by extreme cell
    out2 = [[np.sum(c1d_y[r]) for r in x] for x in h]

    return subH(lCell=out1, uMagn=out2)


def get_t_period_(pn, dict_p, cube, yr_0, doy_0, ax_t):
    """
    Purpose:
        prepare data of certain period defined by dict_p[pn]
    Parsed arguments:
            pn: period name
        dict_p: period dictionary containing key [pn]
        *cube0: output from get_cube_(...)
    Returns:
        tuple:
              [0]   t_p: nD array t_max covering specified period
              [1]  yr_p: year index of cube
              [2] doy_p: day-of-year index of cube
              [3]    y0: beginning year
              [4]    y1: ending year
    """

    t0 = l__('preparing ' + pn)
    if pn == 'data':
        t_p, yr_p, doy_p = cube.data, yr_0, doy_0
        y0, y1 = min(yr_0), max(yr_0)
    else:
        y0, y1 = dict_p[pn][0], dict_p[pn][-1]
        ind = ind_inRange_(yr_0, y0, y1)
        t_p = extract_byAxes_(cube, ax_t, ind).data
        yr_p = yr_0[ind]
        doy_p = doy_0[ind]
    t_p = nanMask_(t_p)
    ll_('preparing ' + pn, t0)

    return (t_p, yr_p, doy_p, y0, y1)


def init_c_thr_(cube, ax_t, rref, pctl=90, hw=True):
    """
    Purpose:
        prepare initial cube of threshold for saving to a file
    Parsed arguments:
          cube: original cube from get_cube_(...)
          ax_t: 'time' axis of cube
        dict_p: period dictionary containing hey 'ref'
            hw: if for heat wave
    Returns:
         c_thr: initial cube for threshold
    """

    t0 = l__('preparing initial cube for threshold')
    c_thr = extract_byAxes_(cube, ax_t, np.s_[:366])
    #select 2000 as it is a leap year...
    c_thr.coord('time').units = cf_units.Unit('days since 1850-1-1',
                                              calendar='gregorian')
    d0 = cf_units.date2num(datetime(2000, 1, 1),
                           c_thr.coord('time').units.origin,
                           c_thr.coord('time').units.calendar)
    dimT = c_thr.coord('time').copy(np.arange(366, dtype='int32') + d0)
    c_thr.replace_coord(dimT)
    c_thr.var_name = 't_thr'
    c_thr.standard_name = 'air_temperature'
    if hw:
        lngn_ = 'T Threshold for Heat Wave'
        k_ = 'HWMI'
    else:
        lngn_ = 'T Threshold for Cold Wave'
        k_ = 'CWMI'
    v_ = '{}th percentile over period {}; '.format(pctl, rref)
    c_thr.long_name = lngn_
    c_thr.attributes.update({k_: v_ + "notes: coord('time').points are fake,"
                                 " but provide information day-of-year."})

    rm_t_aux_cube(c_thr)
    ll_('preparing initial cube for threshold', t0)

    return c_thr


def get_thr_(mdir, fnthr, dict_p, rref, *cube0, pctl=90, hw=True):
    """
    ... prepare data of threshold; try reading from file first if exist ...

    Parsed arguments:
           mdir: directory of intermedia data
          fnthr: file name
         dict_p: period dictionary containing key ['ref']
         *cube0: output from get_cube_
           pctl: percentile as threshold (default 90)
             hw: if for heat wave (default True)
    Returns:
          t_thr: array of t_max threshold with shape[ax_t] = 366
           fr_f: if derived from an existing file
          t_ref: nD array of t_max of reference period
         yr_ref: year index of t_ref
        doy_ref: day-of-year index of t_ref
    """

    fn = mdir + fnthr
    fr_f = True
    if os.path.isfile(fn):
        logging.info(' reading thr from file ...')
        t_thr = iris.load_cube(fn).data
        t_thr = nanMask_(t_thr)
        logging.info(' threshold ready...')
        logging.info(' ')
        return (t_thr, fr_f)
    elif len(cube0) == 4:
        t_ref, yr_ref, doy_ref = get_t_period_('ref', dict_p, *cube0)[:3]
        t_thr = thr_(t_ref, doy_ref, cube0[3], pctl=pctl)
        fr_f = False
        c_thr = init_c_thr_(cube0[0], cube0[3], rref, pctl=pctl, hw=hw)
        tmp = t_thr.copy()
        tmp[np.isnan(tmp)] = -9999.
        c_thr.data = np.ma.masked_equal(tmp, -9999.)
        os.makedirs(mdir, exist_ok=True)
        iris.save(c_thr, fn)
        logging.info(' threshold ready...')
        logging.info(' ')
        return (t_thr, fr_f, t_ref, yr_ref, doy_ref)
    else:
        logging.exception("File not exist and failed to created without "
                          "providing output from get_cube_(...)! EXIT")
        raise


def _extr_sort_n(y1d, xx, nORx='x'):
    y1d = np.sort(y1d)
    return y1d[-xx:] if nORx == 'x' else y1d[:xx]


def _extr_pctl(y1d, xx, nORx='x'):
    if nORx == 'x':
        tmp = np.percentile(y1d, xx)
        return y1d[y1d >= tmp]
    else:
        tmp = np.percentile(y1d, 100 - xx)
        return y1d[y1d <= tmp]


def data4kde_(t_ref, dt_ref, yr_ref, minL=3, hw=True, mtd='ymax'):
    """
    ... derive annual max unscaled magnitude of sub-heat-wave for kde fitting

    Parsed arguments:
         t_ref: 1D array of t_max covering reference period
        dt_ref: 1D array of t_max minus threshold
        yr_ref: year index
          minL: minimum days to define a heat/cold wave event (default 3)
            hw: if for heat wave (default True)
           mtd: method; can be one of ('ymax'(default), 'pctlxx', 'maxnxx')
    Returns:
        annual max unscaled magnitude of sub-heat-wave
    """

    if not len(t_ref) == len(dt_ref) == len(yr_ref):
        raise Exception('length of data and year index should be idetical!')
    yrs = np.unique(yr_ref)

    if mtd == 'ymax':
        obs = list()
        for yr in yrs:
            if hw:
                obs.append(subH_(t_ref[yr_ref == yr],
                                 dt_ref[yr_ref == yr],
                                 minL).max_uMagn())
            else:
                obs.append(subH_(t_ref[yr_ref == yr],
                                 dt_ref[yr_ref == yr],
                                 minL).min_uMagn())
        obs = np.asarray(obs, np.float64)
        obs = obs[~np.isnan(obs)]
        return obs
    else:
        xx = int(mtd[4:])
        nORx = 'x' if hw else 'n'
        umagns = np.asarray(subH_(t_ref, dt_ref, minL).flt_uMagn(),
                            np.float64)
        if len(umagns) == 0:
            return umagns
        else:
            if mtd[:4] == 'pctl':
                if not 0 < xx < 100:
                    raise Exception("unknown 'mtd'!")
                return _extr_pctl(umagns, xx, nORx)
            elif mtd[:4] == 'maxn':
                return _extr_sort_n(umagns, xx, nORx)
            else:
                raise Exception("unknown 'mtd'!")


def get_kde_(mdir, fnkde, dict_p, kde_opts, minL, *othr,
             pctl=90, hw=True, mtd='ymax'):
    """
    ... prepare kde (cdf and support values);
        try reading from file first if exist ...

    Parsed arguments:
            mdir: directory of intermedia data
             fn_: prename defined
            fn_e: end part of file name
          dict_p: period dictionary containing key ['ref']
        kde_opts: options for kde
            minL: length of sub heat wave
           *othr: output from get_thr_(...) + (get_cube_(...),)
            pctl: percentile as threshold (default 90)
              hw: if for heat-wave (default true)
             mtd: method; can be one of ('ymax':default, 'pctlxx', 'maxnxx')
    Returns:
        kde[cdf]: cdf values of KernelDistributionEstimate object
          kde[x]: support values of KernelDistributionEstimate object
    """

    fn = mdir + fnkde
    if os.path.isfile(fn):
        logging.info(' reading kde from file...')
        kde = np.load(fn)
    elif len(othr) >= 3:
        if othr[1]:
            t_ref, yr_ref, doy_ref = get_t_period_('ref', dict_p,
                                                   *othr[2])[:3]
            ax_t = othr[2][3]
            dt_ref = dt_thr_(t_ref, doy_ref, ax_t, othr[0], hw=hw)
        else:
            t_ref, yr_ref, doy_ref, ax_t = othr[2], othr[3], \
                                           othr[4], othr[5][3]
            dt_ref = dt_thr_(t_ref, doy_ref, ax_t, othr[0], hw=hw)
        shp = list(t_ref.shape)
        if 'gridsize' in kde_opts:
            shp[ax_t] = kde_opts['gridsize']
        else:
            shp[ax_t] = 1024
        t0 = l__('calculating kde based on reference data')
        kde = {}
        kde['cdf'] = np.full(shp, np.nan)
        kde['x'] = np.full(shp, np.nan)
        ax_fn_mp_((t_ref, dt_ref), ax_t, kde_fr_data_, (kde['cdf'], kde['x']),
                  yr_ref, minL, hw, mtd, kde_opts)
        #for i in range(nSlice_(shp, ax_t)):
        #    nsl = nSlice_(shp, ax_t)
        #    dgt = r'{:d}'.format(math.floor(math.log10(nsl)) + 1)
        #    logging.info((' slice #{:0' + dgt + 'd}/{:d}').format(i, nsl))
        #    ind = ind_shape_i_(shp, i, ax_t)
        #    obs = data4kde_(t_ref[ind], dt_ref[ind], yr_ref,
        #                    minL=minL, hw=hw, mtd=mtd)
        #    if len(obs) != 0:
        #        out = kde_(obs, **kde_opts)
        #        if hw:
        #            slice_back_(kde['cdf'], out.cdf, i, ax_t)
        #        else:
        #            slice_back_(kde['cdf'], out.sf, i, ax_t)
        #        slice_back_(kde['x'], out.support, i, ax_t)
        os.makedirs(mdir, exist_ok=True)
        np.savez(fn, cdf=kde['cdf'], x=kde['x'])
        ll_('calculating kde based on reference data', t0)
    else:
        logging.exception("File not exist and failed to created without "
                          "providing get_thr_(...) + (get_cube_(...),)! "
                          "EXIT")
        raise

    logging.info(' kde ready...')
    logging.info(' ')
    return kde


def kde_fr_data_(t_ref, dt_ref, yr_ref, minL, hw, mtd, kde_opts):
    obs = data4kde_(t_ref, dt_ref, yr_ref, minL=minL, hw=hw, mtd=mtd)
    if len(obs) != 0:
        out = kde_(obs, **kde_opts)
        return (out.cdf, out.support) if hw else (out.sf, out.support)
    else:
        return None

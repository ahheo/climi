"""
>--#########################################################################--<
>------------------------Heat/Cold Wave Magnitude Index-----------------------<
>------------------------Warm/Cold Spell Duration Index-----------------------<
>--#########################################################################--<

*****REFERENCE***** : Russo, Simone, et al. "Magnitude of extreme heat waves
                      in present climate and their projection in a warming
                      world." Journal of Geophysical Research: Atmospheres
                      119.22 (2014): 12-500.

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 16.09.2019
Date last modified: 08.04.2020
"""


import numpy as np
import iris
import iris.coord_categorisation as cat
import math
import logging
from climi.uuuu import *
from .zzzz import *


__all__ = ['hwmi__']


def pre__(dCube=None, rCube=None,
          dict_p=None, mdir=None, rref=None, fnkde=None, fnthr=None,
          minL=None, pctl=None, mtd=None, mm=None,
          kdeo={}, pn='data', hw=True):
    """
    ... prepare data for calculating get_hwmi_() ...

    Parsed arguments:
        **dd__: dictionary keys:
                'dCube', 'rCube',
                'dict_p', 'mdir', 'rref', 'fnkde', 'fnthr',
                'minL', 'pctl', 'mtd', 'kdeo', 'pn'
            hw: True for heat-wave otherwise for cold-wave
    Returns:
           ref: dictionary keys: 't', 'dt', 'yr', 'doy'
                -   t: nD array of t_max of reference period
                -  dt: nD array of t_max difference to threshold
                -  yr: year index associated with t/dt
                - doy: day-of-year index associated with t/dt
           inv: similar to ref but for investigation period
          ax_t: 'time' axis of data
        c_hwmi: initiatial cube for 'hwmi'
        c_wsdi: initiatial cube for 'wsdi'
    """

    if any([i is None for i in [dCube, rCube, dict_p, mdir, rref,
                                fnkde, fnthr, minL, pctl, mtd, mm]]):
        return (None,) * 6

    ax_t = dCube[3]

    #getting data of investigation:
    t_inv, yr_inv, doy_inv, y0, y1 = get_t_period_(pn, dict_p, *dCube)

    #geting threshold
    othr = get_thr_(mdir, fnthr, dict_p, rref, *rCube, pctl=pctl, hw=hw)
    dt_inv = dt_thr_(t_inv, doy_inv, ax_t, othr[0], hw=hw)

    #getting kde
    kdeArg = othr + (rCube,)
    kde = get_kde_(mdir, fnkde, dict_p, kdeo, minL, *kdeArg,
                   pctl=pctl, hw=hw, mtd=mtd)

    #creating a cube for saving hwmi (and wsdi)
    t0 = ('preparing initial cube')

    idn_ = ['hwmi', 'Heat Wave Magnitude Index',
            'wsdi', 'Warm Spell Duration Index'] if hw else \
           ['cwmi', 'Cold Wave Magnitude Index',
            'csdi', 'Cold Spell Duration Index']

    addattr_ = {'NOTE': 'Ref. period for ' + idn_[0] + ': {}'.format(rref)}
    c_hwmi = initAnnualCube_(dCube[0], [y0, y1],
                             name=idn_[1],
                             units=1,
                             var_name=idn_[0],
                             attrU=addattr_,
                             mm=mm)
    addattr_ = {'NOTE': 'Ref. period for ' + idn_[2] + ': {}'.format(rref)}
    c_wsdi = initAnnualCube_(dCube[0], [y0, y1],
                             name=idn_[3],
                             units='days',
                             var_name=idn_[2],
                             attrU=addattr_,
                             mm=mm)
    rm_t_aux_cube(c_hwmi)
    rm_t_aux_cube(c_wsdi)
    logging.info(' data ready for get_hwmi_/get_cwmi ...')
    logging.info(' ')

    return (kde,
            {'t': t_inv, 'dt': dt_inv, 'yr': yr_inv},
            ax_t,
            c_hwmi,
            c_wsdi,
            minL)


def get_hwmi_slice(k_cdf, k_x, t_inv, dt_inv, yr_inv, minL=3):
    """
    ... get hwmi/cwmi and wsdi/csdi along 1D cube slice ("time") ...

    Parsed arguments:
         k_cdf: 1D array of kde cdf values
           k_x: 1D array of kde support values
         t_inv: 1D array of t_max of investigation period
        dt_inv: 1D array of t_inv difference to threshold
        yr_inv: D array of investigation year-index
          minL: minimum days to define a heat wave event
    Returns:
          hwmi: 1D array of hwmi
          wsdi: 1D array of wsdi
    """

    if not (len(t_inv) == len(dt_inv) == len(yr_inv)):
        raise Exception('length of data and year index should be identical!')

    yrs = np.unique(yr_inv)

    hwmi = np.zeros(yrs.shape)
    wsdi = np.zeros(yrs.shape, dtype='int')

    if not np.isnan(k_cdf[0]):
        for i, yr in enumerate(yrs):
            subH_i =subH_(t_inv[yr_inv == yr], dt_inv[yr_inv == yr], minL)
            if subH_i.nCell() != 0:
                hwmi[i] = np.max([np.sum(np.interp(r, k_x, k_cdf))
                                  for r in subH_i.uMagn])
                wsdi[i] = subH_i.maxL()

    return (hwmi, wsdi)


def get_hwmi_(kde, inv, ax_t, c_hwmi, c_wsdi, minL):
    """
    ... get hwmi/cwmi and wsdi/csdi over nD cube ...

    Parsed arguments:
           kde: dictionary keys: 'cdf', 'x'
                - cdf: cdf values of KernelDistributionEstimate object
                -   x: support values of KernelDistributionEstimate object
           inv: dictionary keys: 't', 'dt', 'yr', 'doy'
                -   t: nD array of t_max of period for investigation
                -  dt: nD array of t_max difference to threshold
                -  yr: year index associated with t/dt
                - doy: day-of-year index associated with t/dt
          ax_t: 'time' axis of data
        c_hwmi: initial cube for 'hwmi'
        c_wsdi: initial cube for 'wsdi'
          minL: minimum days to define a heat wave event
    Returns:
        dictionary with keys: "hwmi", "wsdi", data in cube
    """

    t0 = l__('get_hwmi_() loop over slices')
    ax_fn_mp_((kde['cdf'], kde['x'], inv['t'], inv['dt']),
              ax_t,
              get_hwmi_slice,
              iris.cube.CubeList([c_hwmi, c_wsdi]),
              inv['yr'], minL=minL)
    #for i in range(nSlice_(c_hwmi.shape, ax_t)):
    #    ind = ind_shape_i_(c_hwmi.shape, i, ax_t)
    #    try:
    #        if c_hwmi[ind][0].data.mask:
    #            continue
    #    except AttributeError:
    #        pass
    #    nsl = nSlice_(c_hwmi.shape, ax_t)
    #    dgt = r'{:d}'.format(math.floor(math.log10(nsl)) + 1)
    #    logging.info((' slice #{:0' + dgt + 'd}/{:d}').format(i, nsl))
    #    out = get_hwmi_slice(kde['cdf'][ind], kde['x'][ind],
    #                         inv['t'][ind], inv['dt'][ind], inv['yr'],
    #                         minL=minL)
    #    slice_back_(c_hwmi, out['hwmi'], i, ax_t)
    #    slice_back_(c_wsdi, out['wsdi'], i, ax_t)
    ll_('get_hwmi_() loop over slices', t0)

    return {'hwmi': c_hwmi, 'wsdi': c_wsdi}


def hwmi__(hw=True, **dd__):
    """
    ... pre__ & get_hwmi_ ...
    """
    #logging.info('{}'.format(np.all(rCube[0][0, :, :].data.mask)))
    tmp = pre__(hw=hw, **dd__)
    if tmp[0] is None:
        return None
    else:
        out = get_hwmi_(*tmp)
        return out

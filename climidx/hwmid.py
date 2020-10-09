"""
>--#¤&#¤%/%&(¤%¤#%"¤#"#¤%&¤#!"#%#%&()/=?=()/&%¤%&/()/£$€¥{[}]()=?=)(/&¤%&/()--<
>---------------------Heat/Cold Wave Magnitude Index daily--------------------<
>------------------------Warm/Cold Spell Duration Index-----------------------<
>--#¤&&()/=?=()/&#¤%&/(=)(/&$€¥{[]}]\±$¡$£€@"#%¤#¤&/("#¤!&/(%&"#¤"¤%&#%/)/*#--<

*****REFERENCE***** : Russo, Simone, et al. "Top ten European heatwaves since
                      1950 and their occurrence in the coming decades",
                      Environmental Research Letters, 10 (2015) 124003.

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 11.06.2020
Date last modified: 11.06.2020
"""


import numpy as np
import iris
import iris.coord_categorisation as cat
import math
import logging
from uuuu.ffff import *
from uuuu.cccc import *
from .zzzz import *


__all__ = ['hwmid__']


def pre__(dCube=None, rCube=None,
          dict_p=None, mdir=None, rref=None, fnthr=None,
          minL=None, pctl=None, pn='data', hw=True):
    """
    ... prepare data for calculating get_hwmi_() ...

    Parsed arguments:
        **dd__: dictionary keys:
                'dCube', 'rCube',
                'dict_p', 'mdir', 'rref', 'fnthr',
                'minL', 'pctl', 'pn'
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
                                fnthr, minL, pctl]]):
        return (None,) * 6

    ax_t = dCube[3]

    #getting data of investigation:
    t_inv, yr_inv, doy_inv, y0, y1 = get_t_period_(pn, dict_p, *dCube)

    #geting threshold
    othr = get_thr_(mdir, fnthr, dict_p, rref, *rCube, pctl=pctl, hw=hw)
    dt_inv = dt_thr_(t_inv, doy_inv, ax_t, othr[0], hw=hw)

    #getting md
    if len(othr) > 2:
        t_ref = othr[2]
    else:
        t_ref = get_t_period_('ref', dict_p, *rCube)[0]
    p25, p75 = p25_75_(t_ref, ax_t)
    md_inv = md_(t_inv, dt_inv, p25, p75, hw=hw)

    #creating a cube for saving hwmi (and wsdi)
    t0 = ('preparing initial cube')

    idn_ = ['hwmid', 'Heat Wave Magnitude Index daily',
            'wsdi', 'Warm Spell Duration Index'] if hw else \
           ['cwmid', 'Cold Wave Magnitude Index daily',
            'csdi', 'Cold Spell Duration Index']

    addattr_ = {'NOTE': 'Ref. period for ' + idn_[0] + ': {}'.format(rref)}
    c_hwmi = initAnnualCube_(dCube[0], [y0, y1],
                             name=idn_[1],
                             units=1,
                             var_name=idn_[0],
                             attrU=addattr_)
    addattr_ = {'NOTE': 'Ref. period for ' + idn_[2] + ': {}'.format(rref)}
    c_wsdi = initAnnualCube_(dCube[0], [y0, y1],
                             name=idn_[3],
                             units='days',
                             var_name=idn_[2],
                             attrU=addattr_)
    rm_t_aux_cube(c_hwmi)
    rm_t_aux_cube(c_wsdi)
    logging.info(' data ready for get_hwmid_/get_cwmid ...')
    logging.info(' ')

    return ({'md': md_inv, 'yr': yr_inv},
            ax_t,
            c_hwmi,
            c_wsdi,
            minL)


def get_hwmid_slice(md_inv, yr_inv, minL):
    """
    ... get hwmid/cwmid and wsdi/csdi along 1D cube slice ("time") ...

    Parsed arguments:
         k_cdf: 1D array of kde cdf values
           k_x: 1D array of kde support values
        md_inv: 1D array of hwmid daily magnitude
        yr_inv: D array of investigation year-index
          minL: minimum days to define a heat wave event
    Returns:
          hwmi: 1D array of hwmid
          wsdi: 1D array of wsdi
    """

    if len(md_inv) != len(yr_inv):
        raise Exception('length of data and year index should be identical!')

    yrs = np.unique(yr_inv)

    hwmi = np.zeros(yrs.shape)
    wsdi = np.zeros(yrs.shape, dtype='int')

    for i, yr in enumerate(yrs):
        tE_i =tE_(md_inv[yr_inv == yr], minL)
        if tE_i.nCell() != 0:
            hwmi[i] = tE_i.max_uMagn()
            wsdi[i] = tE_i.maxL()

    return (hwmi, wsdi)


def get_hwmid_(inv, ax_t, c_hwmi, c_wsdi, minL):
    """
    ... get hwmi/cwmi and wsdi/csdi over nD cube ...

    Parsed arguments:
           inv: dictionary keys: 'md', 'yr'
                -  md: nD array of daily magnitude
                -  yr: year index associated with t/dt
          ax_t: 'time' axis of data
        c_hwmi: initial cube for 'hwmi'
        c_wsdi: initial cube for 'wsdi'
          minL: minimum days to define a heat wave event
    Returns:
        dictionary with keys: "hwmi", "wsdi", data in cube
    """

    t0 = l__('get_hwmid_() loop over slices')
    ax_fn_mp_(inv['md'],
              ax_t,
              get_hwmid_slice,
              iris.cube.CubeList([c_hwmi, c_wsdi]),
              inv['yr'], minL=minL)
    ll_('get_hwmid_() loop over slices', t0)

    return {'hwmi': c_hwmi, 'wsdi': c_wsdi}


def hwmid__(hw=True, **dd__):
    """
    ... pre_hwmi_ & get_hwmi_ ...
    """
    tmp = pre__(hw=hw, **dd__)
    out = get_hwmid_(*tmp)
    return out

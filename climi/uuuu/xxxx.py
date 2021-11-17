"""                                                                             
>--#########################################################################--< 
>--------------------------------xr functions---------------------------------<
>--#########################################################################--<
*
*
############################################################################### 
            Author: Changgui Lin                                                
            E-mail: changgui.lin@smhi.se                                        
      Date created: 06.09.2019                                                  
Date last modified: 11.11.2020                                                 
"""

import xarray as xr
import pandas as pd
import numpy as np
import igra

from .ffff import *


__all__ = [
           'dds_',
           'xr_daily_mean_',
           'xr_monthly_mean_',
           'xr_seasonal_mean_',
           'xr_annual_mean_',
           'uvds_'
          ]


def dds_(ds, ff='D', nn=None):
    dsr = ds.resample(date=ff)
    o = dsr.mean()
    if nn:
        tmp = dsr.count()
        o = o.where(tmp >= nn)
    return o


def xr_daily_mean_(ds, nn=None):
    return dds_(ds, ff='D', nn=nn)


def xr_monthly_mean_(ds, nn=None):
    return dds_(ds, ff='M', nn=nn)


def xr_seasonal_mean_(ds, nn=None):
    return dds_(ds, ff='QS-DEC', nn=nn)


def xr_annual_mean_(ds, nn=None):
    return dds_(ds, ff='A', nn=nn)


def uvds_(ds):
    uwind, vwind = windds2uv_(ds.winds, ds.windd)
    uwind.assign_attrs(units='m/s', standard_name='u-component wind')
    vwind.assign_attrs(units='m/s', standard_name='v-component wind')
    o = ds.assign(uwind=uwind, vwind=vwind)
    return o.drop_vars('windd')

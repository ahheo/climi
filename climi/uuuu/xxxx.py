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
import os

from .ffff import *


__all__ = [
           'dds_',
           'xr_daily_mean_',
           'xr_monthly_mean_',
           'xr_seasonal_mean_',
           'xr_annual_mean_',
           'uvds_',
           'gsod_attrs_',
           'gsod_single_',
           'gsod_'
          ]


_djn = os.path.join


def dds_(ds, ff='D', nn=None):
    '''
    ... resample xarray dataset (temporally) ...
    inputs:
        ds: xarray.Dataset with a 'date' dim
        ff: target frequency
            'D': daily, equal to xr_daily_mean_
            'M': monthly, equal to xr_monthly_mean_
            'QS-DEC':, seasonal, equal to xr_seasonal_mean_
            'A': annual, equal to xr_annual_mean_
            and others 
        nn: min number of samples for output valid values
    output:
        same type as ds
    '''
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


def _missing_to_nan_column(df, name, value):
    df[name] = df[name].replace(value, np.nan)

GSOD_MV = dict(
        temp=(9999.9, dict(units='K')),
        dewp=(9999.9, dict(units='K')),
        slp=(9999.9, dict(units='mb')),
        stp=(9999.9, dict(units='mb')),
        visib=(999.9, dict(units='km')),
        wdsp=(999.9, dict(units='m/s')),
        mxspd=(999.9, dict(units='m/s')),
        gust=(999.9, dict(units='m/s')),
        max=(9999.9, dict(units='K')),
        min=(9999.9, dict(units='K')),
        prcp=(99.99, dict(units='mm')),
        sndp=(999.9, dict(units='mm'))
        )

GSOD_VAR = [
        'temp', 'temp_attributes',
        'dewp', 'dewp_attributes',
        'slp', 'slp_attributes',
        'stp', 'stp_attributes',
        'visib', 'visib_attributes',
        'wdsp', 'wdsp_attributes',
        'mxspd',
        'gust',
        'max',
        'max_attributes',
        'min', 'min_attributes',
        'prcp', 'prcp_attributes',
        'sndp',
        'frshtt']

def _f2k(x):
    return (x - 32) * 5 / 9 + 273.15

def _knot2ms(x):
    return x * 0.514444445

def _mi2km(x):
    return x * 1.609344

def _in2mm(x):
    return x * 25.4


def gsod_attrs_(csvfile):
    o = pd.read_csv(csvfile, dtype={'STATION': str}).rename(
            str.lower, axis='columns')
    attrs = dict(
            usaf=o.station[0][:6],
            wban=o.station[0][-5:],
            name=o.name[0],
            lat=o.latitude[0],
            lon=o.longitude[0],
            elv=o.elevation[0]
            )
    return attrs


def gsod_single_(csvfile, variables=None, _attrs=False):
    o = pd.read_csv(csvfile, dtype={'STATION': str}).rename(
            str.lower, axis='columns')
    o['date'] = pd.to_datetime(o.date)
    o = o.set_index('date')
    if _attrs:
        attrs = dict(
                usaf=o.station[0][:6],
                wban=o.station[0][-5:],
                name=o.name[0],
                lat=o.latitude[0],
                lon=o.longitude[0],
                elv=o.elevation[0]
                )
    o = o.drop(
            columns=['station', 'latitude', 'longitude', 'elevation', 'name']
            )
    if variables is None:
        variables = set(o.columns)
    else:
        assert set(variables).issubset(o.columns)
        o = o.loc[:, variables]
    for cc in variables:
        if cc in GSOD_MV.keys():
            _missing_to_nan_column(o, cc, GSOD_MV[cc][0])
    for cc in ['temp', 'dewp', 'max', 'min']:
        if cc in o.columns:
            o[cc] = _f2k(o[cc])
    for cc in ['wdsp', 'mxspd', 'gust']:
        if cc in o.columns:
            o[cc] = _knot2ms(o[cc])
    if 'visib' in o.columns:
        o['visib'] = _mi2km(o['visib'])
    for cc in ['prcp', 'sndp']:
        if cc in o.columns:
            o[cc] = _in2mm(o[cc])
    return (o, attrs) if _attrs else o


def gsod_(csvfiles=None,
          root=None, usaf=None, wban='99999', ys=None, ye=None,
          variables=None, _attrs=False):
    import glob
    import re
    if csvfiles is not None:
        if isinstance(csvfiles, str):
            o = gsod_single_(csvfiles, variables=variables)
            if _attrs:
                attrs = gsod_attrs_(csvfiles)
        elif isIter_(csvfiles, xi=str):
            o = [gsod_single_(i, variables=variables) for i in csvfiles]
            o = pd.concat(o)
            if _attrs:
                attrs = gsod_attrs_(csvfiles[0])
        else:
            raise TypeError("unknown 1st argin!!!")
        o = o.to_xarray()
        if _attrs:
            o = o.assign_attrs(attrs)
        for cc in o.data_vars:
            if cc in GSOD_MV.keys():
                o[cc] = o[cc].assign_attrs(**GSOD_MV[cc][1])
        return o
    elif all([i is not None for i in (root, usaf)]):
        root = root[:-1] if root[-1] == '/' else root
        fnfmt = _djn(root, '*', '{:0>6}{:0>5}.csv'.format(usaf, wban))
        fns = sorted(glob.glob(fnfmt))
        def _year_fr_fn(fn):
            return int(fn.replace(root, '')[1:5])
        if ys is not None and ye is not None:
            assert ys <= ye
            fns = [i for i in fns if _year_fr_fn(i) in range(ys, ye + 1)]
        if fns:
            return gsod_(csvfiles=fns, variables=variables, _attrs=_attrs)
        else:
            raise Exception("no file found!!!")
    else:
        raise Exception("You may know more what is going wrong than me!!!")

"""
>--#¤&#¤%/%&(¤%¤#%"¤#"#¤%&¤#!"#%#%&()/=?=()/&%¤%&/()/£$€¥{[}]()=?=)(/&¤%&/()--<
>--------------------Calculations of Some Climate Indices---------------------<
>------------------------------by Changgui Lin--------------------------------<
>----------------------------------at SMHI------------------------------------<
>--------------------------------07 Feb 2020--------------------------created-<
>--------------------------------07 Feb 2020-------------------------modified-<
>--#¤&&()/=?=()/&#¤%&/(=)(/&$€¥{[]}]\±$¡$£€@"#%¤#¤&/("#¤!&/(%&"#¤"¤%&#%/)/*#--<
* mEffPr_                     : cube, pr, evspsbl, climatology (season, year)
* mPrrn_                      : cube, pr, prsn, climatology (season, year)
* mDTR_                       : cube, tasmax, tasmin
* dHumiWarmDays_              : 1d arr, hurs, tas
* dHumiWarmDays_cube          : cube, hurs, tas
* dPr7_                       : 1d arr, pr
* dPr7_cube                   : cube, pr
* dLongestDryDays_            : 1d arr, pr
* dDryDays_                   : cube, pr
* dExtrPrDays_                : cube, pr
* dWarmDays_                  : cube, tasmax
* dCalmDays_                  : cube, sfcWind
* dConWarmDays_               : 1d arr, tasmax
* dConCalmDays_               : 1d arr, sfcWind
* dColdDays_                  : cube, tasmax
* dDegDay20_                  : 1d arr, tas
* dDegDay8_vegSeason_         : 1d arr, tas
* dDegDay17_                  : 1d arr, tas
* dEndSpringFrost_            : 1d arr, tasmin
* dFirstDayWithoutFrost_      : 1d arr, tasmin
* dFrostDays_                 : cube, tasmin
* dTropicNights_              : cube, tasmin
* dZeroCrossingDays_          : 1d arr, tasmax, tasmin
* dZeroCrossingDays_cube      : cube, tasmax, tasmin
* dStartEndVegSeason_         : 1d arr, tas, thr=5/2
* dWindyDays_                 : cube, wsgsmax
* dFreezRainDays_             : 1d arr, pr, ps, tas, ta925, ta850, ta700,
                                                     hus925, hus850, hus700
* dSncDays_                   : cube, snc
* dSndGT15Days_               : cube, snd
* dSndLE10Days_               : cube, snd
* dSndGT10LE20Days_           : cube, snd
* rho_fr_t_q_p_               : nd array, t, q, p
* dRhoPL_                     : 1d arr, ta, hus
* dRhoPS_                     : 1d arr, tas, huss, ps
* dDegDay_                    : cube, tas/tasmax/tasmin
* dPRSN_fr_PR_T_              : cube, pr, tas
* dPRRN_fr_PR_T_              : cube, pr, tas
* dRainOnSnow_                : cube, prrn, snc, snw
* dColdRainDays_              : cube, pr, tas
* dWarmSnowDays_              : cube, pr, tas
* dColdPRRNdays_              : cube, pr, prsn, tas
* dWarmPRSNdays_              : cube, prsn, tas
"""


import numpy as np
import iris
import iris.coord_categorisation as cat
from uuuu import *


__all__ = ['mEffPr_',
           'mPrrn_',
           'mDTR_',
           'dHumiWarmDays_',
           'dHumiWarmDays_cube',
           'dPr7_',
           'dPr7_cube',
           'dLongestDryDays_',
           'dDryDays_',
           'dExtrPrDays_',
           'dWarmDays_',
           'dCalmDays_',
           'dConWarmDays_',
           'dConCalmDays_',
           'dColdDays_',
           'dDegDay20_',
           'dDegDay8_vegSeason_',
           'dDegDay17_',
           'dEndSpringFrost_',
           'dFirstDayWithoutFrost_',
           'dFrostDays_',
           'dTropicNights_',
           'dZeroCrossingDays_',
           'dZeroCrossingDays_cube',
           'dStartEndVegSeason_',
           'dWindyDays_',
           'dFreezRainDays_',
           'dSncDays_',
           'dSndGT15Days_',
           'dSndLE10Days_',
           'dSndGT10LE20Days_',
           'rho_fr_t_q_p_',
           'dRhoPL_',
           'dRhoPS_',
           'dDegDay_',
           'dPRSN_fr_PR_T_',
           'dPRRN_fr_PR_T_',
           'dRainOnSnow_',
           'dColdRainDays_',
           'dWarmSnowDays_',
           'dColdPRRNdays_',
           'dWarmPRSNdays_']


K0 = 273.15


def _rl(x):
    return x.data if isinstance(x, iris.cube.Cube) else x


def mEffPr_(cPr, cET, freq):
    c = cPr.copy(cPr.data - cET.data)
    pst_(c, 'effective precipitation', var_name='eff_pr')
    return pSTAT_cube(c, 'MEAN', *freq)


def mPrrn_(cPr, cPrsni, freq):
    c = cPr.copy(cPr.data - cPrsn.data)
    pst_(c, 'rainfall_flux', var_name='prrn')
    return pSTAT_cube(c, 'MEAN', *freq)


def mDTR_(cTasmax, cTasmin, freq):
    c = cTasmax.copy(cTasmax.data - cTasmin.data)
    pst_(c, 'diurnal temperature range', var_name='dtr')
    return pSTAT_cube(c, 'MEAN', *freq)


def dHumiWarmDays_(rh, tas, yrs):
    #rh[np.isnan(rh)] = -9999.
    #tas[np.isnan(tas)] = -9999.
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.sum(np.logical_and(_rl(rh[yrs == yr]) > 90,
                                     _rl(tas[yrs == yr]) > K0 + 10))
    return o


def dHumiWarmDays_cube(cHurs, cTas, freq):
    c = cTas.copy(np.logical_and(cHurs.data > 90, cTas.data > K0 + 10))
    c = pSTAT_cube(c, 'COUNT', *freq, function=lambda cell: cell)
    pst_(c, 'humid warm days', 'days')
    return c


def dPr7_(pr, yrs):
    #pr[np.isnan(pr)] = 0.
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.max(rMEAN1d_(_rl(pr[yrs == yr]), 7))
    return o


def dPr7_cube(cPr, freq):
    yr_doy_cube(cPr)
    c = cPr.rolling_window('time', iris.analysis.MEAN, 7)
    c = extract_byAxes_(c, 'time', c.coord('year').points%1 == 0)
    rm_t_aux_cube(c, keep='year')
    c = pSTAT_cube(c, 'MAX', *freq)
    pst_(c, 'max 7-day precipitation', var_name='pr7d')
    return c


def dLongestDryDays_(pr, yrs):
    #pr[np.isnan(pr)] = 0.
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        ts = np.concatenate([[1.,], _rl(pr[yrs == yr])])
        ts = np.split(ts, np.where(ts >= 1. / (24 * 3600))[0])
        ts = [len(x) - 1 for x in ts if len(x) > 0]
        if len(ts) > 0:
            o[i] = np.max(ts)
    return o


def dDryDays_(cPr, freq):
    c = pSTAT_cube(cPr, 'COUNT', *freq,
                   function=lambda cell: cell < 1. / (24 * 3600))
    pst_(c, 'dry days', 'days')
    return c


def dExtrPrDays_(cPr, freq):
    o0 = pSTAT_cube(cPr, 'COUNT', *freq,
                    function=lambda cell: cell > 10. / (24 * 3600))
    o1 = pSTAT_cube(cPr, 'COUNT', *freq,
                    function=lambda cell: cell > 25. / (24 * 3600))
    pst_(o0, 'heavy-precipitaton days', 'days')
    pst_(o1, 'extreme-precipitaton days', 'days')
    return (o0, o1)


def dWarmDays_(cTasmax, freq):
    c = pSTAT_cube(cTasmax, 'COUNT', *freq,
                   function=lambda cell: cell > K0 + 20)
    pst_(c, 'warm days', 'days')
    return c


def dCalmDays_(cSfcWind, freq):
    c = pSTAT_cube(cSfcWind, 'COUNT', *freq,
                   function=lambda cell: cell < 2)
    pst_(c, 'calm days', 'days')
    return c


def dConCalmDays_(sfcw, yrs):
    #t0[np.isnan(t0)] = -9999.
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        ts = np.concatenate([[2,], _rl(sfcw[yrs == yr])])
        ts = np.split(ts, np.where(ts >= 2)[0])
        ts = [len(x) - 1 for x in ts if len(x) > 3]
        if len(ts) > 0:
            o[i] = np.sum(ts)
    return o


def dConWarmDays_(tx, yrs):
    #t0[np.isnan(t0)] = -9999.
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        ts = np.concatenate([[K0 + 20], _rl(tx[yrs == yr])])
        ts = np.split(ts, np.where(ts <= K0 + 20)[0])
        ts = [len(x) - 1 for x in ts if len(x) > 3]
        if len(ts) > 0:
            o[i] = np.sum(ts)
    return o


def dColdDays_(cTasmax, freq):
    c = pSTAT_cube(cTasmax, 'COUNT', *freq,
                   function=lambda cell: cell < K0 - 7)
    pst_(c, 'cold days', 'days')
    return c


def dDegDay20_(tas, yrs):
    t0 = _rl(tas.copy())
    t0 -= K0 + 20
    #t0[np.isnan(t0)] = 0
    t0[t0 < 0] = 0
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.sum(t0[yrs == yr])
    return o


def dDegDay8_vegSeason_(tas, yrs):
    t0 = _rl(tas.copy())
    #t0[np.isnan(t0)] = 0
    t0 = rMEAN1d_(t0, 4, 'same')
    t1 = _rl(tas.copy())
    t1 -= K0 + 8
    #t1[np.isnan(t1)] = 0
    t1[t1 < 0] = 0
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.sum(t1[np.logical_and(yrs == yr, t0 > K0 + 5)])
    return o


def dDegDay17_(tas, yrs):
    t0 = _rl(tas.copy())
    t0 = K0 + 17 - t0
    #t0[np.isnan(t0)] = 0
    t0[t0 < 0] = 0
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.sum(t0[yrs == yr])
    return o


def dFirstDayWithoutFrost_(tn, yrs, doy):
    #tn[np.isnan(tn)] = -9999.
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        ts = _rl(tn[yrs == yr])
        ydoy = doy[yrs == yr]
        try:
            tmp = ydoy[ts >= K0][0]
        except IndexError:
            tmp = np.nan
        o[i] = tmp
    return o


def dEndSpringFrost_(tn, yrs, doy):
    #tn[np.isnan(tn)] = -9999.
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        ind = np.logical_and(yrs == yr, doy<=213)
        ts = _rl(tn[ind])
        ydoy = doy[ind]
        try:
            tmp = ydoy[ts < K0][-1]
        except IndexError:
            tmp = np.nan
        o[i] = tmp
    return o


def dFrostDays_(cTasmin, freq):
    c = pSTAT_cube(cTasmin, 'COUNT', *freq,
                   function=lambda cell: cell < K0)
    pst_(c, 'frost days', 'days')
    return c


def dTropicNights_(cTasmin, freq):
    c = pSTAT_cube(cTasmin, 'COUNT', *freq,
                   function=lambda cell: cell > K0 + 17)
    pst_(c, 'tropic nights', 'days')
    return c


def dZeroCrossingDays_cube(cTasmax, cTasmin, freq):
    ind = np.logical_and(cTasmax.data > K0, cTasmin.data < K0)
    c = pSTAT_cube(cTasmax.copy(ind), 'COUNT', *freq,
                   function=lambda cell: cell)
    pst_(c, 'zero-crossing days', 'days')
    return c


def dZeroCrossingDays_(tx, tn, yrs):
    #tx[np.isnan(tx)] = -9999.
    #tn[np.isnan(tn)] = 9999.
    tt = np.logical_and(_rl(tx) > K0, _rl(tn) < K0)
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.sum(tt[yrs == yr])
    return o


def dStartEndVegSeason_(tas, yrs, doy, thr=5):
    #tas[np.isnan(tas)] = -9999.
    uYrs = np.unique(yrs)
    o0 = np.ma.zeros((len(uYrs),))
    o1 = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        ydoy = doy[yrs == yr]
        ts = rMEAN1d_(_rl(tas[yrs == yr]), 4)
        if np.all(ts < K0 + thr):
            o0[i], o1[i] = ydoy[0], ydoy[0]
        else:
            inds = np.where(ts >= K0 + thr)[0] + 3
            o0[i], o1[i] = ydoy[inds[0]], ydoy[inds[-1]]
    return (o0, o1)


def dWindyDays_(cWsgsmax, freq):
    c = pSTAT_cube(cWsgsmax, 'COUNT', *freq,
                   function=lambda cell: cell > 21)
    pst_(c, 'windy days', 'days')
    return c


def dMarkFreezRain_(pr, ps, tas, t925, t850, t700, q925, q850, q700,
                    pr_min=1.8e-5, t_cold=273.24, t_melt=272.51,
                    rh_melt=89., h_cold=6900):
    import atmos
    aa = _rl(pr) > pr_min
    bb = _rl(tas) <= t_cold
    ta = np.vstack((_rl(t925), _rl(t850), _rl(t700)))
    qa = np.vstack((_rl(q925), _rl(q850), _rl(q700)))
    prl = np.asarray([92500, 85000, 70000]).reshape((3, 1))
    rh = atmos.calculate('RH', qv=qa, T=ta, p=prl)
    cc = ta > t_melt
    dd = rh >= rh_melt
    ee = (_rl(ps) - h_cold - prl) >= 0
    ff = np.any(np.logical_and(cc, dd, ee), axis=0)
    return np.logical_and(aa, bb, ff)


def dFreezRainDays_(pr, ps, tas, t925, t850, t700, q925, q850, q700, yrs,
                    pr_min=1.8e-5, t_cold=273.24, t_melt=272.51,
                    rh_melt=89., h_cold=6900):
    tt = dMarkFreezRain_(pr, ps, tas, t925, t850, t700, q925, q850, q700,
                         pr_min=pr_min, t_cold=t_cold, t_melt=t_melt,
                         rh_melt=rh_melt, h_cold=h_cold)
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.sum(tt[yrs == yr])
    return o
    

def dSncDays_(cSnc, freq, thr=.0):
    c = pSTAT_cube(cSnc, 'COUNT', *freq,
                   function=lambda cell: cell > thr)
    sthr = ' over {:g}%'.format(thr) if thr > 0 else ''
    pst_(c, 'days with snow cover' + sthr, 'days')
    return c


def dSndGT15Days_(cSnd, freq):
    c = pSTAT_cube(cSnd, 'COUNT', *freq,
                   function=lambda cell: cell > .15)
    pst_(c, 'snow-depth-over-15cm days', 'days', 'snd15_')
    return c


def dSndLE10Days_(cSnd, freq):
    c = pSTAT_cube(cSnd, 'COUNT', *freq,
                   function=lambda cell: np.logical_and(cell > 0.,
                                                        cell <= .1))
    pst_(c, 'days with 0-10cm snow-depth', 'days', 'snd_10')
    return c


def dSndGT10LE20Days_(cSnd, freq):
    c = pSTAT_cube(cSnd, 'COUNT', *freq,
                   function=lambda cell: np.logical_and(cell > .1,
                                                        cell <= .2))
    pst_(c, 'days with 10-20cm snow-depth', 'days', 'snd10_20')
    return c


def rho_fr_t_q_p_(t, q, p):
    import atmos
    return  atmos.calculate('rho', T=_rl(t), qv=_rl(q), p=_rl(p))


def dRhoPL_(ta_pl, hus_pl, yrs, pl=92500):
    tt = rho_fr_t_q_p_(ta_pl, hus_pl, pl)
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.mean(tt[yrs == yr])
    return o


def dRhoPS_(tas, huss, ps, yrs):
    tt = rho_fr_t_q_p_(tas, huss, ps)
    uYrs = np.unique(yrs)
    o = np.ma.zeros((len(uYrs),))
    for i, yr in enumerate(uYrs):
        o[i] = np.mean(tt[yrs == yr])
    return o


def dDegDay_(cT, freq, thr=0., left=False):
    c = cT.copy()
    c.data -= thr + K0
    if left:
        c.data *= -1
    c.data[c.data < 0] = 0
    c = pSTAT_cube(c, 'SUM', *freq)
    lg_ = 'lt' if left else 'gt'
    vn_ = 'dd_{:g}' if left else 'dd{:g}_'
    pst_(c, 'degree day {}{:g}'.format(lg_, thr),
         var_name=vn_.format(thr))
    return c


def dRainOnSnow_(cPrrn, cSnc, freq,
                 cSnw=None, thr_r=5., thr_c=25., thr_w=.003, attr=None):
    if cSnw:
        c = cPrrn.copy(np.logical_and(cPrrn.data > thr_r / 3600 / 24,
                                      cSnc.data > thr_c, cSnw.data > thr_w))
    else:
        c = cPrrn.copy(np.logical_and(cPrrn.data > thr_r / 3600 / 24,
                                      cSnc.data > thr_c))
    c = pSTAT_cube(c, 'COUNT', *freq, function=lambda cell: cell)
    pst_(c, 'rain on snow', 'days')
    if attr:
        c.attributes.update(dict(ROS=attr))
    return c


def dPRSN_fr_PR_T_(cPr, cTas, thr=.58):
    data = cPr.data.copy()
    data[cTas.data >= K0 + thr] = 0.
    c = cPr.copy(data)
    pst_(c, 'snowfall_flux', var_name='prsn')
    c.attributes.update(dict(PRSN='PRSN from p & t (.lt. {:g})'.format(thr)))
    return c


def dPRRN_fr_PR_T_(cPr, cTas, thr=.58):
    data = cPr.data.copy()
    data[cTas.data < K0 + thr] = 0.
    c = cPr.copy(data)
    pst_(c, 'rainfall_flux', var_name='prrn')
    c.attributes.update(dict(PRRN='PRRN from p & t (.ge. {:g})'.format(thr)))
    return c


def dColdRainDays_(cPr, cTas, freq, thr=.58):
    ind = np.logical_and(cTas.data <= K0 + 2, cTas.data >= K0 + thr)
    ind0 = np.logical_and(ind, cPr.data > 1. / 24 / 3600)
    ind1 = np.logical_and(ind, cPr.data > 10. / 24 / 3600)
    ind2 = np.logical_and(ind, cPr.data > 20. / 24 / 3600)
    o0 = pSTAT_cube(cPr.copy(ind0), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o0, 'days with cold rain', 'days')
    o0.attributes.update(dict(PRRN='PRRN from p & t (.ge. {:g})'.format(thr)))
    o1 = pSTAT_cube(cPr.copy(ind1), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o1, 'days with cold rain gt 10', 'days')
    o1.attributes.update(dict(PRRN='PRRN from p & t (.ge. {:g})'.format(thr)))
    o2 = pSTAT_cube(cPr.copy(ind2), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o2, 'days with cold rain gt 20', 'days')
    o2.attributes.update(dict(PRRN='PRRN from p & t (.ge. {:g})'.format(thr)))
    return (o0, o1, o2)


def dWarmSnowDays_(cPr, cTas, freq, thr=.58):
    ind = np.logical_and(cTas.data >= K0 - 2, cTas.data < K0 + thr)
    ind0 = np.logical_and(ind, cPr.data > 1. / 24 / 3600)
    ind1 = np.logical_and(ind, cPr.data > 10. / 24 / 3600)
    ind2 = np.logical_and(ind, cPr.data > 20. / 24 / 3600)
    o0 = pSTAT_cube(cPr.copy(ind0), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o0, 'days with warm snow', 'days')
    o0.attributes.update(dict(PRSN='PRSN from p & t (.lt. {:g})'.format(thr)))
    o1 = pSTAT_cube(cPr.copy(ind1), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o1, 'days with warm snow gt 10', 'days')
    o1.attributes.update(dict(PRSN='PRSN from p & t (.lt. {:g})'.format(thr)))
    o2 = pSTAT_cube(cPr.copy(ind2), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o2, 'days with warm snow gt 20', 'days')
    o2.attributes.update(dict(PRSN='PRSN from p & t (.lt. {:g})'.format(thr)))
    return (o0, o1, o2)


def dColdPRRNdays_(cPr, cPrsn, cTas, freq):
    ind = cTas.data <= K0 + 2
    ind0 = np.logical_and(ind, cPr.data - cPrsn.data > 1. / 24 / 3600)
    ind1 = np.logical_and(ind, cPr.data - cPrsn.data > 10. / 24 / 3600)
    ind2 = np.logical_and(ind, cPr.data - cPrsn.data > 20. / 24 / 3600)
    o0 = pSTAT_cube(cPr.copy(ind0), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o0, 'days with cold rain', 'days')
    o1 = pSTAT_cube(cPr.copy(ind1), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o1, 'days with cold rain gt 10', 'days')
    o2 = pSTAT_cube(cPr.copy(ind2), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o2, 'days with cold rain gt 20', 'days')
    return (o0, o1, o2)


def dWarmPRSNdays_(cPrsn, cTas, freq):
    ind = cTas.data >= K0 - 2
    ind0 = np.logical_and(ind, cPrsn.data > 1. / 24 / 3600)
    ind1 = np.logical_and(ind, cPrsn.data > 10. / 24 / 3600)
    ind2 = np.logical_and(ind, cPrsn.data > 20. / 24 / 3600)
    o0 = pSTAT_cube(cPrsn.copy(ind0), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o0, 'days with warm snow', 'days')
    o1 = pSTAT_cube(cPrsn.copy(ind1), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o1, 'days with warm snow gt 10', 'days')
    o2 = pSTAT_cube(cPrsn.copy(ind2), 'COUNT', *freq,
                    function=lambda cell: cell)
    pst_(o2, 'days with warm snow gt 20', 'days')
    return (o0, o1, o2)

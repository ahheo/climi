from climi.uuuu import *
from climi.climidx import *

import numpy as np
import iris
import iris.coord_categorisation as cat
import os
import yaml
import time
import warnings
import logging
import argparse

from time import localtime, strftime


_here_ = get_path_(__file__)


#INDEX DICT: FORMAT:
#    NAME: (#, TR_i, VARIABLES, FUNCTION, TR_o, GROUPS)
#where TR_i, TR_o are input and output temporal resolution.
i__ = {
    'ET': (1, 0, ['et'],
        None, ['year'], 'p'),
    'HumiWarmDays': (2, 1, ['hurs', 't'],
        dHumiWarmDays_cube, ['season'], 'ht'),
    'EffPR': (3, 0, ['et', 'pr'],
        None, ['season', 'year'], 'p'),
    'PR7Dmax': (4, 1, ['pr'],
        dPr7_, ['year'], 'p'),
    'LnstDryDays': (5, 1, ['pr'],
        dLongestDryDays_, ['season'], 'p'),
    #'DryDays': (6, 1, ['pr'],
    #    dDryDays_, ['month'], 'p'),
    'DryDays': (6, 1, ['pr'],
        dDryDays_, ['year'], 'p'),
    'PR': (7, 0, ['pr'],
        None, ['season', 'year'], 'p'),
    #'PR': (7, 0, ['pr'],
    #    None, ['month'], 'p'),
    'PRmax': (8, 1, ['pr'],
        'MAX', ['month', 'year'], 'p'),
    #'PRmax': (8, 1, ['pr'],
    #    'MAX', ['month'], 'p'),
    'PRRN': (9, 0, ['pr', 'prsn'],
        None, ['season', 'year'], 'p'),
    'PRSN': (10, 0, ['pr', 'prsn'],
        None, ['season', 'year'], 'p'),
    'NetRO': (11, 0, ['ro'],
        None, ['year'], 'p'),
    'SD': (12, 0, ['sd'],
        None, ['year'], 's'),
    'RLDS': (13, 0, ['rl'],
        None, ['season'], 's'),
    'RSDS': (14, 0, ['rs'],
        None, ['season'], 's'),
    'CoolingDegDay': (15, 1, ['tx'],
        dDegDay_, ['month', 'year'], 't'),
    'ConWarmDays': (16, 1, ['tx'],
        dConWarmDays_, ['year'], 'tc'),
    'TX': (17, 0, ['tx'],
        None, ['year'], 't'),
    #'TX': (17, 0, ['tx'],
    #    None, ['season', 'year'], 't'),
    #'TX': (17, 0, ['tx'],
    #    None, ['month'], 't'),
    'WarmDays': (18, 1, ['tx'],
        dWarmDays_, ['season', 'year'], 't'),
    'ColdDays': (19, 1, ['tx'],
        dColdDays_, ['season', 'year'], 't'),
    'DegDay20': (20, 1, ['t'],
        dDegDay_, ['year'], 't'),
    'DegDay8': (21, 1, ['t'],
        dDegDay8_vegSeason_, ['year'], 't'),
    'DegDay17': (22, 1, ['t'],
        dDegDay_, ['year'], 't'),
    'TN': (23, 0, ['tn'],
        None, ['year'], 't'),
    #'TN': (23, 0, ['tn'],
    #    None, ['season', 'year'], 't'),
    #'TN': (23, 0, ['tn'],
    #    None, ['month'], 't'),
    'SpringFrostDayEnd': (24, 1, ['tn'],
        dEndSpringFrost_, ['year'], 't'),
    'FrostDays': (25, 1, ['tn'],
        dFrostDays_, ['season', 'year'], 't'),
    'TropicNights': (26, 1, ['tn'],
        dTropicNights_, ['year'], 't'),
    #'ZeroCrossingDays': (27, 1, ['tx', 'tn'],
    #    dZeroCrossingDays_cube, ['year'], 't'),
    'ZeroCrossingDays': (27, 1, ['tx', 'tn'],
        dZeroCrossingDays_cube, ['season'], 't'), # SVT_ERIK
    'VegSeasonDayEnd-5': (28, 1, ['t'],
        dStartEndVegSeason_, ['year'], 't'),
    'VegSeasonDayEnd-2': (29, 1, ['t'],
        dStartEndVegSeason_, ['year'], 't'),
    'VegSeasonDayStart-5': (30, 1, ['t'],
        dStartEndVegSeason_, ['year'], 't'),
    'VegSeasonDayStart-2': (31, 1, ['t'],
        dStartEndVegSeason_, ['year'], 't'),
    'VegSeasonLength-5': (32, 1, ['t'],
        dStartEndVegSeason_, ['year'], 't'),
    'VegSeasonLength-2': (33, 1, ['t'],
        dStartEndVegSeason_, ['year'], 't'),
    'SfcWind': (34, 1, ['ws', 'us', 'vs'],
        None, ['month', 'season', 'year'], 'w'),
    #'SfcWind': (34, 1, ['ws', 'us', 'vs'],
    #    None, ['month'], 'w'),
    'WindGustMax': (35, 1, ['wsgs'],
        'MAX', ['year'], 'w'),
    'WindyDays': (36, 1, ['wsgs'],
        dWindyDays_, ['season', 'year'], 'w'),
    'PRgt10Days': (37, 1, ['pr'],
        dExtrPrDays_, ['season', 'year'], 'p'),
    'PRgt25Days': (38, 1, ['pr'],
        dExtrPrDays_, ['season', 'year'], 'p'),
    'SncDays': (39, 1, ['snc'],
        dSncDays_, ['year'], 'p'),
    'Snd10Days': (40, 1, ['snd'],
        dSndGT10LE20Days_, ['year'], 'p'),
    'Snd20Days': (41, 1, ['snd'],
        dSndGT10LE20Days_, ['year'], 'p'),
    'SNWmax': (42, 1, ['snw'],
        'MAX', ['year'], 'p'),
    'TAS': (43, 0, ['t'],
        None, ['year'], 't'),
    #'TAS': (43, 0, ['t'],
    #    None, ['season', 'year'], 't'),
    #'TAS': (43, 0, ['t'],
    #    None, ['month'], 't'),
    'DTR': (44, 0, ['tx', 'tn'],
        mDTR_, ['month'], 't'),
    'Rho925': (45, 1, ['t925', 'hus925', 'ps'],
        None, ['month'], 'w'),
    'RhoS': (46, 1, ['t', 'huss', 'ps'],
        None, ['month'], 'w'),
    'SuperCooledPR': (47, 1, ['t', 'pr', 'ps', 't925', 't850', 't700',
                              'hus925', 'hus850', 'hus700'],
        dFreezRainDays_, ['year'], 'p'),
    'Snc25Days': (48, 1, ['snc'],
        dSncDays_, ['year'], 'p'),
    'R5OScw': (49, 1, ['pr', 't', 'snc', 'snw'],
        dRainOnSnow_, ['year'], 'p'),
    'R1OScw': (50, 1, ['pr', 't', 'snc', 'snw'],
        dRainOnSnow_, ['year'], 'p'),
    'R5OSc': (51, 1, ['pr', 't', 'snc'],
        dRainOnSnow_, ['year'], 'p'),
    'R1OSc': (52, 1, ['pr', 't', 'snc'],
        dRainOnSnow_, ['year'], 'p'),
    'PRSNmax': (53, 1, ['prsn'],
        'MAX', ['year'], 'p'),
    'CalmDays': (54, 1, ['ws', 'us', 'vs'],
        dCalmDays_, ['season', 'year'], 'w'),
    'ConCalmDays': (55, 1, ['ws', 'us', 'vs'],
        dConCalmDays_, ['year'], 'wc'),
    'Wind975': (56, 1, ['u975', 'v975', 'ps'],
        None, ['month', 'season', 'year'], 'w'),
    'Wind975toSfc': (57, 1, ['u975', 'v975', 'ws', 'ps', 'us', 'vs'],
        None, ['season', 'year'], 'w'),
    'ColdRainDays': (58, 1, ['pr', 't'],
        dColdRainDays_, ['year'], 'p'),
    'ColdRainGT10Days': (59, 1, ['pr', 't'],
        dColdRainDays_, ['year'], 'p'),
    'ColdRainGT20Days': (60, 1, ['pr', 't'],
        dColdRainDays_, ['year'], 'p'),
    'WarmSnowDays': (61, 1, ['pr', 't'],
        dWarmSnowDays_, ['year'], 'p'),
    'WarmSnowGT10Days': (62, 1, ['pr', 't'],
        dWarmSnowDays_, ['year'], 'p'),
    'WarmSnowGT20Days': (63, 1, ['pr', 't'],
        dWarmSnowDays_, ['year'], 'p'),
    'ColdPRRNdays': (64, 1, ['pr', 't', 'prsn'],
        dColdPRRNdays_, ['year'], 'p'),
    'ColdPRRNgt10Days': (65, 1, ['pr', 't', 'prsn'],
        dColdPRRNdays_, ['year'], 'p'),
    'ColdPRRNgt20Days': (66, 1, ['pr', 't', 'prsn'],
        dColdPRRNdays_, ['year'], 'p'),
    'WarmPRSNdays': (67, 1, ['t', 'prsn'],
        dWarmPRSNdays_, ['year'], 'p'),
    'WarmPRSNgt10Days': (68, 1, ['t', 'prsn'],
        dWarmPRSNdays_, ['year'], 'p'),
    'WarmPRSNgt20Days': (69, 1, ['t', 'prsn'],
        dWarmPRSNdays_, ['year'], 'p'),
    'SST': (70, 0, ['sst'],
        None, ['season', 'year'], 't'),
    'SIC': (71, 0, ['sic'],
        None, ['season', 'year'], 'ti'),
    'Rho975': (72, 1, ['t975', 'hus975', 'ps'],
        None, ['month'], 'w'),
    'h6SuperCooledPR': (73, 2, ['t', 'pr', 'ps', 't925', 't850', 't700',
                                'hus925', 'hus850', 'hus700'],
        dFreezRainDays_, ['year'], 'p'),
    'Wind925': (74, 1, ['u925', 'v925', 'ps'],
        None, ['month', 'season', 'year'], 'w'),
    'Wind925toSfc': (75, 1, ['u925', 'v925', 'ws', 'ps', 'us', 'vs'],
        None, ['season', 'year'], 'w'),
    'FirstDayWithoutFrost': (76, 1, ['tn'],
        dFirstDayWithoutFrost_, ['year'], 't'),
    'CalmDays925': (77, 1, ['u925', 'v925', 'ps'],
        dCalmDays_, ['year'], 'w'),
    'ConCalmDays925': (78, 1, ['u925', 'v925', 'ps'],
        dConCalmDays_, ['year'], 'wc'),
    'CalmDays975': (79, 1, ['u975', 'v975', 'ps'],
        dCalmDays_, ['year'], 'w'),
    'ConCalmDays975': (80, 1, ['u975', 'v975', 'ps'],
        dConCalmDays_, ['year'], 'wc'),
    'MinusDays': (81, 1, ['t'],
        dMinusDays_, ['year'], 't'),
    'FreezingDays': (82, 1, ['tx'],
        dFreezingDays_, ['year'], 't'),
    'ColdRainWarmSnowDays': (83, 1, ['pr', 't'],
        dColdRainWarmSnowDays_, ['year'], 'p'),
    'Wind50m': (84, 1, ['u50m', 'v50m'],
        None, ['month', 'season', 'year'], 'w'),
    'Rho50m': (85, 1, ['t50m', 'hus50m', 'p50m'],
        None, ['month'], 'w'),
    'Wind100m': (86, 1, ['u100m', 'v100m'],
        None, ['month', 'season', 'year'], 'w'),
    'Rho100m': (87, 1, ['t100m', 'hus100m', 'p100m'],
        None, ['month'], 'w'),
    'Wind200m': (88, 1, ['u200m', 'v200m'],
        None, ['month', 'season', 'year'], 'w'),
    'Rho200m': (89, 1, ['t200m', 'hus200m', 'p200m'],
        None, ['month'], 'w'),
    'Wind950': (90, 1, ['u950', 'v950'],
        None, ['month', 'season', 'year'], 'w'),
    'Rho950': (91, 1, ['t950', 'hus950', 'ps'],
        None, ['month'], 'w'),
    'Wind950toSfc': (92, 1, ['u950', 'v950', 'ws', 'ps', 'us', 'vs'],
        None, ['season', 'year'], 'w'),
    'CalmDays50m': (93, 1, ['u50m', 'v50m'],
        dCalmDays_, ['year'], 'w'),
    'ConCalmDays50m': (94, 1, ['u50m', 'v50m'],
        dConCalmDays_, ['year'], 'wc'),
    'CalmDays100m': (95, 1, ['u100m', 'v100m'],
        dCalmDays_, ['year'], 'w'),
    'ConCalmDays100m': (96, 1, ['u100m', 'v100m'],
        dConCalmDays_, ['year'], 'wc'),
    'CalmDays200m': (97, 1, ['u200m', 'v200m'],
        dCalmDays_, ['year'], 'w'),
    'ConCalmDays200m': (98, 1, ['u200m', 'v200m'],
        dConCalmDays_, ['year'], 'wc'),
    'CalmDays950': (99, 1, ['u950', 'v950', 'ps'],
        dCalmDays_, ['year'], 'w'),
    'ConCalmDays950': (100, 1, ['u950', 'v950'],
        dConCalmDays_, ['year'], 'wc'),
    'h3SfcWind': (101, 3, ['ws', 'us', 'vs'],
        None, ['hour-month', 'hour-season', 'hour'], 'w'),
    'h3RhoS': (102, 3, ['t', 'huss', 'ps'],
        None, ['hour-month'], 'w'),
    'h3Wind975': (103, 3, ['u975', 'v975'],
        None, ['hour-month', 'hour-season', 'hour'], 'w'),
    'h3Rho975': (104, 3, ['t925', 'hus925', 'ps'],
        None, ['hour-month'], 'w'),
    'h3Wind950': (105, 3, ['u950', 'v950'],
        None, ['hour-month', 'hour-season', 'hour'], 'w'),
    'h3Rho950': (106, 3, ['t950', 'hus950', 'ps'],
        None, ['hour-month'], 'w'),
    'h3Wind925': (107, 3, ['u925', 'v925'],
        None, ['hour-month', 'hour-season', 'hour'], 'w'),
    'h3Rho925': (108, 3, ['t925', 'hus925', 'ps'],
        None, ['hour-month'], 'w'),
    'h3Wind50m': (109, 3, ['u50m', 'v50m'],
        None, ['hour-month', 'hour-season', 'hour'], 'w'),
    'h3Rho50m': (110, 3, ['t50m', 'hus50m', 'p50m'],
        None, ['hour-month'], 'w'),
    'h3Wind100m': (111, 3, ['u100m', 'v100m'],
        None, ['hour-month', 'hour-season', 'hour'], 'w'),
    'h3Rho100m': (112, 3, ['t100m', 'hus100m', 'p100m'],
        None, ['hour-month'], 'w'),
    'h3Wind200m': (113, 3, ['u200m', 'v200m'],
        None, ['hour-month', 'hour-season', 'hour'], 'w'),
    'h3Rho200m': (114, 3, ['t200m', 'hus200m', 'p200m'],
        None, ['hour-month'], 'w')
    }


var_ = {
    'et': 'evspsbl',
    'hurs': 'hurs',
    'huss': 'huss',
    'hus975': 'hus975',
    'hus950': 'hus950',
    'hus925': 'hus925',
    'hus850': 'hus850',
    'hus700': 'hus700',
    'hus50m': 'hus50m',
    'hus100m': 'hus100m',
    'hus200m': 'hus200m',
    'pr': 'pr',
    'prsn': 'prsn',
    'ps': 'ps',
    'p50m': 'p50m',
    'p100m': 'p100m',
    'p200m': 'p200m',
    'rl': 'rlds',
    'rs': 'rsds',
    'ro': 'mrro',
    'sd': 'sund',
    'snc': 'snc',
    'snd': 'snd',
    'snw': 'snw',
    'sst': 'tos',
    'sic': 'sic',
    't': 'tas',
    'tx': 'tasmax',
    'tn': 'tasmin',
    't975': 'ta975',
    't950': 'ta950',
    't925': 'ta925',
    't850': 'ta850',
    't700': 'ta700',
    't50m': 'ta50m',
    't100m': 'ta100m',
    't200m': 'ta200m',
    'us': 'uas',
    'u975': 'ua975',
    'u950': 'ua950',
    'u925': 'ua925',
    'u50m': 'ua50m',
    'u100m': 'ua100m',
    'u200m': 'ua200m',
    'vs': 'vas',
    'v975': 'va975',
    'v950': 'va950',
    'v925': 'va925',
    'v50m': 'va50m',
    'v100m': 'va100m',
    'v200m': 'va200m',
    'ws': 'sfcWind',
    'wsgs': 'wsgsmax'
    }


def _vv2(a, xvn=9):
    a = sorted(a, key=lambda x:(len(x[1]), sorted(x[1][0])), reverse=True)
    while len(a) > 0:
        aa, bb = [a[0][0]], a[0][1]
        del(a[0])
        if len(a) > 0:
            for i in a.copy():
                tmp = ouniqL_(bb + i[1])
                if len(tmp) <= xvn:
                    aa.append(i[0])
                    bb = tmp
                    a.remove(i)
        yield (aa, bb)


def _vv(ii_, tint, subg=None, xvn=9):
    if tint == 'mon':
        nn = 0
    elif tint == 'day':
        nn = 1
    elif tint == '6hr':
        nn = 2
    elif tint == '3hr':
        nn = 3
    elif tint == '1hr':
        nn = 4
    if nn > 0 and subg is None:
        tmp = [(i, i__[i][2]) for i in ii_ if i__[i][1] == nn]
        return list(_vv2(tmp, xvn))
    else:
        tmp = [i__[i][2] for i in ii_ if i__[i][1] == nn]
        tmp_ = set(flt_l(tmp))
        if nn > 0 and len(tmp_) > xvn:
            if subg == 'v':
                tmp__ = ss_fr_sl_(list(map(set, tmp)))
                return [([ii for ii in ii_
                          if all([iii in i for iii in i__[ii][2]])], i)
                        for i in tmp__]
            elif subg == 'i':
                ii__ = [ii_[:(len(ii_)//2)], ii_[(len(ii_)//2):]]
                tmp__ = [_vv(i, tint) for i in ii__]
                return nli_(tmp__)
        else:
            return ([i for i in ii_ if i__[i][1] == nn], tmp_)


def _vd_fr_vv(vl):
    if isinstance(vl, tuple):
        vd = dict()
        for i in vl[1]:
            vd.update({'c_' + i: var_[i]})
        return (vl[0], vd)
    elif isinstance(vl, list):
        return [_vd_fr_vv(i) for i in vl]
    else:
        raise Exception('check input!')


def _vd(ii_, tint, subg=None, xvn=9):
    return _vd_fr_vv(_vv(ii_, tint, subg, xvn))


def _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, folder='cordex'):
    t1 = l__('>>>{}'.format(tint))
    ka0 = dict(dn=dn, gwl=gwl, po_=po_)
    if tint == 'mon':
        f__ = _mclimidx
    elif tint == 'day':
        f__ = _dclimidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    elif tint == '6hr':
        f__ = _h6climidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    elif tint == '3hr':
        f__ = _h3climidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    elif tint == '1hr':
        f__ = _h1climidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    def __xyz(vd):
        ccc = dict()
        if isinstance(vd, tuple):
            t000 = l__("rf__(): {} variables".format(len(vd[1].keys())))
            ll_(', '.join(vd[1].keys()))
            ll_(', '.join(vd[0]))
            tmp, tmp_, tmp__ = [], [], []
            wsyes = False # SURFACE WIND SPEED
            for kk in vd[1].keys():
                if wsyes and kk in ('c_us', 'c_vs'): # SURFACE WIND SPEED
                    continue # SURFACE WIND SPEED
                if y0y1:
                    cc, ee = rf__(pi_, tint, var=vd[1][kk], period=y0y1,
                                  reg_d=reg_d, folder=folder)
                else:
                    cc, ee = rf__(pi_, tint, var=vd[1][kk], reg_d=reg_d,
                                  folder=folder)
                if cc:
                    tmp.append(kk)
                    if kk == 'c_ws': # SURFACE WIND SPEED
                        wsyes = True # SURFACE WIND SPEED
                if ee == 'cce':
                    tmp_.append(kk)
                elif ee == 'yye':
                    tmp__.append(kk)
                ccc.update({kk: cc})
            if len(tmp__) > 0:
                ll_('YYE: {}'.format(', '.join(tmp__)))
            if len(tmp_) > 0:
                ll_('CCE: {}'.format(', '.join(tmp_)))
            if len(tmp) > 0:
                ll_(', '.join(tmp))
            ll_("rf__()", t000)
            f__(il_=vd[0], **ka0, **ccc)
        elif isinstance(vd, list):
            for i in vd:
                __xyz(i)
        else:
            raise Exception("check out put of '_vd'")
    vD = _vd(il_, tint)
    __xyz(vD)
    ll_('<<<{}'.format(tint), t1)


s4 = ('djf', 'mam', 'jja', 'son')


def _sv(v_, o, dgpi, freq=None):
    fn = '{}{}.nc'
    dn, gwl, po_ = dgpi[:3]
    freq = freq if freq else i__[v_][4]
    if len(freq) == 1:
        cubesv_(o, fn.format(po_, '_'.join((v_, dn, freq[0], gwl))))
    else:
        for oo, ff in zip(o, freq):
            cubesv_(oo, fn.format(po_, '_'.join((v_, dn, ff, gwl))))


def _dd(v_, cube, dgpi, freq=None):
    freq = freq if freq else i__[v_][4]
    if v_ in dgpi[-1] and cube:
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                       *freq)
        _sv(v_, o, dgpi, freq=freq)
        ll_(v_, t000)


def _d0(v_, cube, dgpi, fA_=(), fK_={}, pK_=None, freq=None):
    if isIter_(v_):
        vk_ = v_[0]
        lmsg = '/'.join(('{}',) * len(v_)) + ' {}'
        vids = [i__[i][0] for i in v_]
        lmsg_ = lmsg.format(*vids, v_[0][:8])
    else:
        vk_ = v_
        lmsg_ = '{} {}'.format(i__[v_][0], v_)
    t000 = l__(lmsg_)
    freq = freq if freq else i__[vk_][4]
    cc = cube if isMyIter_(cube) else (cube,)
    o = i__[vk_][3](*cc, freq, *fA_, **fK_)
    if pK_:
        pst_(o, **pK_)
    if not isIter_(v_) or len(v_) == 1:
        _sv(vk_, o, dgpi, freq=freq)
    else:
        for i, ii in zip(v_, o):
            _sv(i, ii, dgpi, freq=freq)
    ll_(lmsg_, t000)


def _tt(cube, y0y1=None):
    rm_t_aux_cube(cube)
    yr_doy_cube(cube)
    cat.add_season(cube, 'time', name='season', seasons=s4)
    tyrs, tdoy = cube.coord('year').points, cube.coord('doy').points
    y_y_ = y0y1 if y0y1 else tyrs[[0, -1]]
    tsss = cube.coord('season').points
    seasonyr_cube(cube, s4)
    tsyr = cube.coord('seasonyr').points
    ax_t = cube.coord_dims('time')[0]
    return (ax_t, y_y_, tyrs, tdoy, tsss, tsyr)


def _d1(v_, cube, dgpi, ax_t, y_y_,
        cK_={}, fA_=(), fK_={}, freq=None, out=False):
    def _inic(icK_):
        return initAnnualCube_(cube[0] if isMyIter_(cube) else cube,
                               y_y_, **icK_)
    if isIter_(v_):
        vk_ = v_[0]
        lmsg = '/'.join(('{}',) * len(v_)) + ' {}'
        vids = [i__[i][0] for i in v_]
        lmsg_ = lmsg.format(*vids, v_[0][:8])
    else:
        vk_ = v_
        lmsg_ = '{} {}'.format(i__[v_][0], v_)
    if ((freq and len(freq) != 1) or
        (freq is None and len(i__[vk_][4]) != 1)):
        raise Exception("exec-freq more than 1 currently not available!")
    t000 = l__(lmsg_)
    if isinstance(cK_, dict):
        o = _inic(cK_)
    else:
        o = iris.cube.CubeList([_inic(i) for i in cK_])
    o_ = _afm_n(cube, ax_t, i__[vk_][3], o, *fA_, **fK_)
    o = o_ if o_ else o
    if not isIter_(v_) or len(v_) == 1:
        _sv(vk_, o, dgpi, freq=freq)
    else:
        for i, ii in zip(v_, o):
            _sv(i, ii, dgpi, freq=freq)
    ll_(lmsg_, t000)
    if out:
        return o


def _mclimidx(dn=None, gwl=None, po_=None, il_=None,
              c_pr=None, c_et=None, c_prsn=None, c_ro=None, c_sd=None,
              c_rs=None, c_rl=None, c_t=None, c_tx=None, c_tn=None,
              c_sst=None, c_sic=None):

    dgpi = [dn, gwl, po_, il_]
    if any([i is None for i in dgpi]):
        raise ValueError("input of 'dn', 'gwl', 'po_', 'il_' is mandotory!")

    os.makedirs(po_, exist_ok=True)

    def _mm(v_, cube, freq=None):
        if v_ in il_ and cube:
            o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                           *i__[v_][4])
            _sv(v_, o, dgpi, freq=freq)

    _mm('PR', c_pr)                                                         #PR
    _mm('ET', c_et)                                                         #ET
    if 'EffPR' in il_ and c_pr is not None and c_et is not None:
        o = c_pr.copy(c_pr.data - c_et.data)
        pst_(o, 'effective precipitation', var_name='eff_pr')
        _mm('EffPR', o)                                                  #EffPR
    _mm('PRSN', c_prsn)                                                   #PRSN
    if 'PRRN' in il_ and c_pr is not None and c_prsn is not None:
        o = c_pr.copy(c_pr.data - c_prsn.data)
        pst_(o, 'rainfall_flux', var_name='prrn')
        _mm('PRRN', o)                                                    #PRRN
    v_ = 'NetRO'
    if 'NetRO' in il_ and c_ro is not None:
        o = extract_season_cube(c_ro, 'amjjas')
        o.attributes.update({'Season': 'amjjas'})
        _mm('NetRO', o)                                                  #NetRO
    _mm('SD', c_sd)                                                         #SD
    _mm('RSDS', c_rs)                                                     #RSDS
    _mm('RLDS', c_rl)                                                     #RLDS
    _mm('TAS', c_t)                                                        #TAS
    _mm('TX', c_tx)                                                         #TX
    _mm('TN', c_tn)                                                         #TN
    v_ = 'DTR'
    if v_ in il_ and c_tx is not None and c_tn is not None:
        _d0(v_, (c_tx, c_tn), dgpi)                                        #DTR
    _mm('SST', c_sst)                                                      #SST
    _mm('SIC', c_sic)                                                      #SIC


def _dclimidx(dn=None, gwl=None, po_=None, il_=None, y0y1=None,
              c_hurs=None, c_huss=None, c_hus975=None, c_hus950=None,
              c_hus925=None, c_hus850=None, c_hus700=None, c_hus50m=None,
              c_hus100m=None, c_hus200m=None, c_pr=None, c_prsn=None,
              c_ps=None, c_p50m=None, c_p100m=None, c_p200m=None,
              c_snc=None, c_snd=None, c_snw=None, c_t=None,
              c_tx=None, c_tn=None, c_t975=None, c_t950=None,
              c_t925=None, c_t850=None, c_t700=None, c_t50m=None,
              c_t100m=None, c_t200m=None, c_u975=None, c_u950=None,
              c_u925=None, c_u50m=None, c_u100m=None, c_u200m=None,
              c_us=None, c_v975=None, c_v950=None, c_v925=None,
              c_v50m=None, c_v100m=None, c_v200m=None, c_vs=None,
              c_ws=None, c_wsgs=None):

    dgpi = [dn, gwl, po_, il_]
    if any([i is None for i in dgpi]):
        raise ValueError("input of 'dn', 'gwl', 'po_', 'il_' is mandotory!")

    os.makedirs(po_, exist_ok=True)

    if c_wsgs is not None:
        _dd('WindGustMax', c_wsgs, dgpi)                           #WindGustMax
        v_ = 'WindyDays'
        if v_ in il_:
            _d0(v_, c_wsgs, dgpi)                                    #WindyDays
    if (c_ws is None and c_us and c_vs and
        any([i in il_ for i in ['sfcWind', 'CalmDays', 'ConCalmDays',
                                'Wind975toSfc', 'Wind925toSfc',
                                'Wind950toSfc']])):
        #c_ws = c_us.copy(np.sqrt(c_us.data**2 + c_vs.data**2))
        c_ws = ws_cube(c_us, c_vs)
        pst_(c_ws, 'surface wind speed', var_name='sfcWind')
    o = None
    if any([i in il_ for i in ['Wind975', 'Wind975toSfc', 'CalmDays975',
                               'ConCalmDays975']]):
        if c_u975 and c_v975:
            #o = c_u975.copy(np.sqrt(c_u975.data**2 + c_v975.data**2))
            o = ws_cube(c_u975, c_v975)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 97500.)
            pst_(o, 'wind speed at 975 mb', var_name='w975')
    _dd('Wind975', o, dgpi)                                            #Wind975
    if o:
        v_ = 'CalmDays975'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays975
        v_ = 'ConCalmDays975'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)      #ConCalmDays975
    if 'Wind975toSfc' in  il_ and o and c_ws:
        o = o.copy(o.data / c_ws.data)
        pst_(o, 'wsr_975_to_sfc', '1')
        _dd('Wind975toSfc', o, dgpi)                              #Wind975toSfc
    o = None
    if any([i in il_ for i in ['Wind950', 'Wind950toSfc', 'CalmDays950',
                               'ConCalmDays950']]):
        if c_u950 and c_v950:
            #o = c_u950.copy(np.sqrt(c_u950.data**2 + c_v950.data**2))
            o = ws_cube(c_u950, c_v950)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 95000.)
            pst_(o, 'wind speed at 950 mb', var_name='w950')
    _dd('Wind950', o, dgpi)                                            #Wind950
    if o:
        v_ = 'CalmDays950'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays950
        v_ = 'ConCalmDays950'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)      #ConCalmDays950
    if 'Wind950toSfc' in  il_ and o and c_ws:
        o = o.copy(o.data / c_ws.data)
        pst_(o, 'wsr_950_to_sfc', '1')
        _dd('Wind950toSfc', o, dgpi)                              #Wind950toSfc
    o = None
    if any([i in il_ for i in ['Wind925', 'Wind925toSfc', 'CalmDays925',
                               'ConCalmDays925']]):
        if c_u925 and c_v925:
            #o = c_u925.copy(np.sqrt(c_u925.data**2 + c_v925.data**2))
            o = ws_cube(c_u925, c_v925)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 92500.)
            pst_(o, 'wind speed at 925 mb', var_name='w925')
    _dd('Wind925', o, dgpi)                                            #Wind925
    if o:
        v_ = 'CalmDays925'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays925
        v_ = 'ConCalmDays925'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)      #ConCalmDays925
    if 'Wind925toSfc' in  il_ and o and c_ws:
        o = o.copy(o.data / c_ws.data)
        pst_(o, 'wsr_925_to_sfc', '1')
        _dd('Wind925toSfc', o, dgpi)                              #Wind925toSfc
    o = None
    if any([i in il_ for i in ['Wind50m', 'CalmDays50m', 'ConCalmDays50m']]):
        if c_u50m and c_v50m:
            #o = c_u50m.copy(np.sqrt(c_u50m.data**2 + c_v50m.data**2))
            o = ws_cube(c_u50m, c_v50m)
            pst_(o, 'wind speed at 50m', var_name='w50m')
    _dd('Wind50m', o, dgpi)                                            #Wind50m
    if o:
        v_ = 'CalmDays50m'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays50m
        v_ = 'ConCalmDays50m'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)      #ConCalmDays50m
    o = None
    if any([i in il_ for i in ['Wind100m', 'CalmDays100m', 'ConCalmDays100m']]):
        if c_u100m and c_v100m:
            #o = c_u100m.copy(np.sqrt(c_u100m.data**2 + c_v100m.data**2))
            o = ws_cube(c_100m, c_v100m)
            pst_(o, 'wind speed at 100m', var_name='w100m')
    _dd('Wind100m', o, dgpi)                                          #Wind100m
    if o:
        v_ = 'CalmDays100m'
        if v_ in il_:
            _d0(v_, o, dgpi)                                      #CalmDays100m
        v_ = 'ConCalmDays100m'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)     #ConCalmDays100m
    o = None
    if any([i in il_ for i in ['Wind200m', 'CalmDays200m', 'ConCalmDays200m']]):
        if c_u200m and c_v200m:
            #o = c_u200m.copy(np.sqrt(c_u200m.data**2 + c_v200m.data**2))
            o = ws_cube(c_200m, c_v200m)
            pst_(o, 'wind speed at 200m', var_name='w200m')
    _dd('Wind200m', o, dgpi)                                          #Wind200m
    if o:
        v_ = 'CalmDays200m'
        if v_ in il_:
            _d0(v_, o, dgpi)                                      #CalmDays200m
        v_ = 'ConCalmDays200m'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)     #ConCalmDays200m
    _dd('SfcWind', c_ws, dgpi)                                         #SfcWind
    if c_ws:
        v_ = 'CalmDays'
        if v_ in il_:
            _d0(v_, c_ws, dgpi)                                       #CalmDays
        v_ = 'ConCalmDays'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_ws, y0y1)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, c_ws, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)      #ConCalmDays
    if c_tx:
        v_ = 'WarmDays'
        if v_ in il_:
            _d0(v_, c_tx, dgpi)                                       #WarmDays
        v_ = 'ColdDays'
        if v_ in il_:
            _d0(v_, c_tx, dgpi)                                       #ColdDays
        v_ = 'FreezingDays'
        if v_ in il_:
            _d0(v_, c_tx, dgpi)                                   #FreezingDays
        v_ = 'CoolingDegDay'
        if v_ in il_:
            _d0(v_, c_tx, dgpi, fK_=dict(thr=20),
                pK_=dict(name='degree day cooling', var_name='dd20x_'))
                                                                 #CoolingDegDay
        v_ = 'ConWarmDays'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_tx, y0y1)
            cK_ = dict(name='continue warm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, c_tx, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)      #ConWarmDays
    if c_tn:
        v_ = 'FrostDays'
        if v_ in il_:
            _d0(v_, c_tn, dgpi)                                      #FrostDays
        v_ = 'TropicNights'
        if v_ in il_:
            _d0(v_, c_tn, dgpi)                                    #TropicNight
        if any([i in il_ for i in ['SpringFrostDayEnd',
                                   'FirstDayWithoutFrost']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_tn, y0y1)
        v_ = 'SpringFrostDayEnd'
        if v_ in il_:
            cK_ = dict(name='spring frost end-day', units=1,
                       var_name='frost_end')
            fA_ = (tyrs, tdoy)
            _d1(v_, c_tn, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)
                                                             #SpringFrostDayEnd
        v_ = 'FirstDayWithoutFrost'
        if v_ in il_:
            cK_ = dict(name='first day without frost', units=1,
                       var_name='day1nofrost')
            fA_ = (tyrs, tdoy)
            _d1(v_, c_tn, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)
                                                          #FirstDayWithoutFrost
    v_ = 'ZeroCrossingDays'
    if v_ in il_ and c_tx and c_tn:
        _d0(v_, (c_tx, c_tn), dgpi)                           #ZeroCrossingDays
    if c_t:
        v_ = 'MinusDays'
        if v_ in il_:
            _d0(v_, c_t, dgpi)                                       #MinusDays
        v_ = 'DegDay20'
        if v_ in il_:
            _d0(v_, c_t, dgpi, fK_=dict(thr=20))                      #DegDay20
        v_ = 'DegDay17'
        if v_ in il_:
            _d0(v_, c_t, dgpi, fK_=dict(thr=17, left=True))           #DegDay17
        if any([i in il_ for i in ['DegDay8',
                                   'VegSeasonDayStart-5',
                                   'VegSeasonDayEnd-5', 'VegSeasonLength-5',
                                   'VegSeasonDayStart-2',
                                   'VegSeasonDayEnd-2', 'VegSeasonLength-2',
                                   'SuperCooledPR']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_t, y0y1)
        v_ = 'DegDay8'
        if v_ in il_:
            cK_ = dict(name='degree day g8 vegetation season',
                       var_name='dd8_')
            fA_ = (tyrs,)
            _d1(v_, c_t, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)           #DegDay8
        if any([i in il_ for i in ['VegSeasonDayStart-5', 'VegSeasonDayEnd-5',
                                   'VegSeasonLength-5']]):
            o = _d1(['VegSeasonDayStart-5', 'VegSeasonDayEnd-5'],
                    c_t, dgpi, ax_t, y_y_,
                    cK_=(dict(name='vegetation season day-start', units=1,
                              var_name='veg_s'),
                         dict(name='vegetation season day-end', units=1,
                              var_name='veg_e')),
                    fA_=(tyrs, tdoy),
                    fK_=dict(thr=5),
                    out=True)
            v_ = 'VegSeasonLength-5'
            t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
            o = o[1] - o[0]
            pst_(o, 'vegetation season length', 'days', 'veg_l')
            _sv(v_, o, dgpi)
            ll_(v_, t000)                                          #VegSeason-5
        if any([i in il_ for i in ['VegSeasonDayStart-2', 'VegSeasonDayEnd-2',
                                   'VegSeasonLength-2']]):
            o = _d1(['VegSeasonDayStart-2', 'VegSeasonDayEnd-2'],
                    c_t, dgpi, ax_t, y_y_,
                    cK_=(dict(name='vegetation season day-start', units=1,
                              var_name='veg_s'),
                         dict(name='vegetation season day-end', units=1,
                              var_name='veg_e')),
                    fA_=(tyrs, tdoy),
                    fK_=dict(thr=2),
                    out=True)
            v_ = 'VegSeasonLength-2'
            t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
            o = o[1] - o[0]
            pst_(o, 'vegetation season length', 'days', 'veg_l')
            _sv(v_, o, dgpi)
            ll_(v_, t000)                                          #VegSeason-2
    v_ = 'HumiWarmDays'
    if v_ in il_ and c_hurs and c_t:
        _d0(v_, (c_hurs, c_t), dgpi)                              #HumiWarmDays
    v_ = 'RhoS'
    if v_ in il_ and all([i is not None for i in (c_t, c_huss, c_ps)]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t.copy(rho_fr_t_q_p_(c_t.data, c_huss.data, c_ps.data))
        pst_(o, 'surface air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                                  #RhoS
    v_ = 'Rho975'
    if v_ in il_ and all([i is not None for i in [c_t975, c_hus975]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t975.copy(rho_fr_t_q_p_(c_t975.data, c_hus975.data, 97500.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 97500.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                                #Rho975
    v_ = 'Rho950'
    if v_ in il_ and all([i is not None for i in [c_t950, c_hus950]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t950.copy(rho_fr_t_q_p_(c_t950.data, c_hus950.data, 95000.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 95000.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                                #Rho950
    v_ = 'Rho925'
    if v_ in il_ and all([i is not None for i in [c_t925, c_hus925]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t925.copy(rho_fr_t_q_p_(c_t925.data, c_hus925.data, 92500.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 92500.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                                #Rho925
    v_ = 'Rho50m'
    if v_ in il_ and all([i is not None for i in [c_t50m, c_hus50m, c_p50m]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t50m.copy(rho_fr_t_q_p_(c_t50m.data, c_hus50m.data, c_p50m.data))
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                                #Rho50m
    v_ = 'Rho100m'
    if v_ in il_ and all([i is not None
                          for i in [c_t100m, c_hus100m, c_p100m]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t100m.copy(rho_fr_t_q_p_(c_t100m.data, c_hus100m.data,
                                       c_p100m.data))
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                               #Rho100m
    v_ = 'Rho200m'
    if v_ in il_ and all([i is not None
                          for i in [c_t200m, c_hus200m, c_p200m]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t200m.copy(rho_fr_t_q_p_(c_t200m.data, c_hus200m.data,
                                       c_p200m.data))
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                               #Rho200m
    if c_pr:
        v_ = 'DryDays'
        if v_ in il_:
            _d0(v_, c_pr, dgpi)                                        #DryDays
        v_ = 'PRmax'
        _dd(v_, c_pr, dgpi)                                              #PRmax
        if any([i in il_ for i in ['PRgt10Days', 'PRgt25Days']]):
            _d0(['PRgt10Days', 'PRgt25Days'], c_pr, dgpi)            #ExtrPrDay
        if any([i in il_ for i in ['LnstDryDays', 'SuperCooledPR',
                                   'PR7Dmax']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_pr, y0y1)
        v_ = 'PR7Dmax'
        if v_ in il_:
            cK_ = dict(name='max 7-day precipitation', var_name='pr7d')
            fA_ = (tyrs,)
            _d1(v_, c_pr, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)          #PR7Dmax
        v_ = 'LnstDryDays'
        if v_ in il_:
            t000 = l__('{} {} LL'.format(i__[v_][0], v_))
            cK_ = dict(name='longest dry days', units='days')
            for ss in s4:
                yy = [y_y_[0] + 1, y_y_[-1]] if ss == s4[0] else y_y_
                ind = np.logical_and(tsss==ss, ind_inRange_(tsyr, *yy))
                _d1(v_, extract_byAxes_(c_pr, ax_t, ind), dgpi,
                    ax_t, yy, cK_=cK_,
                    fA_=(tsyr[ind],), freq=(ss,))
            ll_(v_, t000)                                          #LnstDryDays
    v_ = 'SuperCooledPR'
    if v_ in il_ and all([i is not None for i in
                          (c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                           c_hus925, c_hus850, c_hus700)]):
        cK_ = dict(name='supercooled precipitation day', units='days')
        fA_ = (tyrs,)
        _d1(v_, (c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                 c_hus925, c_hus850, c_hus700), dgpi,
            ax_t, y_y_, cK_=cK_, fA_=fA_)                        #SuperCooledPR
    if c_snc:
        v_ = 'SncDays'
        if v_ in il_:
            _d0(v_, c_snc, dgpi)                                       #SncDays
        v_ = 'Snc25Days'
        if v_ in il_:
            _d0(v_, c_snc, dgpi, fK_=dict(thr=25))                   #Snc25Days
    if c_snd:
        v_ = 'Snd10Days'
        if v_ in il_:
            _d0(v_, c_snd, dgpi)                                     #Snd10Days
        v_ = 'Snd20Days'
        if v_ in il_:
            _d0(v_, c_snd, dgpi)                                     #Snd20Days
    v_ = 'SNWmax'
    if v_ in il_ and c_snw:
        _dd(v_, c_snw, dgpi)                                            #SNWmax
    v_ = 'PRSNmax'
    if v_ in il_ and c_prsn:
        _dd(v_, c_prsn, dgpi)                                          #PRSNmax
    v_ = 'ColdRainWarmSnowDays'
    if v_ in il_ and c_pr and c_t:
        _d0(v_, (c_pr, c_t), dgpi)                        #ColdRainWarmSnowDays
    if (any([i in il_ for i in ['ColdRainDays', 'ColdRainGT10Days',
                                'ColdRainGT20Days']]) and c_pr and c_t):
        _d0(['ColdRainDays', 'ColdRainGT10Days', 'ColdRainGT20Days'],
            (c_pr, c_t), dgpi)                                    #ColdRainDays
    if (any([i in il_ for i in ['WarmSnowDays', 'WarmSnowGT10Days',
                                'WarmSnowGT20Days']]) and c_pr and c_t):
        _d0(['WarmSnowDays', 'WarmSnowGT10Days', 'WarmSnowGT20Days'],
            (c_pr, c_t), dgpi)                                    #WarmSnowDays
    if (any([i in il_ for i in ['WarmPRSNdays', 'WarmPRSNgt10Days',
                                'WarmPRSNgt20Days']]) and c_prsn and c_t):
        _d0(['WarmPRSNdays', 'WarmPRSNgt10Days', 'WarmPRSNgt20Days'],
            (c_prsn, c_t), dgpi)                                  #WarmPRSNDays
    if (any([i in il_ for i in ['ColdPRRNdays', 'ColdPRRNgt10Days',
                                'ColdPRRNgt20Days']])
        and c_prsn and c_pr and c_t):
        _d0(['ColdPRRNdays', 'ColdPRRNgt10Days', 'ColdPRRNgt20Days'],
            (c_pr, c_prsn, c_t), dgpi)                            #ColdPRRNDays
    if (any([i in il_ for i in ['R5OScw', 'R1OScw', 'R5OSc', 'R1OSc']]) and
        c_pr and c_t):
        #c_prsn = dPRSN_fr_PR_T_(c_pr, c_t)
        o = dPRRN_fr_PR_T_(c_pr, c_t)
    else:
        o = None
    if o and c_snc and c_snw:
        attr = None if 'PRRN' not in o.attributes else  o.attributes['PRRN']
        fK_ = dict(cSnw=c_snw, attr=attr)
        v_ = 'R5OScw'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=fK_)                          #R5OScw
        v_ = 'R1OScw'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=dict(thr_r=1., **fK_))        #R1OScw
    if o and c_snc:
        attr = None if 'PRRN' not in o.attributes else o.attributes['PRRN']
        fK_ = dict(attr=attr)
        v_ = 'R5OSc'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=fK_)                           #R5OSc
        v_ = 'R1OSc'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=dict(thr_r=1., **fK_))         #R1OSc


def _h6climidx(**kwArgs):
    __hclimidx(tint='h6', **kwArgs)


def _h3climidx(**kwArgs):
    __hclimidx(tint='h3', **kwArgs)


def _h1climidx(**kwArgs):
    __hclimidx(tint='h1', **kwArgs)


def _hclimidx(tint='h6', dn=None, gwl=None, po_=None, il_=None, y0y1=None,
              c_pr=None, c_ps=None, c_t=None, c_t925=None, c_t850=None,
              c_t700=None, c_hus925=None, c_hus850=None, c_hus700=None,
              c_hus975=None, c_hus950=None, c_hus50m=None, c_hus100m=None,
              c_hus200m=None, c_t975=None, c_t950=None, c_t50m=None,
              c_t100m=None, c_t200m=None, c_p50m=None, c_p100m=None,
              c_p200m=None, c_ws=None, c_us=None, c_u975=None,
              c_u950=None, c_u925=None, c_u50m=None, c_u100m=None,
              c_u200m=None, c_vs=None, c_v975=None, c_v950=None,
              c_v925=None, c_v50m=None, c_v100m=None, c_v200m=None):

    dgpi = [dn, gwl, po_, il_]
    if any([i is None for i in dgpi]):
        raise ValueError("input of 'dn', 'gwl', 'po_', 'il_' is mandotory!")
    os.makedirs(po_, exist_ok=True)

    v_ = tint+'sfcWind'
    if c_ws is None and c_us and c_vs and v_ in il_:
        #c_ws = c_us.copy(np.sqrt(c_us.data**2 + c_vs.data**2))
        c_ws = ws_cube(c_us, c_vs)
        pst_(c_ws, 'surface wind speed', var_name='sfcWind')
    _dd(v_, c_ws, dgpi)                                              #hxSfcWind
    v_ = tint+'Wind975'
    o = None
    if v_ in il and c_u975 and c_v975:
        #o = c_u975.copy(np.sqrt(c_u975.data**2 + c_v975.data**2))
        o = ws_cube(c_u975, c_v975)
        if c_ps:
            o = iris.util.mask_cube(o, c_ps.data < 97500.)
        pst_(o, 'wind speed at 975 mb', var_name='w975')
    _dd(v_, o, dgpi)                                                 #hxWind975
    v_ = tint+'Wind950'
    o = None
    if v_ in il and c_u950 and c_v950:
        #o = c_u950.copy(np.sqrt(c_u950.data**2 + c_v950.data**2))
        o = ws_cube(c_u950, c_v950)
        if c_ps:
            o = iris.util.mask_cube(o, c_ps.data < 95000.)
        pst_(o, 'wind speed at 950 mb', var_name='w950')
    _dd(v_, o, dgpi)                                                 #hxWind950
    v_ = tint+'Wind925'
    o = None
    if v_ in il and c_u925 and c_v925:
        #o = c_u925.copy(np.sqrt(c_u925.data**2 + c_v925.data**2))
        o = ws_cube(c_u925, c_v925)
        if c_ps:
            o = iris.util.mask_cube(o, c_ps.data < 92500.)
        pst_(o, 'wind speed at 925 mb', var_name='w925')
    _dd(v_, o, dgpi)                                                 #hxWind925
    v_ = tint+'Wind50m'
    o = None
    if v_ in il and c_u50m and c_v50m:
        #o = c_u50m.copy(np.sqrt(c_u50m.data**2 + c_v50m.data**2))
        o = ws_cube(c_u50m, c_v50m)
        pst_(o, 'wind speed at 50m', var_name='w50m')
    _dd(v_, o, dgpi)                                                 #hxWind50m
    v_ = tint+'Wind100m'
    o = None
    if v_ in il and c_u100m and c_v100m:
        #o = c_u100m.copy(np.sqrt(c_u100m.data**2 + c_v100m.data**2))
        o = ws_cube(c_u100m, c_v100m)
        pst_(o, 'wind speed at 100m', var_name='w100m')
    _dd(v_, o, dgpi)                                                #hxWind100m
    v_ = tint+'Wind200m'
    o = None
    if v_ in il and c_u200m and c_v200m:
        #o = c_u200m.copy(np.sqrt(c_u200m.data**2 + c_v200m.data**2))
        o = ws_cube(c_u200m, c_v200m)
        pst_(o, 'wind speed at 200m', var_name='w200m')
    _dd(v_, o, dgpi)                                                #hxWind200m
    v_ = tint+'RhoS'
    if v_ in il_ and all([i is not None for i in (c_t, c_huss, c_ps)]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t.copy(rho_fr_t_q_p_(c_t.data, c_huss.data, c_ps.data))
        pst_(o, 'surface air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                                #hxRhoS
    v_ = tint+'Rho975'
    if v_ in il_ and all([i is not None for i in [c_t975, c_hus975]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t975.copy(rho_fr_t_q_p_(c_t975.data, c_hus975.data, 97500.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 97500.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                              #hxRho975
    v_ = tint+'Rho950'
    if v_ in il_ and all([i is not None for i in [c_t950, c_hus950]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t950.copy(rho_fr_t_q_p_(c_t950.data, c_hus950.data, 95000.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 95000.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                              #hxRho950
    v_ = tint+'Rho925'
    if v_ in il_ and all([i is not None for i in [c_t925, c_hus925]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t925.copy(rho_fr_t_q_p_(c_t925.data, c_hus925.data, 92500.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 92500.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                              #hxRho925
    v_ = tint+'Rho50m'
    if v_ in il_ and all([i is not None for i in [c_t50m, c_hus50m, c_p50m]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t50m.copy(rho_fr_t_q_p_(c_t50m.data, c_hus50m.data, c_p50m.data))
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                              #hxRho50m
    v_ = tint+'Rho100m'
    if v_ in il_ and all([i is not None
                          for i in [c_t100m, c_hus100m, c_p100m]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t100m.copy(rho_fr_t_q_p_(c_t100m.data, c_hus100m.data,
                                       c_p100m.data))
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                             #hxRho100m
    v_ = tint+'Rho200m'
    if v_ in il_ and all([i is not None
                          for i in [c_t200m, c_hus200m, c_p200m]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t200m.copy(rho_fr_t_q_p_(c_t200m.data, c_hus200m.data,
                                       c_p200m.data))
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o, dgpi)                                             #hxRho200m
    v_ = tint+'SuperCooledPR'
    if v_ in il_ and all([i is not None for i in
                          (c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                           c_hus925, c_hus850, c_hus700)]):
        ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_pr, y0y1)
        cK_ = dict(name='supercooled precipitation events', units='1',
                   var_name='prsc')
        fA_ = (tyrs,)
        _d1(v_, (c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                 c_hus925, c_hus850, c_hus700), dgpi,
            ax_t, y_y_, cK_=cK_, fA_=fA_)                      #hxSuperCooledPR


def _szG(cL):
    if not isMyIter_(cL):
        return np.prod(cL.shape) * 8 * 1.e-9
    else:
        return np.prod(cL[0].shape) * len(cL) * 8 * 1.e-9


def _afm_n(cL, ax, func, out, *args, npr=32, xm=160, **kwargs):
    if _szG(cL) < xm:
        ax_fn_mp_([i.data for i in cL] if isMyIter_(cL) else cL.data,
                  ax, func, out, *args, npr=npr, **kwargs)
    else:
        n = int(np.ceil(_szG(cL) / xm))
        cLL = [nTslice_cube(i, n) for i in cL] if isMyIter_(cL) else \
              nTslice_cube(cL, n)
        outL = [nTslice_cube(i, n) for i in out] if isMyIter_(out) else \
               nTslice_cube(out, n)
        nn = len(cLL[0]) if isMyIter_(cL) else len(cLL)
        t000 = l__('loop nTslice')
        for i in range(nn):
            ax_fn_mp_([ii[i].copy().data for ii in cLL] if isMyIter_(cL) else
                      cLL[i].copy().data,
                      ax, func,
                      [ii[i] for ii in outL] if isMyIter_(out) else outL[i],
                      *args, npr=npr, **kwargs)
            ll_(prg_(i, nn), t000)
        out_ = [concat_cube_(iris.cube.CubeList(i)) for i in outL]\
               if isMyIter_(out) else concat_cube_(iris.cube.CubeList(outL))
        return out_


def _xx(pi_, freq):
    if isinstance(pi_, str):
        return pi_ + '*' + freq + '/'
    elif isinstance(pi_, (tuple, list, set, np.ndarray)):
        return [_xx(i, freq) for i in pi_]
    else:
        raise ValueError("'pi_' must be str type or array-like of str")


def _to_xhr(cube, x=6, valid=True):
    nh = 24 / x
    cat.add_categorised_coord(cube, 'xxx', 'time',
                              lambda coord, v: np.ceil(v * nh) / nh)
    o = cube.aggregated_by('xxx',iris.analysis.MEAN)
    if valid:
        dpo = np.diff(o.coord('xxx').points)
        ddpo = np.diff(np.where(dpo != 0)[0])[1]
        sss = None if dpo[ddpo-1] != 0 else 1
        eee = None if dpo[-ddpo] != 0 else -1
    else:
        sss, eee = None, None
    cube.remove_coord('xxx')
    return extract_byAxes_(o, 'time', np.s_[sss:eee])


def _xxx(cube, freq0, freq1):
    freqs = ['mon', 'day', '6hr', '3hr', '1hr']
    if any([i not in freqs for i in [freq0, freq1]]):
        raise ValueError("unknown frequency names!")
    if freqs.index(freq1) > freqs.index(freq0):
        raise ValueError("cannot convert from low to high frequency!")
    if freq1 == 'mon':
        return pSTAT_cube(cube, 'MEAN', 'month')
    elif freq1 == 'day':
        return pSTAT_cube(cube, 'MEAN', 'day')
    elif freq1 == '6hr':
        if cube.cell_methods and cube.cell_methods[0].method.upper() == 'MEAN':
            return _to_xhr(cube)
        else:
            nn = 2 if freq0 == '3hr' else 6
            return extract_byAxes_(cube, 'time', np.s_[::nn])
    else:
        if cube.cell_methods and cube.cell_methods[0].method.upper() == 'MEAN':
            return _to_xhr(cube, x=3)
        else:
            return extract_byAxes_(cube, 'time', np.s_[::3])


def rf__(pi_, freq, folder='cordex', reg_d=None, **kwargs):
    freqs = ['mon', 'day', '6hr', '3hr', '1hr']
    _rf = eval('{}_dir_cubeL'.format(folder))
    o = None
    e = None
    for f_ in freqs[freqs.index(freq):]:
        p_ = _xx(pi_, f_)
        o = _rf(p_, ifconcat=True, **kwargs)
        if o:
            o = o['cube']
            if o:
                if reg_d:
                    o = intersection_(o, **reg_d)
                if 'period' in kwargs:
                    o = extract_period_cube(o, *kwargs['period'], yy=True)
                    e = None if o else 'yye'
                break
            else:
                e = 'cce'
    if f_ != freq and isinstance(o, iris.cube.Cube):
        o = _xxx(o, f_, freq)
    return (o, e)


def _gg(folder='cordex'):
    if folder == 'cmip5':
        yf = _here_ + 'gcm_gwls_.yml'
    elif folder == 'cordex':
        yf = _here_ + 'gcm_gwls.yml'
    else:
        raise Exception("unknown folder: {!r}".format(folder))
    with open(yf, 'r') as ymlfile:
        gg = yaml.safe_load(ymlfile)
    return gg


def _pp(yf0):
    with open(yf0, 'r') as ymlfile:
        pp = yaml.safe_load(ymlfile)
    return pp


def _yy_dn(pD, dn, gwl, gg, curr):
    if gwl[:3] == 'gwl':
        try:
            y0 = gg[gwl][pD['rcp']][pD['gcm']][pD['rip']]
            y0y1 = [y0, y0 + 29]
        except KeyError:
            y0y1 = None
    elif gwl == 'current' and pD['rcp'] == 'rcp85':
        y0y1 = curr
        dn = dn.replace('rcp85', 'historical')
    elif gwl == 'curr0130':
        y0y1 = curr
    else:
        y0y1 = curr
    return (y0y1, dn)


def cmip6_rcp_(il_, reg_d, reg_n, po_,
               sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'cmip6_smhi_len.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip6_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='cmip6')
        ll_('<<<<<<<' + dn, t0)


def cmip5_imp_rcp_(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'cmip5_import_.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip5_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='cmip5')
        ll_('<<<<<<<' + dn, t0)


def cmip5_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'cmip5_import.yml')
    #pp = _pp(_here_ + 'cmip5_import_cp.yml')
    gg = _gg('cmip5')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip5_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n))
        y0y1, dn = _yy_dn(tmp, dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        t0 = l__('>>>>>>>' + dn)
        pi1 = pp['root'] + p_
        pi0 = pi1.replace(tmp['rcp'], 'historical')
        pi_ = pi0 if y0y1[1] <= 2005 else (pi1 if y0y1[0] > 2005 else
                                           [pi0, pi1])
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, folder='cmip5')
        ll_('<<<<<<<' + dn, t0)


def norcp_rcp_(il_, reg_d, reg_n, po_,
               sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'norcp_.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2norcp_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], tmp['prd'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day', '3hr'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='norcp')
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp_(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_import__.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_import.yml')
    gg = _gg()
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
        y0y1, dn = _yy_dn(tmp, dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        t0 = l__('>>>>>>>' + dn)
        pi1 = pp['root'] + p_
        pi0 = pi1.replace(tmp['rcp'], 'historical')
        pi_ = pi0 if y0y1[1] <= 2005 else (pi1 if y0y1[0] > 2005 else
                                           [pi0, pi1])
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None):
    gwl = ''
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_import_eval.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, None, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval_dmi(il_, reg_d, reg_n, po_, tii=None):
    gwl = ''
    pp = _pp(_here_ + 'eur-11_import_eval.yml')
    y0y1 = [1989, 2010]
    for p_ in pp['p__']:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_eval(il_, reg_d, reg_n, po_, yml=None, tii=None):
    gwl = ''
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_smhi-rca4.yml')
    p_ = pp['root'] + str(pp['eval']) + '/netcdf/'
    gcm = pp[pp['eval']]['gcm']
    rcp = pp[pp['eval']]['rcp']
    rip = pp[pp['eval']]['rip']
    rcm = pp[pp['eval']]['rcm']
    ver = pp[pp['eval']]['ver']
    dn = '_'.join((gcm, rcp, rip, rcm, ver, reg_n))
    t0 = l__('>>>>>>>' + dn)
    tis = ['mon', 'day'] if tii is None else tii.split(',')
    for tint in tis:
        _xyz(il_, tint, p_, dn, gwl, None, po_, reg_d)
    ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp_(il_, reg_d, reg_n, po_,
                    sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_smhi-rca4.yml')
    pp_ = l_ind_(pp['h248'], [int(i) for i in idx.split(',')]) if idx else\
          pp['h248'][sss:eee]
    for ppi in pp_:
        pi_ = '{}{}/netcdf/'.format(pp['root'], ppi)
        dn = '_'.join((pp[ppi]['gcm'], pp[ppi]['rcp'], pp[ppi]['rip'],
                       pp[ppi]['rcm'], pp[ppi]['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                   sss=None, eee=None, idx=None, yml=None, tii=None):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_smhi-rca4.yml')
    gg = _gg()
    pp_ = l_ind_(pp['rcps'], [int(i) for i in idx.split(',')]) if idx else\
          pp['rcps'][sss:eee]
    for p0p1 in pp_:
        pi0, pi1 = p0p1[0], p0p1[1]
        pi0_, pi1_ = ('{}{}/netcdf/'.format(pp['root'], pi0),
                      '{}{}/netcdf/'.format(pp['root'], pi1))
        dn = '_'.join((pp[pi1]['gcm'], pp[pi1]['rcp'], pp[pi1]['rip'],
                       pp[pi1]['rcm'], pp[pi1]['ver'], reg_n))
        y0y1, dn = _yy_dn(pp[pi1], dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        pi_ = pi0_ if y0y1[1] <= 2005 else (pi1_ if y0y1[0] > 2005 else
                                            [pi0_, pi1_])
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eobs20_(il_, reg_d, reg_n, po_, y0y1=None):
    idir = '/nobackup/rossby22/sm_chali/DATA/hw2018/iii/obs/EOBS20/'
    vo = {'c_pr': ('rr', 1. / 3600 / 24, 'kg m-2 s-1'),
          'c_t': ('tg', None, 'K'),
          'c_tx': ('tx', None, 'K'),
          'c_tn': ('tn', None, 'K'),
          'c_rs': ('qq', 1, 'K'),
          }
    def _eobs_load(var):
        o = iris.load_cube('{}{}_ens_mean_0.1deg_reg_v20.0e.nc'
                           .format(idir, vo[var][0]))
        if reg_d is not None:
            o = intersection_(o, **reg_d)
        if vo[var][1] is None and vo[var][2]:
            o.convert_units(vo[var][2])
        elif vo[var][1] == 1 and vo[var][2]:
            o.units = vo[var][2]
        elif vo[var][2]:
            o *= vo[var][1]
            o.units = vo[var][2]
        return o if y0y1 is None else extract_period_cube(o, *y0y1)

    def _getccc(tint):
        vD = _vd(il_, tint, subg=None, xvn=999)[0]
        ccc = dict()
        for i in vD[1].keys():
            if i in vo.keys():
                o = _eobs_load(i)
            ccc.update(i, o)
        return ccc
    t0 = l__('>>>monthly')
    t1 = l__(' >>loading data')
    cccc = _getccc('month')
    ll_(' <<loading data', t1)
    _mclimidx(il_=il_, dn='EOBS20_' + reg_n, gwl='', po_=po_, **cccc)
    ll_('<<<monthly', t0)
    t0 = l__('>>>daily')
    t1 = l__(' >>loading data')
    cccc = _getccc('day')
    ll_(' <<loading data', t1)
    _dclimidx(il_=il_, dn='EOBS20_' + reg_n, gwl='', po_=po_, y0y1=y0y1,
              **cccc)
    ll_('<<<daily', t0)


def erai_(il_, reg_d, reg_n, po_, y0y1=None, tii=None):
    #from uuuu.cccc import _unify_xycoord_points
    #idir = '/nobackup/rossby22/sm_chali/DATA/hw2018/iii/obs/ERAI/'
    #def _erai_load(idir, var):
    #    o = iris.load(idir + var + '_day_ERA*.nc')
    #    o = concat_cube_(o)
    #    if reg_d is not None:
    #        o = intersection_(o, **reg_d)
    #    return extract_period_cube(o, *y0y1)
    #t0 = l__('>>>loading data')
    #c_pr = _erai_load(idir, 'pr')
    #c_t = _erai_load(idir, 'tas')
    #c_tx = _erai_load(idir, 'tasmax')
    #c_tn = _erai_load(idir, 'tasmin')
    #c_tx_m = pSTAT_cube(c_tx, 'MEAN', 'month')
    #c_tn_m = pSTAT_cube(c_tn, 'MEAN', 'month')
    #_unify_xycoord_points((c_tx_m, c_tn_m))
    #m__ = dict(c_pr=c_pr, c_t=c_t, c_tx=c_tx_m, c_tn=c_tn_m)
    #d__ = dict(c_pr=c_pr, c_t=c_t, c_tx=c_tx, c_tn=c_tn)
    #ll_('<<<loading data', t0)
    #t0 = l__('>>>monthly')
    #_mclimidx(il_=il_, dn='ERAI_' + reg_n, gwl='', po_=po_, **m__)
    #ll_('<<<monthly', t0)
    #t0 = l__('>>>daily')
    #_dclimidx(il_=il_, dn='ERAI_' + reg_n, gwl='', po_=po_, y0y1=y0y1, **d__)
    #ll_('<<<daily', t0)
    idir = '/home/rossby/imports/obs/ECMWF/ERAINT/input/'
    tis = ['mon', 'day'] if tii is None else tii.split(',')
    for tint in tis:
        _xyz(il_, tint, idir, 'ERAI_{}'.format(reg_n), '', y0y1, po_, reg_d,
             folder='cmip5')


def main():
    parser = argparse.ArgumentParser('RUN CLIMIDX')
    parser.add_argument("opt",
                        type=str,
                        help="options for dataset on BI")
    parser.add_argument("-x", "--indices",
                        type=str,
                        help="indices to be calculated")
    parser.add_argument("-X", "--indices_excl",
                        type=str,
                        help="indices not to be calculated")
    parser.add_argument("-w", "--gwl",
                        type=str,
                        help="warming levels")
    parser.add_argument("-s", "--start",
                        type=int,
                        help="simulation-loop start")
    parser.add_argument("-e", "--end",
                        type=int,
                        help="simulation-loop end")
    parser.add_argument("-i", "--idx",
                        type=str,
                        help="simulation-loop index")
    parser.add_argument("-y", "--yml",
                        type=str,
                        help="yaml file that stores paths")
    parser.add_argument("-t", "--tint",
                        type=str,
                        help="temporal resolution(s) of input data: mon,day")
    parser.add_argument("--lll",
                        type=str,
                        help="longitude/latitude limits: lo0,lo1,la0,la1")
    parser.add_argument("-d", "--domain",
                        type=str,
                        help="name of domain")
    parser.add_argument("-l", "--log",
                        type=str,
                        help="exclusive log identifier")
    args = parser.parse_args()

    xi_ = args.indices
    if xi_:
        if os.path.isfile(xi_):
            with open(xi_, 'r') as yf:
                il_ = yaml.safe_load(yf)
        elif xi_[:4] == 'grp_':
            il_ = [i for i in i__.keys()
                   if all(ii in i__[i][5] for ii in xi_[4:])]
        else:
            il_ = xi_.split(',')
    else:
        il_ = list(i__.keys())
    if args.indices_excl:
        el_ = args.indices_excl.split(',')
        for i in el_:
            il_.remove(i)
    if args.lll:
        lo0, lo1, la0, la1 = [float(i) for i in args.lll.split(',')]
        reg_d = {'longitude': [lo0, lo1], 'latitude': [la0, la1]}
    else:
        reg_d = None
    #reg_d = {'longitude': [10.0, 23.0], 'latitude': [55.0, 69.0]}
    #reg_n = 'SWE'
    #reg_d = {'longitude': [-25.0, 45.0], 'latitude': [25.0, 75.0]}
    reg_n = args.domain if args.domain else 'ALL'
    rxx = os.environ.get('r26')
    #rdir = '/nobackup/rossby22/sm_chali/DATA/energi/res/'
    rdir = '{}DATA/energi/res/'.format(rxx)
    pf_ = lambda x: '{}{}/{}/'.format(rdir, x, reg_n)
    poe_ = pf_('eval')
    poo_ = pf_('obs')
    pcdx = pf_('h248/cordex/EUR11') #rdir + 'h248/cordex/EUR11/' + reg_n + '/'
    pcmp5 = pf_('h248/cmip5') #rdir + 'h248/cmip5/' + reg_n + '/'
    pcmp6 = pf_('h248/cmip6') #rdir + 'h248/cmip6/' + reg_n + '/'
    pnorcp = pf_('h248/norcp') #rdir + 'h248/norcp/' + reg_n + '/'
    pcdx_ = pf_('gwls/cordex/EUR11') #rdir + 'gwls/' + reg_n + '/'
    pcmp5_ = pf_('gwls/cmip5') #rdir + 'obs/' + reg_n + '/'
    pcmp6_ = pf_('gwls/cmip6')

    warnings.filterwarnings("ignore", category=UserWarning)
    logn = [args.opt]
    if args.gwl:
        logn.append(args.gwl)
    if args.log:
        logn.append(args.log)
    logn = '-'.join(logn)
    nlog = len(find_patt_(r'{}_*.log'.format(logn),
               schF_keys_('', logn, ext='.log')))
    logging.basicConfig(filename=logn + '_'*nlog + '.log',
                        filemode='w',
                        level=logging.INFO)
    logging.info(' {:_^42}'.format('start of program'))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))
    logging.info(' ')

    seiyt = dict(sss=args.start, eee=args.end, idx=args.idx,
                 yml=args.yml, tii=args.tint)

    if args.opt in ('0', 'eobs20'):
        eobs20_(il_, reg_d, reg_n, poo_)
    elif args.opt in ('1', 'erai'):
        erai_(il_, reg_d, reg_n, poo_, tii=args.tint)
    elif args.opt in ('2', 'cdx_eval'):
        eur11_imp_eval(il_, reg_d, reg_n, poe_, **seiyt)
    elif args.opt in ('21', 'cdx_eval_dmi'):
        eur11_imp_eval_dmi(il_, reg_d, reg_n, poe_, tii=args.tint)
    elif args.opt in ('22', 'cdx_eval_smhi'):
        eur11_smhi_eval(il_, reg_d, reg_n, poe_, yml=args.yml, tii=args.tint)
    elif args.opt in ('3', 'cdx'):
        eur11_imp_rcp_(il_, reg_d, reg_n, pcdx, **seiyt)
    elif args.opt in ('30', 'cdx_smhi'):
        eur11_smhi_rcp_(il_, reg_d, reg_n, pcdx, **seiyt)
    elif args.opt in ('31', 'cdx_gwl'):
        eur11_imp_rcp(il_, reg_d, reg_n, pcdx_, gwl=args.gwl, **seiyt)
    elif args.opt in ('32', 'cdx_gwl_smhi'):
        eur11_smhi_rcp(il_, reg_d, reg_n, pcdx_, gwl=args.gwl, **seiyt)
    elif args.opt in ('33', 'cdx_0130'):
        eur11_imp_rcp(il_, reg_d, reg_n, pcdx_, curr=[2001, 2030],
                      gwl='curr0130', **seiyt)
    elif args.opt in ('34', 'cdx_0130_smhi'):
        eur11_smhi_rcp(il_, reg_d, reg_n, pcdx_, curr=[2001, 2030],
                       gwl='curr0130', **seiyt)
    elif args.opt in ('4', 'norcp'):
        norcp_rcp_(il_, reg_d, reg_n, pnorcp, **seiyt)
    elif args.opt in ('5', 'cmp5'):
        cmip5_imp_rcp_(il_, reg_d, reg_n, pcmp5, **seiyt)
    elif args.opt in ('51', 'cmp5_gwl'):
        cmip5_imp_rcp(il_, reg_d, reg_n, pcmp5_, gwl=args.gwl, **seiyt)
    elif args.opt in ('54', 'cmp5_xxx'):
        cmip5_imp_rcp(il_, reg_d, reg_n, pcmp5_, curr=[1981, 2100],
                      gwl='xxx', **seiyt)
    elif args.opt in ('6', 'cmp6'):
        cmip6_rcp_(il_, reg_d, reg_n, pcmp6, **seiyt)
    else:
        pass


if __name__ == '__main__':
    start_time = time.time()
    main()
    logging.info(' ')
    logging.info(' {:_^42}'.format('end of program'))
    logging.info(' {:_^42}'.format('TOTAL'))
    logging.info(' ' + rTime_(time.time() - start_time))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))

from climi.uuuu import *
from climi.climidx import *

import numpy as np
import iris
import iris.coord_categorisation as cat
import os
import yaml
import time
import logging
import argparse

from time import localtime, strftime


_here_ = get_path_(__file__)


i__ = {'ET': (1, 0, ['et'], None, ['year']),
       'HumiWarmDays': (2, 1, ['hurs', 't'], dHumiWarmDays_cube, ['season']),
       'EffPR': (3, 0, ['et', 'pr'], None, ['season', 'year']),
       'PR7Dmax': (4, 1, ['pr'], dPr7_, ['year']),
       'LnstDryDays': (5, 1, ['pr'], dLongestDryDays_, ['season']),
       'DryDays': (6, 1, ['pr'], dDryDays_, ['month']),
       'PR': (7, 0, ['pr'], None, ['season', 'year']),
       #'PR': (7, 0, ['pr'], None, ['month']),
       'PRmax': (8, 1, ['pr'], 'MAX', ['month', 'year']),
       #'PRmax': (8, 1, ['pr'], 'MAX', ['month']),
       'PRRN': (9, 0, ['pr', 'prsn'], None, ['season', 'year']),
       'PRSN': (10, 0, ['pr', 'prsn'], None, ['season', 'year']),
       'NetRO': (11, 0, ['ro'], None, ['year']),
       'SD': (12, 0, ['sd'], None, ['year']),
       'RLDS': (13, 0, ['rl'], None, ['season']),
       'RSDS': (14, 0, ['rs'], None, ['season']),
       'CoolingDegDay': (15, 1, ['tx'], dDegDay_, ['month', 'year']),
       'ConWarmDays': (16, 1, ['tx'], dConWarmDays_, ['year']),
       'TX': (17, 0, ['tx'], None, ['season', 'year']),
       #'TX': (17, 0, ['tx'], None, ['month']),
       'WarmDays': (18, 1, ['tx'], dWarmDays_, ['season', 'year']),
       'ColdDays': (19, 1, ['tx'], dColdDays_, ['season', 'year']),
       'DegDay20': (20, 1, ['t'], dDegDay_, ['year']),
       'DegDay8': (21, 1, ['t'], dDegDay8_vegSeason_, ['year']),
       'DegDay17': (22, 1, ['t'], dDegDay_, ['year']),
       'TN': (23, 0, ['tn'], None, ['season', 'year']),
       #'TN': (23, 0, ['tn'], None, ['month']),
       'SpringFrostDayEnd': (24, 1, ['tn'], dEndSpringFrost_, ['year']),
       'FrostDays': (25, 1, ['tn'], dFrostDays_, ['season', 'year']),
       'TropicNights': (26, 1, ['tn'], dTropicNights_, ['year']),
       'ZeroCrossingDays': (27, 1, ['tx', 'tn'],
                            dZeroCrossingDays_cube, ['season']),
       'VegSeasonDayEnd-5': (28, 1, ['t'], dStartEndVegSeason_, ['year']),
       'VegSeasonDayEnd-2': (29, 1, ['t'], dStartEndVegSeason_, ['year']),
       'VegSeasonDayStart-5': (30, 1, ['t'], dStartEndVegSeason_, ['year']),
       'VegSeasonDayStart-2': (31, 1, ['t'], dStartEndVegSeason_, ['year']),
       'VegSeasonLength-5': (32, 1, ['t'], dStartEndVegSeason_, ['year']),
       'VegSeasonLength-2': (33, 1, ['t'], dStartEndVegSeason_, ['year']),
       'SfcWind': (34, 1, ['ws'], None, ['month', 'season', 'year']),
       'WindGustMax': (35, 1, ['wsgs'], 'MAX', ['year']),
       'WindyDays': (36, 1, ['wsgs'], dWindyDays_, ['season', 'year']),
       'PRgt10Days': (37, 1, ['pr'], dExtrPrDays_, ['season', 'year']),
       'PRgt25Days': (38, 1, ['pr'], dExtrPrDays_, ['season', 'year']),
       'SncDays': (39, 1, ['snc'], dSncDays_, ['year']),
       'Snd10Days': (40, 1, ['snd'], dSndGT10LE20Days_, ['year']),
       'Snd20Days': (41, 1, ['snd'], dSndGT10LE20Days_, ['year']),
       'SNWmax': (42, 1, ['snw'], 'MAX', ['year']),
       'TAS': (43, 0, ['t'], None, ['season', 'year']),
       #'TAS': (43, 0, ['t'], None, ['month']),
       'DTR': (44, 0, ['tx', 'tn'], mDTR_, ['month']),
       'Rho925': (45, 1, ['t925', 'hus925', 'ps'], None, ['month']),
       'RhoS': (46, 1, ['t', 'huss', 'ps'], None, ['month']),
       'SuperCooledPR': (47, 1, ['t', 'pr', 'ps', 't925', 't850', 't700',
                                 'hus925', 'hus850', 'hus700'],
                         dFreezRainDays_, ['year']),
       'Snc25Days': (48, 1, ['snc'], dSncDays_, ['year']),
       #'R5OScw': (49, 1, ['pr', 't', 'prsn', 'snc', 'snw']),
       #'R1OScw': (50, 1, ['pr', 't', 'prsn', 'snc', 'snw']),
       #'R5OSc': (51, 1, ['pr', 't', 'prsn', 'snc']),
       #'R1OSc': (52, 1, ['pr', 't', 'prsn', 'snc']),
       'R5OScw': (49, 1, ['pr', 't', 'snc', 'snw'], dRainOnSnow_, ['year']),
       'R1OScw': (50, 1, ['pr', 't', 'snc', 'snw'], dRainOnSnow_, ['year']),
       'R5OSc': (51, 1, ['pr', 't', 'snc'], dRainOnSnow_, ['year']),
       'R1OSc': (52, 1, ['pr', 't', 'snc'], dRainOnSnow_, ['year']),
       'PRSNmax': (53, 1, ['prsn'], 'MAX', ['year']),
       'CalmDays': (54, 1, ['ws'], dCalmDays_, ['season', 'year']),
       'ConCalmDays': (55, 1, ['ws'], dConCalmDays_, ['year']),
       'Wind975': (56, 1, ['u975', 'v975', 'ps'],
                   None, ['month', 'season', 'year']),
       'Wind975toSfc': (57, 1, ['u975', 'v975', 'ws', 'ps'],
                        None, ['season', 'year']),
       'ColdRainDays': (58, 1, ['pr', 't'], dColdRainDays_, ['year']),
       'ColdRainGT10Days': (59, 1, ['pr', 't'], dColdRainDays_, ['year']),
       'ColdRainGT20Days': (60, 1, ['pr', 't'], dColdRainDays_, ['year']),
       'WarmSnowDays': (61, 1, ['pr', 't'], dWarmSnowDays_, ['year']),
       'WarmSnowGT10Days': (62, 1, ['pr', 't'], dWarmSnowDays_, ['year']),
       'WarmSnowGT20Days': (63, 1, ['pr', 't'], dWarmSnowDays_, ['year']),
       'ColdPRRNdays': (64, 1, ['pr', 't', 'prsn'], dColdPRRNdays_, ['year']),
       'ColdPRRNgt10Days': (65, 1, ['pr', 't', 'prsn'],
                            dColdPRRNdays_, ['year']),
       'ColdPRRNgt20Days': (66, 1, ['pr', 't', 'prsn'],
                            dColdPRRNdays_, ['year']),
       'WarmPRSNdays': (67, 1, ['t', 'prsn'], dWarmPRSNdays_, ['year']),
       'WarmPRSNgt10Days': (68, 1, ['t', 'prsn'], dWarmPRSNdays_, ['year']),
       'WarmPRSNgt20Days': (69, 1, ['t', 'prsn'], dWarmPRSNdays_, ['year']),
       'SST': (70, 0, ['sst'], None, ['season', 'year']),
       'SIC': (71, 0, ['sic'], None, ['season', 'year']),
       'Rho975': (72, 1, ['t975', 'hus975', 'ps'], None, ['month']),
       'hSuperCooledPR': (73, 2, ['t', 'pr', 'ps', 't925', 't850', 't700',
                                  'hus925', 'hus850', 'hus700'],
                          dFreezRainDays_, ['year']),
       'Wind925': (74, 1, ['u925', 'v925', 'ps'],
                   None, ['month', 'season', 'year']),
       'Wind925toSfc': (75, 1, ['u925', 'v925', 'ws', 'ps'],
                        None, ['season', 'year']),
       'FirstDayWithoutFrost': (76, 1, ['tn'],
                                dFirstDayWithoutFrost_, ['year']),
       'CalmDays925': (77, 1, ['u925', 'v925', 'ps'], dCalmDays_, ['year']),
       'ConCalmDays925': (78, 1, ['u925', 'v925', 'ps'],
                          dConCalmDays_, ['year']),
       'CalmDays975': (79, 1, ['u975', 'v975', 'ps'], dCalmDays_, ['year']),
       'ConCalmDays975': (80, 1, ['u975', 'v975', 'ps'],
                          dConCalmDays_, ['year'])}


var_ = {'pr': 'pr', 'et': 'evspsbl', 'prsn': 'prsn', 'ro': 'mrro',
        'sd': 'sund', 'rs': 'rsds', 'rl': 'rlds', 't': 'tas',
        'tx': 'tasmax', 'tn': 'tasmin', 'ws': 'sfcWind',
        'wsgs': 'wsgsmax', 'snc': 'snc', 'snd': 'snd', 'snw': 'snw',
        'hurs': 'hurs', 'huss': 'huss', 'ps': 'ps', 't975': 'ta975',
        't925': 'ta925', 't850': 'ta850', 't700': 'ta700', 'hus975': 'hus975',
        'hus925': 'hus925', 'hus850': 'hus850', 'hus700': 'hus700',
        'u975': 'ua975', 'v975': 'va975', 'u925': 'ua925', 'v925': 'va925',
        'sst': 'tos', 'sic': 'sic'}


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
    if nn == 1 and subg is None:
        tmp = [(i, i__[i][2]) for i in ii_ if i__[i][1] == nn]
        return list(_vv2(tmp, xvn))
    else:
        tmp = [i__[i][2] for i in ii_ if i__[i][1] == nn]
        tmp_ = set(flt_l(tmp))
        if nn == 1 and len(tmp_) > xvn:
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


def _vd(ii_, tint, subg=None):
    return _vd_fr_vv(_vv(ii_, tint, subg))


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
        f__ = _hclimidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    def __xyz(vd):
        ccc = dict()
        if isinstance(vd, tuple):
            t000 = l__("rf__(): {} variables".format(len(vd[1].keys())))
            ll_(', '.join(vd[1].keys()))
            ll_(', '.join(vd[0]))
            tmp = []
            for kk in vd[1].keys():
                if y0y1:
                    cc = rf__(pi_, tint, var=vd[1][kk], period=y0y1,
                              reg_d=reg_d, folder=folder)
                else:
                    cc = rf__(pi_, tint, var=vd[1][kk], reg_d=reg_d,
                              folder=folder)
                if cc:
                    tmp.append(kk)
                ccc.update({kk: cc})
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


def _mclimidx(c_pr=None, c_et=None, c_prsn=None, c_ro=None, c_sd=None,
              c_rs=None, c_rl=None, c_t=None, c_tx=None, c_tn=None,
              c_sst=None, c_sic=None,
              dn=None, gwl=None, po_=None, il_=None):

    if any([i is None for i in [dn, gwl, po_, il_]]):
        raise ValueError("input of 'dn', 'gwl', 'po_', 'il_' is mandotory!")

    os.makedirs(po_, exist_ok=True)

    def _sv(v_, o, freq=None):
        fn = '{}{}.nc'
        freq = freq if freq else i__[v_][4]
        if len(freq) == 1:
            cubesv_(o, fn.format(po_, '_'.join((v_, dn, freq[0], gwl))))
        else:
            for oo, ff in zip(o, freq):
                cubesv_(oo, fn.format(po_, '_'.join((v_, dn, ff, gwl))))

    def _mm(v_, cube, freq=None):
        fn = '{}{}.nc'
        if v_ in il_ and cube:
            o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                           *i__[v_][4])
            _sv(v_, o, freq=freq)

    def _m0(v_, cube, fA_=(), fK_={}, pK_=None, freq=None):
        if isMyIter_(v_):
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
        if not isMyIter_(v_) or len(v_) == 1:
            _sv(vk_, o, freq=freq)
        else:
            for i, ii in zip(v_, o):
                _sv(i, ii, freq=freq)
        ll_(lmsg_, t000)

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
        _m0(v_, (c_tx, c_tn))                                              #DTR
    _mm('SST', c_sst)                                                      #SST
    _mm('SIC', c_sic)                                                      #SIC


def _dclimidx(c_pr=None, c_t=None, c_tx=None, c_tn=None, c_wsgs=None,
              c_snc=None, c_snd=None, c_snw=None, c_hurs=None, c_huss=None,
              c_ps=None, c_t975=None, c_t925=None, c_t850=None, c_t700=None,
              c_hus975=None, c_hus925=None, c_hus850=None, c_hus700=None,
              c_prsn=None, c_ws=None, c_u975=None, c_v975=None, c_u925=None,
              c_v925=None,
              dn=None, gwl=None, po_=None, il_=None, y0y1=None):

    if any([i is None for i in [dn, gwl, po_, il_]]):
        raise ValueError("input of 'dn', 'gwl', 'po_', 'il_' is mandotory!")

    os.makedirs(po_, exist_ok=True)

    s4 = ('djf', 'mam', 'jja', 'son')

    def _sv(v_, o, freq=None):
        fn = '{}{}.nc'
        freq = freq if freq else i__[v_][4]
        if len(freq) == 1:
            cubesv_(o, fn.format(po_, '_'.join((v_, dn, freq[0], gwl))))
        else:
            for oo, ff in zip(o, freq):
                cubesv_(oo, fn.format(po_, '_'.join((v_, dn, ff, gwl))))

    def _dd(v_, cube, freq=None):
        freq = freq if freq else i__[v_][4]
        if v_ in il_ and cube:
            t000 = l__('{} {}'.format(i__[v_][0], v_))
            o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                           *freq)
            _sv(v_, o, freq=freq)
            ll_(v_, t000)

    def _d0(v_, cube, fA_=(), fK_={}, pK_=None, freq=None):
        if isMyIter_(v_):
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
        if not isMyIter_(v_) or len(v_) == 1:
            _sv(vk_, o, freq=freq)
        else:
            for i, ii in zip(v_, o):
                _sv(i, ii, freq=freq)
        ll_(lmsg_, t000)

    def _tt(cube):
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

    def _d1(v_, cube, ax_t, y_y_,
            cK_={}, fA_=(), fK_={}, freq=None, out=False):
        def _inic(icK_):
            return initAnnualCube_(cube[0] if isMyIter_(cube) else cube,
                                   y_y_, **icK_)
        if isMyIter_(v_):
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
        if not isMyIter_(v_) or len(v_) == 1:
            _sv(vk_, o, freq=freq)
        else:
            for i, ii in zip(v_, o):
                _sv(i, ii, freq=freq)
        ll_(lmsg_, t000)
        if out:
            return o

    if c_wsgs is not None:
        _dd('WindGustMax', c_wsgs)                                 #WindGustMax
        _d0('WindyDays', c_wsgs)                                     #WindyDays
    o = None
    if any([i in il_ for i in ['Wind975', 'Wind975toSfc', 'CalmDays975',
                               'ConCalmDays975']]):
        if c_u975 and c_v975:
            o = c_u975.copy(np.sqrt(c_u975.data**2 + c_v975.data**2))
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 97500.)
            pst_(o, 'wind speed at 975 mb', var_name='w975')
    _dd('Wind975', o)                                                  #Wind975
    if o:
        v_ = 'CalmDays975'
        if v_ in il_:
            _d0(v_, o)                                             #CalmDays975
        v_ = 'ConCalmDays975'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, ax_t, y_y_, cK_=cK_, fA_=fA_)            #ConCalmDays975
    if 'Wind975toSfc' in  il_ and o and c_ws:
        o = o.copy(o.data / c_ws.data)
        pst_(o, 'wsr_975_to_sfc', '1')
        _dd('Wind975toSfc', o)                                    #Wind975toSfc
    o = None
    if any([i in il_ for i in ['Wind925', 'Wind925toSfc', 'CalmDays925',
                               'ConCalmDays925']]):
        if c_u925 and c_v925:
            o = c_u925.copy(np.sqrt(c_u925.data**2 + c_v925.data**2))
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 92500.)
            pst_(o, 'wind speed at 925 mb', var_name='w925')
    _dd('Wind925', o)                                                  #Wind925
    if o:
        v_ = 'CalmDays925'
        if v_ in il_:
            _d0(v_, o)                                             #CalmDays925
        v_ = 'ConCalmDays925'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, o, ax_t, y_y_, cK_=cK_, fA_=fA_)            #ConCalmDays925
    if 'Wind925toSfc' in  il_ and o and c_ws:
        o = o.copy(o.data / c_ws.data)
        pst_(o, 'wsr_925_to_sfc', '1')
        _dd('Wind925toSfc', o)                                    #Wind925toSfc
    _dd('SfcWind', c_ws)                                               #SfcWind
    if c_ws:
        v_ = 'CalmDays'
        if v_ in il_:
            _d0(v_, c_ws)                                             #CalmDays
        v_ = 'ConCalmDays'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_ws)
            cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, c_ws, ax_t, y_y_, cK_=cK_, fA_=fA_)            #ConCalmDays
    if c_tx:
        v_ = 'WarmDays'
        if v_ in il_:
            _d0(v_, c_tx)                                             #WarmDays
        v_ = 'ColdDays'
        if v_ in il_:
            _d0(v_, c_tx)                                             #ColdDays
        v_ = 'CoolingDegDay'
        if v_ in il_:
            _d0(v_, c_tx, fK_=dict(thr=20),
                pK_=dict(name='degree day cooling', var_name='dd20x_'))
                                                                 #CoolingDegDay
        v_ = 'ConWarmDays'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_tx)
            cK_ = dict(name='continue warm days', units='days')
            fA_ = (tyrs,)
            _d1(v_, c_tx, ax_t, y_y_, cK_=cK_, fA_=fA_)            #ConWarmDays
    if c_tn:
        v_ = 'FrostDays'
        if v_ in il_:
            _d0(v_, c_tn)                                            #FrostDays
        v_ = 'TropicNights'
        if v_ in il_:
            _d0(v_, c_tn)                                          #TropicNight
        if any([i in il_ for i in ['SpringFrostDayEnd',
                                   'FirstDayWithoutFrost']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_tn)
        v_ = 'SpringFrostDayEnd'
        if v_ in il_:
            cK_ = dict(name='spring frost end-day', units=1,
                       var_name='frost_end')
            fA_ = (tyrs, tdoy)
            _d1(v_, c_tn, ax_t, y_y_, cK_=cK_, fA_=fA_)      #SpringFrostDayEnd
        v_ = 'FirstDayWithoutFrost'
        if v_ in il_:
            cK_ = dict(name='first day without frost', units=1,
                       var_name='day1nofrost')
            fA_ = (tyrs, tdoy)
            _d1(v_, c_tn, ax_t, y_y_, cK_=cK_, fA_=fA_)   #FirstDayWithoutFrost 
    v_ = 'ZeroCrossingDays'
    if v_ in il_ and c_tx and c_tn:
        _d0(v_, (c_tx, c_tn))                                 #ZeroCrossingDays
    if c_t:
        v_ = 'DegDay20'
        if v_ in il_:
            _d0(v_, c_t, fK_=dict(thr=20))                            #DegDay20
        v_ = 'DegDay17'
        if v_ in il_:
            _d0(v_, c_t, fK_=dict(thr=17, left=True))                 #DegDay17
        if any([i in il_ for i in ['DegDay8',
                                   'VegSeasonDayStart-5',
                                   'VegSeasonDayEnd-5', 'VegSeasonLength-5',
                                   'VegSeasonDayStart-2',
                                   'VegSeasonDayEnd-2', 'VegSeasonLength-2',
                                   'SuperCooledPR']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_t)
        v_ = 'DegDay8'
        if v_ in il_:
            cK_ = dict(name='degree day g8 vegetation season',
                       var_name='dd8_')
            fA_ = (tyrs,)
            _d1(v_, c_t, ax_t, y_y_, cK_=cK_, fA_=fA_)                 #DegDay8
        if any([i in il_ for i in ['VegSeasonDayStart-5', 'VegSeasonDayEnd-5',
                                   'VegSeasonLength-5']]):
            o = _d1(['VegSeasonDayStart-5', 'VegSeasonDayEnd-5'],
                    c_t, ax_t, y_y_,
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
            _sv(v_, o)
            ll_(v_, t000)                                          #VegSeason-5
        if any([i in il_ for i in ['VegSeasonDayStart-2', 'VegSeasonDayEnd-2',
                                   'VegSeasonLength-2']]):
            o = _d1(['VegSeasonDayStart-2', 'VegSeasonDayEnd-2'],
                    c_t, ax_t, y_y_,
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
            _sv(v_, o)
            ll_(v_, t000)                                          #VegSeason-2
    v_ = 'HumiWarmDays'
    if v_ in il_ and c_hurs and c_t:
        _d0(v_, (c_hurs, c_t))                                    #HumiWarmDays
    v_ = 'RhoS'
    if v_ in il_ and all([i is not None for i in (c_t, c_huss, c_ps)]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t.copy(rho_fr_t_q_p_(c_t.data, c_huss.data, c_ps.data))
        pst_(o, 'surface air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o)                                                        #RhoS
    v_ = 'Rho975'
    if v_ in il_ and all([i is not None for i in [c_t975, c_hus975]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t975.copy(rho_fr_t_q_p_(c_t975.data, c_hus975.data, 97500.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 97500.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o)                                                      #Rho975
    v_ = 'Rho925'
    if v_ in il_ and all([i is not None for i in [c_t925, c_hus925]]):
        t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
        o = c_t925.copy(rho_fr_t_q_p_(c_t925.data, c_hus925.data, 92500.))
        if c_ps is not None:
            o = iris.util.mask_cube(o, c_ps.data < 92500.)
        pst_(o, 'air density', 'kg m-3', 'rho')
        ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
        _dd(v_, o)                                                      #Rho925
    if c_pr:
        v_ = 'DryDays'
        if v_ in il_:
            _d0(v_, c_pr)                                              #DryDays
        v_ = 'PRmax'
        if v_ in il_:
            _dd(v_, c_pr)                                                #PRmax
        if any([i in il_ for i in ['PRgt10Days', 'PRgt25Days']]):
            _d0(['PRgt10Days', 'PRgt25Days'], c_pr)                  #ExtrPrDay
        if any([i in il_ for i in ['LnstDryDays', 'SuperCooledPR',
                                   'PR7Dmax']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_pr)
        v_ = 'PR7Dmax'
        if v_ in il_:
            cK_ = dict(name='max 7-day precipitation', var_name='pr7d')
            fA_ = (tyrs,)
            _d1(v_, c_pr, ax_t, y_y_, cK_=cK_, fA_=fA_)                #PR7Dmax
        v_ = 'LnstDryDays'
        if v_ in il_:
            t000 = l__('{} {} LL'.format(i__[v_][0], v_))
            cK_ = dict(name='longest dry days', units='days')
            for ss in s4:
                yy = [y_y_[0] + 1, y_y_[-1]] if ss == s4[0] else y_y_
                ind = np.logical_and(tsss==ss, ind_inRange_(tsyr, *yy))
                _d1(v_, extract_byAxes_(c_pr, ax_t, ind), ax_t, yy, cK_=cK_,
                    fA_=(tsyr[ind],), freq=(ss,))
            ll_(v_, t000)                                          #LnstDryDays
    v_ = 'SuperCooledPR'
    if v_ in il_ and all([i is not None for i in
                          (c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                           c_hus925, c_hus850, c_hus700)]):
        cK_ = dict(name='supercooled precipitation day', units='days')
        fA_ = (tyrs,)
        _d1(v_, (c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                 c_hus925, c_hus850, c_hus700),
            ax_t, y_y_, cK_=cK_, fA_=fA_)                        #SuperCooledPR
    if c_snc:
        v_ = 'SncDays'
        if v_ in il_:
            _d0(v_, c_snc)                                             #SncDays
        v_ = 'Snc25Days'
        if v_ in il_:
            _d0(v_, c_snc, fK_=dict(thr=25))                         #Snc25Days
    if c_snd:
        v_ = 'Snd10Days'
        if v_ in il_:
            _d0(v_, c_snd)                                           #Snd10Days
        v_ = 'Snd20Days'
        if v_ in il_:
            _d0(v_, c_snd)                                           #Snd20Days
    v_ = 'SNWmax'
    if v_ in il_ and c_snw:
        _dd(v_, c_snw)                                                  #SNWmax
    v_ = 'PRSNmax'
    if v_ in il_ and c_prsn:
        _dd(v_, c_prsn)                                                #PRSNmax
    if (any([i in il_ for i in ['ColdRainDays', 'ColdRainGT10Days',
                                'ColdRainGT20Days']]) and c_pr and c_t):
        _d0(['ColdRainDays', 'ColdRainGT10Days', 'ColdRainGT20Days'],
            (c_pr, c_t))                                          #ColdRainDays
    if (any([i in il_ for i in ['WarmSnowDays', 'WarmSnowGT10Days',
                                'WarmSnowGT20Days']]) and c_pr and c_t):
        _d0(['WarmSnowDays', 'WarmSnowGT10Days', 'WarmSnowGT20Days'],
            (c_pr, c_t))                                          #WarmSnowDays
    if (any([i in il_ for i in ['WarmPRSNdays', 'WarmPRSNgt10Days',
                                'WarmPRSNgt20Days']]) and c_prsn and c_t):
        _d0(['WarmPRSNdays', 'WarmPRSNgt10Days', 'WarmPRSNgt20Days'],
            (c_prsn, c_t))                                        #WarmPRSNDays
    if (any([i in il_ for i in ['ColdPRRNdays', 'ColdPRRNgt10Days',
                                'ColdPRRNgt20Days']])
        and c_prsn and c_pr and c_t):
        _d0(['ColdPRRNdays', 'ColdPRRNgt10Days', 'ColdPRRNgt20Days'],
            (c_pr, c_prsn, c_t))                                  #ColdPRRNDays
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
            _d0(v_, (o, c_snc), fK_=fK_)                                #R5OScw
        v_ = 'R1OScw'
        if v_ in il_:
            _d0(v_, (o, c_snc), fK_=dict(thr_r=1., **fK_))              #R1OScw
    if o and c_snc:
        attr = None if 'PRRN' not in o.attributes else o.attributes['PRRN']
        fK_ = dict(attr=attr)
        v_ = 'R5OSc'
        if v_ in il_:
            _d0(v_, (o, c_snc), fK_=fK_)                                 #R5OSc
        v_ = 'R1OSc'
        if v_ in il_:
            _d0(v_, (o, c_snc), fK_=dict(thr_r=1., **fK_))               #R1OSc


def _hclimidx(c_pr=None, c_ps=None, c_t=None, c_t925=None, c_t850=None,
              c_t700=None, c_hus925=None, c_hus850=None, c_hus700=None,
              dn=None, gwl=None, po_=None, il_=None, y0y1=None):
    if any([i is None for i in [dn, gwl, po_, il_]]):
        raise ValueError("input of 'dn', 'gwl', 'po_', 'il_' is mandotory!")
    os.makedirs(po_, exist_ok=True)
    v_ = 'hSuperCooledPR'
    if v_ in il_ and all([i is not None for i in
                          (c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                           c_hus925, c_hus850, c_hus700)]):
        t000 = l__('data loading c_pr')
        rm_t_aux_cube(c_pr)
        yr_doy_cube(c_pr)
        tyrs, tdoy = c_pr.coord('year').points, c_pr.coord('doy').points
        if y0y1 is None:
            y0y1 = tyrs[[0, -1]]
        ax_t = c_pr.coord_dims('time')[0]
        ll_('data loading c_pr', t000)                #########################
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        au = dict(CLIMIDX='6hr events')
        c = initAnnualCube_(c_pr, y0y1, 'supercooled precipitation events',
                            '1', var_name='prsc', attr_updt=au)
        c_ = _afm_n((c_pr, c_ps, c_t, c_t925, c_t850, c_t700,
                     c_hus925, c_hus850, c_hus700), ax_t, dFreezRainDays_,
                     c, tyrs)
        c = c_ if c_ else c
        cubesv_(c, po_ + '_'.join((v_, dn, 'year', gwl)) + '.nc')
        ll_(v_, t000)                                           #hSuperCooledPR


def _szG(cL):
    if not isMyIter_(cL):
        return np.product(cL.shape) * 8 / 1.e9
    else:
        return np.product(cL[0].shape) * len(cL) * 8 / 1.e9


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
        return pi_ + freq + '/'
    elif isinstance(pi_, (tuple, list, set, np.ndarray)):
        return [_xx(i, freq) for i in pi_]
    else:
        raise ValueError("'pi_' must be str type or array-like of str")


def _3hr_to_6hr(cube):
    cat.add_categorised_coord(cube, 'xxx', 'time',
                              lambda coord, x: np.round(x * 4) / 4)
    o = cube.aggregated_by('xxx',iris.analysis.MEAN)
    cube.remove_coord('xxx')
    n0 = len(cube.coord('time').points[::2])
    n1 = len(o.coord('time').points)
    end = None if n0 == n1 else n0 - n1
    return extract_byAxes_(o, 'time', np.s_[:end])


def _xxx(cube, freq0, freq1):
    freqs = ['mon', 'day', '6hr', '3hr']
    if any([i not in freqs for i in [freq0, freq1]]):
        raise ValueError("unknown frequency names!")
    if freqs.index(freq1) > freqs.index(freq0):
        raise ValueError("cannot convert from low to high frequency!")
    if freq1 == 'mon':
        return pSTAT_cube(cube, 'MEAN', 'month')
    if freq1 == 'day':
        return pSTAT_cube(cube, 'MEAN', 'day')
    elif cube.cell_methods and cube.cell_methods[0].method.upper() == 'MEAN':
        return _3hr_to_6hr(cube)
    else:
        return extract_byAxes_(cube, 'time', np.s_[::2])


def rf__(pi_, freq, folder='cordex', reg_d=None, **kwargs):
    freqs = ['mon', 'day', '6hr', '3hr']
    _rf = eval('{}_dir_cubeL'.format(folder))
    o = None
    for f_ in freqs[freqs.index(freq):]:
        p_ = _xx(pi_, f_)
        o = _rf(p_, ifconcat=True, **kwargs)
        if o is not None:
            o = o['cube']
            if reg_d is not None:
                o = intersection_(o, **reg_d)
            if 'period' in kwargs:
                o = extract_period_cube(o, *kwargs['period'], yy=True)
            break
    if f_ != freq and o is not None:
        o = _xxx(o, f_, freq)
    return o


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
        y0y1 = None
    return (y0y1, dn)


def cmip5_imp_rcp_(il_, reg_d, reg_n, po_, sss=None, eee=None):
    pp = _pp(_here_ + 'cmip5_import_.yml')
    ppp = pp['p_'][sss:eee]
    for p_ in ppp:
        tmp = path2cmip5_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='cmip5')
        ll_('<<<<<<<' + dn, t0)


def cmip5_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None):
    pp = _pp(_here_ + 'cmip5_import.yml')
    gg = _gg('cmip5')
    ppp = pp['p_'][sss:eee]
    for p_ in ppp:
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
        tis = ['mon', 'day']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, folder='cmip5')
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp_(il_, reg_d, reg_n, po_, sss=None, eee=None):
    pp = _pp(_here_ + 'eur-11_import__.yml')
    for p_ in pp['p_'][sss:eee]:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['version'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day', '6hr']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None):
    pp = _pp(_here_ + 'eur-11_import_.yml')
    gg = _gg()
    for p_ in pp['p_'][sss:eee]:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['version'], reg_n))
        y0y1, dn = _yy_dn(tmp, dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        t0 = l__('>>>>>>>' + dn)
        pi1 = pp['root'] + p_
        pi0 = pi1.replace(tmp['rcp'], 'historical')
        pi_ = pi0 if y0y1[1] <= 2005 else (pi1 if y0y1[0] > 2005 else
                                           [pi0, pi1])
        tis = ['mon', 'day', '6hr']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval(il_, reg_d, reg_n, po_, sss=None, eee=None):
    gwl = ''
    pp = _pp(_here_ + 'eur-11_import_eval.yml')
    for p_ in pp['p_'][sss:eee]:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['version'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day', '6hr']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, None, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval_dmi(il_, reg_d, reg_n, po_):
    gwl = ''
    pp = _pp(_here_ + 'eur-11_import_eval.yml')
    y0y1 = [1989, 2010]
    for p_ in pp['p__']:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['version'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day', '6hr']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_eval(il_, reg_d, reg_n, po_):
    gwl = ''
    pp = _pp(_here_ + 'eur-11_smhi-rca4.yml')
    p_ = pp['root'] + str(pp['eval']) + '/netcdf/'
    gcm = pp[pp['eval']]['gcm']
    rcp = pp[pp['eval']]['rcp']
    rip = pp[pp['eval']]['rip']
    rcm = pp[pp['eval']]['rcm']
    version = pp[pp['eval']]['version']
    dn = '_'.join((gcm, rcp, rip, rcm, version, reg_n))
    t0 = l__('>>>>>>>' + dn)
    tis = ['mon', 'day', '6hr']
    for tint in tis:
        _xyz(il_, tint, p_, dn, gwl, None, po_, reg_d)
    ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp_(il_, reg_d, reg_n, po_, sss=None, eee=None):
    pp = _pp(_here_ + 'eur-11_smhi-rca4.yml')
    for ppi in pp['h248'][sss:eee]:
        pi_ = '{}{}/netcdf/'.format(pp['root'], ppi)
        dn = '_'.join((pp[ppi]['gcm'], pp[ppi]['rcp'], pp[ppi]['rip'],
                       pp[ppi]['rcm'], pp[ppi]['version'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day', '6hr']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                   sss=None, eee=None):
    pp = _pp(_here_ + 'eur-11_smhi-rca4.yml')
    gg = _gg()
    for p0p1 in pp['rcps'][sss:eee]:
        pi0, pi1 = p0p1[0], p0p1[1]
        pi0_, pi1_ = ('{}{}/netcdf/'.format(pp['root'], pi0),
                      '{}{}/netcdf/'.format(pp['root'], pi1))
        dn = '_'.join((pp[pi1]['gcm'], pp[pi1]['rcp'], pp[pi1]['rip'],
                       pp[pi1]['rcm'], pp[pi1]['version'], reg_n))
        y0y1, dn = _yy_dn(pp[pi1], dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        pi_ = pi0_ if y0y1[1] <= 2005 else (pi1_ if y0y1[0] > 2005 else
                                            [pi0_, pi1_])
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day', '6hr']
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d)
        ll_('<<<<<<<' + dn, t0)


def eobs20_(il_, reg_d, reg_n, po_, y0y1=[1989, 2008]):
    from uuuu.cccc import _unify_xycoord_points
    def _eobs_load(idir, var):
        o = iris.load_cube(idir + var + '_ens_mean_0.1deg_reg_v20.0e.nc')
        if reg_d is not None:
            o = intersection_(o, **reg_d)
        return extract_period_cube(o, *y0y1)
    idir = '/nobackup/rossby22/sm_chali/DATA/hw2018/iii/obs/EOBS20/'
    t0 = l__('>>>loading data')
    c_pr = _eobs_load(idir, 'rr')
    c_pr *= 1. / 3600 / 24
    c_pr.units = 'kg m-2 s-1'
    c_t = _eobs_load(idir, 'tg')
    c_t.convert_units('K')
    c_tx = _eobs_load(idir, 'tx')
    c_tx.convert_units('K')
    c_tn = _eobs_load(idir, 'tn')
    c_tn.convert_units('K')
    c_rs = _eobs_load(idir, 'qq')
    c_rs.units = 'W m-2'
    c_tx_m = pSTAT_cube(c_tx, 'MEAN', 'month')
    c_tn_m = pSTAT_cube(c_tn, 'MEAN', 'month')
    _unify_xycoord_points((c_tx_m, c_tn_m))
    m__ = dict(c_pr=c_pr, c_rs=c_rs, c_t=c_t, c_tx=c_tx_m, c_tn=c_tn_m)
    d__ = dict(c_pr=c_pr, c_t=c_t, c_tx=c_tx, c_tn=c_tn)
    ll_('<<<loading data', t0)
    t0 = l__('>>>monthly')
    _mclimidx(il_=il_, dn='EOBS20_' + reg_n, gwl='', po_=po_, **m__)
    ll_('<<<monthly', t0)
    t0 = l__('>>>daily')
    _dclimidx(il_=il_, dn='EOBS20_' + reg_n, gwl='', po_=po_, y0y1=y0y1,
              **d__)
    ll_('<<<daily', t0)


def erai_(il_, reg_d, reg_n, po_, y0y1=[1989, 2008]):
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
    tis = ['mon', 'day']
    for tint in tis:
        _xyz(il_, tint, idir, 'ERAI_{}'.format(reg_n), '', y0y1, po_, reg_d,
             folder='cmip5')
    ll_('<<<<<<<' + dn, t0)


def main():
    parser = argparse.ArgumentParser('RUN CLIMIDX')
    parser.add_argument("opt", type=int,
                        help="options for dataset on BI")
    parser.add_argument("-g", "--gwl",
                        type=str, help="warming levels")
    parser.add_argument("-s", "--start",
                        type=int, help="simulation-loop start")
    parser.add_argument("-e", "--end",
                        type=int, help="simulation-loop end")
    parser.add_argument("-l", "--log",
                        type=str, help="exclusive log identifier")
    args = parser.parse_args()
    il_ = list(i__.keys())
    il_.remove('SIC')
    il_.remove('SST')
    il_.remove('hSuperCooledPR')
    #il_.remove('SuperCooledPR')
    #il_.remove('FirstDayWithoutFrost')
    #il_.remove('SpringFrostDayEnd')
    #il_ = ['TAS', 'TX', 'TN', 'PR', 'PRmax']
    il_ = ['SNWmax', 'R5OScw', 'R1OScw']
    #il_ = ['CalmDays975', 'ConCalmDays975', 'CalmDays925', 'ConCalmDays925']
    #       'Wind975toSfc', 'ColdRainDays', 'ColdRainGT10Days',
    #       'ColdRainGT20Days', 'WarmSnowDays', 'WarmSnowGT10Days',
    #       'WarmSnowGT20Days', 'ColdPRRNdays', 'ColdPRRNgt10Days',
    #       'ColdPRRNgt20Days', 'WarmPRSNdays', 'WarmPRSNgt10Days',
    #       'WarmPRSNgt20Days', 'Rho975', 'Wind925', 'SfcWind',
    #       'Wind925toSfc', 'SuperCooledPR']
    #reg_d = {'longitude': [10.0, 23.0], 'latitude': [55.0, 69.0]}
    #reg_n = 'SWE'
    reg_d = None
    #reg_d = {'longitude': [-25.0, 45.0], 'latitude': [25.0, 75.0]}
    reg_n = 'EUR'
    rxx = os.environ.get('r24')
    #rdir = '/nobackup/rossby22/sm_chali/DATA/energi/res/'
    rdir = '{}DATA/energi/res/'.format(rxx)
    po_ = rdir + 'eval/' + reg_n + '/'
    #pcdx = rdir + 'h248/cordex/EUR11/' + reg_n + '/'
    pcdx = rdir + 'h248/cordex/EUR11/grp1/'
    pcmp = rdir + 'h248/cmip5/' + reg_n + '/'
    po__ = rdir + 'gwls/' + reg_n + '/'
    po___ = rdir + 'obs/' + reg_n + '/'
    po____ = rdir + 'cmip5/' + reg_n + '/'

    logn = [reg_n, str(args.opt)]
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

    if args.opt == 0:
        eur11_imp_eval(il_, reg_d, reg_n, po_, sss=args.start, eee=args.end)
    elif args.opt == 1:
        eur11_imp_eval_dmi(il_, reg_d, reg_n, po_)
    elif args.opt == 2:
        eur11_imp_rcp(il_, reg_d, reg_n, po__,
                      gwl=args.gwl, sss=args.start, eee=args.end)
    elif args.opt == 22:
        eur11_imp_rcp(il_, reg_d, reg_n, po__, curr=[2001, 2030],
                      gwl='curr0130', sss=args.start, eee=args.end)
    elif args.opt == 3:
        eur11_smhi_eval(il_, reg_d, reg_n, po_)
    elif args.opt == 4:
        eur11_smhi_rcp(il_, reg_d, reg_n, po__,
                       gwl=args.gwl, sss=args.start, eee=args.end)
    elif args.opt == 44:
        eur11_smhi_rcp(il_, reg_d, reg_n, po__, curr=[2001, 2030],
                       gwl='curr0130', sss=args.start, eee=args.end)
    elif args.opt == 5:
        eobs20_(il_, reg_d, reg_n, po___)
    elif args.opt == 6:
        erai_(il_, reg_d, reg_n, po___)
    elif args.opt == 7:
        cmip5_imp_rcp(il_, reg_d, reg_n, po____,
                      gwl=args.gwl, sss=args.start, eee=args.end)
    elif args.opt == 77:
        cmip5_imp_rcp(il_, reg_d, reg_n, po____, curr=[2001, 2030],
                      gwl='curr0130', sss=args.start, eee=args.end)
    elif args.opt == 8:
        eur11_smhi_rcp_(il_, reg_d, reg_n, pcdx, sss=args.start, eee=args.end)
    elif args.opt == 9:
        eur11_imp_rcp_(il_, reg_d, reg_n, pcdx, sss=args.start, eee=args.end)
    elif args.opt == 10:
        cmip5_imp_rcp_(il_, reg_d, reg_n, pcmp, sss=args.start, eee=args.end)
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

"""
>--#¤&#¤%/%&(¤%¤#%"¤#"#¤%&¤#!"#%#%&()/=?=()/&%¤%&/()/£$€¥{[}]()=?=)(/&¤%&/()--<
>--------------------------Heat Wave Magnitude Index---application interface--<
>--#¤&&()/=?=()/&#¤%&/(=)(/&$€¥{[]}]\±$¡$£€@"#%¤#¤&/("#¤!&/(%&"#¤"¤%&#%/)/*#--<
* get_cube_obs_     : cube with yr & doy & ax_t     --> hwmi_obs_
* hwmi_obs_         : hwmi for observational data
* get_cube_m_       : cube with yr & doy & ax_t     --> hwmi_cmip5_
                                                        hwmi_cordex_
* get_cube0_m_      : cubes of ref. & inv.          --> hwmi_cmip5_
                                                        hwmi_cordex_
* inloop_rg_hwmi_m_ : care for model data           --> hwmi_cmip5_
                                                        hwmi_cordex_
* hwmi_cmip5_       : hwmi for cmip5 data
* hwmi_cordex_      : hwmi for cordex data

* main              : with controlfile
                    : >>> cube >>> _

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 10.10.2019
Date last modified: 15.05.2020
"""


from climidx import hwmi__, hwmid__
from uuuu import *

import numpy as np
import iris
import os
import time
import math
import warnings
import logging
import yaml
import argparse

from typing import Hashable
from time import localtime, strftime


__all__ = ['hwmi_obs_',
           'hwmi_cmip5_',
           'hwmi_cordex_',
           'hwmi_other_']


def dictdict_(cfg, fn_):
    dd__ = {}
    mdir = '{}{}/med/'.format(cfg['root'], cfg['experiment'])
    rref = rPeriod_(cfg['p_']['ref'])
    minL = cfg['minL'] if 'minL' in cfg else 3
    pctl = cfg['thr_pctl'] if 'thr_pctl' in cfg else 90
    fnthr = '{}_{}_{}_{}.nc'.format('thr', fn_, rref, pctl)
    mtd = cfg['data4kde_mtd'] if 'data4kde_mtd' in cfg else 'ymax'
    fnkde = '{}_{}_{}_{}-{}.npz'.format('kde', fn_, rref, pctl, mtd)
    kdeo = cfg['kde_opts'] if 'kde_opts' in cfg else {}
    dd__.update({'dict_p': cfg['p_'],
                 'mdir': mdir,
                 'rref': rref,
                 'minL': minL,
                 'pctl': pctl,
                 'fnthr': fnthr,
                 'dCube': (),
                 'rCube': (),
                 'pn': 'data'})
    if not cfg['_d']:
        dd__.update({'mtd': mtd,
                     'fnkde': fnkde,
                     'kdeo': kdeo})
    return dd__


def _cmip5_dirs(cfg, gcm, rip):
    try:
        freq = cfg['f_opts']['freq']
    except KeyError:
        freq = 'day'
    incl = [gcm, rip]
    B_ = 'rdir0' in cfg and 'ddir0' in cfg
    if B_:
        cfg.update({'rdir': cfg['rdir0'], 'ddir': cfg['ddir0']})
        return '_'.join(incl)
    else:
        o0, o1 = _cmip5_dir(incl + [cfg['ehr']])
        if o1:
            o0 += '{}/'.format(freq)
            if cfg['ehr'] == 'historical':
                cfg.update({'rdir': o0, 'ddir': o0})
            elif cfg['ehr'][:3] == 'rcp':
                o0_ = o0.replace(cfg['ehr'], 'historical')
                cfg.update({'rdir': [o0_, o0], 'ddir': o0})
            else:
                raise Exception("{!r} invalid for 'ehr'".format(cfg['ehr']))
            return o1
        else:
            return None


def _cmip5_dir(incl, dL=None):
    if dL is None:
        dL = bi_cmip5_info_('/home/rossby/imports/cmip5/')
    out = slctStrL_(dL, incl=incl)
    out.sort()
    if len(out) == 0:
        return (None, None)
    else:
        tmp = path2cmip5_info_(out[-1])
        fn = '_'.join([tmp['gcm'], tmp['rip']])
        return (out[0], fn)


def _cordex_dirs(cfg, gcm, rcm, rip):
    _dir = _smhi_dir if rcm == 'SMHI-RCA4' else _imp_dir
    try:
        freq = cfg['f_opts']['freq']
    except KeyError:
        freq = 'day'
    try:
        version = cfg['r_v'][rcm]
    except KeyError:
        version = None
    incl = [gcm, rip, rcm, version] if version else [gcm, rip, rcm]
    B_ = 'rdir0' in cfg and 'ddir0' in cfg
    if B_:
        cfg.update({'rdir': cfg['rdir0'], 'ddir': cfg['ddir0']})
        return '_'.join(incl)
    else:
        o0, o1 = _dir(incl + [cfg['ehr']])
        if o1:
            o0 += '{}/'.format(freq)
            if cfg['ehr'] in ['evaluation', 'historical']:
                cfg.update({'rdir': o0, 'ddir': o0})
            elif cfg['ehr'][:3] == 'rcp':
                o0_ = '{}{}/'.format(_dir(incl + ['historical'])[0], freq)\
                      if rcm == 'SMHI-RCA4'\
                      else o0.replace(cfg['ehr'], 'historical')
                cfg.update({'rdir': [o0_, o0], 'ddir': o0})
            else:
                raise Exception("{!r} invalid for 'ehr'".format(cfg['ehr']))
            return o1
        else:
            return None


def _imp_dir(incl, dL=None):
    if dL is None:
        dL = bi_cordex_info_('/home/rossby/imports/cordex/EUR-11/')
    out = slctStrL_(dL, incl=incl)
    out.sort()
    if len(out) == 0:
        return (None, None)
    else:
        tmp = path2cordex_info_(out[-1])
        fn = '_'.join([tmp['gcm'], tmp['rip'], tmp['rcm'], tmp['version']])
        return (out[-1], fn)


def _smhi_dir(incl, sY=None):
    if sY is None:
        with open('api/eur-11_smhi-rca4.yml') as yf:
            sY = yaml.safe_load(yf)
    out = [i for i in sY if isinstance(i, int) and
           all([ii in sY[i].values() for ii in incl])]
    if len(out) == 0:
        return (None, None)
    else:
        tmp = sY[out[-1]]
        fn = '_'.join([tmp['gcm'], tmp['rip'], tmp['rcm'], tmp['version']])
        return ('{}{}/netcdf/'.format(sY['root'], out[-1]), fn)


def _var(cfg, hORc):
    try:
        out = cfg['f_opts']['var'][0] + cfg['f_opts']['var'][-1]
    except KeyError:
        out = 'tn' if hORc == 'cold' else 'tx'
    return out


def mk_odir_(cfg):
    odir = '{}{}/res/{}/{}/'.format(cfg['root'], cfg['experiment'],
                                  cfg['proj'], cfg['thr_pctl'])
    if not cfg['_d']:
        mtd = cfg['data4kde_mtd'] if 'data4kde_mtd' in cfg else 'ymax'
        odir += '{}/'.format(mtd)
    if 'ehr' in cfg:
        odir += '{}/'.format(cfg['ehr'])
    os.makedirs(odir, exist_ok=True)
    return odir


def check_med_f_(dd__, _d=False):
    if _d:
        return False
        #return os.path.isfile(dd__['mdir'] + dd__['fnthr'])
    else:
        return os.path.isfile(dd__['mdir'] + dd__['fnkde']) and \
               os.path.isfile(dd__['mdir'] + dd__['fnthr'])


def inloop_func_(hORc, dd__, _d=False):
    f__ = hwmid__ if _d else hwmi__
    if hORc[:4] == 'heat':
       return f__(**dd__)
    elif hORc[:4] == 'cold':
       return f__(**dd__, hw=False)


def tof_(out, odir, fn__, hORc, _d=False, _sdi=True):
    var_ = 'hwmi' if hORc[:4] == 'heat' else 'cwmi'
    iris.save(out['hwmi'], '{}{}{}-{}.nc'.format(odir, var_, 'd' * _d, fn__))
    if _sdi:
        var_ = 'wsdi' if hORc[:4] == 'heat' else 'csdi'
        iris.save(out['wsdi'], '{}{}-{}.nc'.format(odir, var_, fn__))


def frf_(odir, fn__, hORc, _d=False, _sdi=True):
    def _frf(vv, dd):
        o0 = iris.load('{}{}{}-{}_*.nc'.format(odir, vv, 'd' * dd, fn__))
        if len(o0) > 2:
            purefy_cubeL_(o0)
            try:
                o0 = o0.merge_cube()
            except:
                try:
                    o0 = o0.concatenate_cube()
                except:
                    ll_('merge_cube() or concatenate_cube() failure!')
            iris.save(o0, '{}{}{}-{}.nc'.format(odir, vv, 'd' * dd, fn__))
            for i in schF_keys_(odir, '{}{}-{}_'.format(vv, 'd' * dd, fn__)):
                os.remove(i)
    var_ = 'hwmi' if hORc[:4] == 'heat' else 'cwmi'
    _frf(var_, _d)
    if _sdi:
        var_ = 'wsdi' if hORc[:4] == 'heat' else 'csdi'
        _frf(var_, False)


def k_get_(cfg, hORc):
    kGet = {'season': cfg['season']} if 'season' in cfg else {}
    kGet.update({'hORc': hORc})
    return kGet


def get_cube_obs_(idir, dn, rn, dict_rg, *fn, season=None, hORc='heat',
                  keys=[]):
    from iris.experimental.equalise_cubes import equalise_attributes

    warnings.filterwarnings("ignore",
                            message="Missing CF-netCDF measure variable")

    t0 = l__('loading {}'.format(dn))
    if len(fn) != 0:
        intersect_it = any([rn not in i for i in fn])
        fn = [idir + i for i in fn]
    else:
        fn = schF_keys_(idir, dn, rn, *keys)
        intersect_it = False
        if len(fn) == 0:
            fn = schF_keys_(idir, dn, *keys)
            if len(fn) == 0:
                raise Exception("fail to find data at {!r}".format(idir))
            intersect_it = True

    if len(fn) > 1:
        tmp = iris.load(fn)
        equalise_attributes(tmp)
        cube = tmp.concatenate_cube()
    else:
        cube = iris.load_cube(fn[0])

    tmp = schF_keys_(idir, dn, 'sftlf')
    if len(tmp) != 0:
        sftlf = iris.load_cube(tmp[0])
        maskLS_cube(cube, sftlf, thr=50)

    intersect_it &= (rn.upper() not in ('GLOBAL', 'ALL') and rn in dict_rg)

    if intersect_it:
        cube = intersection_(cube, **dict_rg[rn])
    if season:
        cube = extract_season_cube(cube, season)
        seasonyr_cube(cube, season)
    elif hORc[:4] == 'cold':
        seasonyr_cube(cube, 'jasondj')

    yr_doy_cube(cube)
    yr_0 = cube.coord('year').points if season is None \
           and hORc[:4] == 'heat' else cube.coord('seasonyr').points
    doy_0 = cube.coord('doy').points
    ax_t = cube.coord_dims('time')[0]
    ll_('loading {}'.format(dn), t0)

    return (cube, yr_0, doy_0, ax_t)


def hwmi_obs_(cfg):
    """
    Purpose: hwmi for observational data
    """
    odir = mk_odir_(cfg)
    hORc = cfg['hORc'] if 'hORc' in cfg else 'heat'
    _d = cfg['_d'] if '_d' in cfg else False
    var = _var(cfg, hORc)
    for dn in cfg['datasets']:
        logging.info(' >>>>>>> {}'.format(dn))
        for rn in cfg['regions']:
            fn_ = '_'.join((var, dn, rn))
            dd__ = dictdict_(cfg, fn_)
            #file name(s)
            if 'ifn' in cfg:
                if isinstance(cfg['ifn'][dn], Hashable):
                    fnL = {cfg['ifn'][dn]}
                else:
                    fnL = set(cfg['ifn'][dn])
            else:
                fnL = ()
            #taking care of season
            kGet = k_get_(cfg, hORc)
            kGet_ = dict(keys=cfg['f_opts'].values()) if 'f_opts' in cfg\
                    else dict()
            #reading dCube
            try:
                dCube = get_cube_obs_(cfg['idir'], dn, rn, cfg['sub_r'],
                                      *fnL, **kGet, **kGet_)
            except:
                continue
            #reading rCube
            if check_med_f_(dd__, _d):
                rCube = ()
            else:
                rCube = get_cube_obs_(cfg['idir'], dn, rn, cfg['sub_r'],
                                      *fnL, **kGet_)
            #updating dd__
            dd__.update({'dCube': dCube, 'rCube': rCube})
            #preparing output names
            fn__ = '_'.join((fn_, 'ref{}'.format(rPeriod_(cfg['p_']['ref']))))
            if 'season' in cfg:
                 fn__ += '_{}'.format(cfg['season'])
            if 'periods' in cfg:
                for pn in cfg['periods']:
                    dd__.update({'pn': pn})
                    out = inloop_func_(hORc, dd__, _d)
                    tof_(out, odir, '_'.join((fn__, rPeriod_(cfg['p_'][pn]))),
                         hORc, _d, cfg['_sdi'])
            else:
                out = inloop_func_(hORc, dd__, _d)
                tof_(out, odir, fn__, hORc, _d, cfg['_sdi'])
        logging.info(' {} <<<<<<<'.format(dn))


def get_cube_m_(cube0, rn, dict_rg, *sftlf, season=None, hORc='heat'):
    t0 = l__('precube')
    intersect_it = False if rn.upper() in ['GLOBAL', 'ALL'] \
                   or rn not in dict_rg else True
    if intersect_it:
        cube = intersection_(cube0, **dict_rg[rn])
    else:
        cube = cube0.copy()

    if season is not None:
        cube = extract_season_cube(cube, season)
        seasonyr_cube(cube, season)
    elif hORc[:4] == 'cold':
        seasonyr_cube(cube, 'jasondj')

    if len(sftlf) != 0:
        maskLS_cube(cube, sftlf[0], thr=50)
    yr_doy_cube(cube)
    yr_0 = cube.coord('year').points if season is None \
           and hORc[:4] == 'heat' else cube.coord('seasonyr').points
    doy_0 = cube.coord('doy').points
    ax_t = cube.coord_dims('time')[0]
    ll_('precube', t0)

    return (cube, yr_0, doy_0, ax_t)


def dir_cubeL(mm, *Args, **kArgs):
    """
    Purpose: select function to load model data
    """
    if mm[:5] == 'cmip5':
       return cmip5_dir_cubeL(*Args, **kArgs)
    elif mm[:6] == 'cordex':
       return cordex_dir_cubeL(*Args, **kArgs)
    else:
       raise Exception('unknown dataset name!')


def get_cube0_m_(cfg, mdict):
    """
    Purpose: load model data
    """
    t0 = l__('loading cube0')
    if 'periods' in cfg:
        if cfg['rdir'] == cfg['ddir']:
            p0 = min(flt_l(cfg['p_'].values()))
            p1 = max(flt_l(cfg['p_'].values()))
            tmp = dir_cubeL(cfg['proj'], cfg['ddir'], **cfg['f_opts'],
                            **mdict, period=[p0, p1], ifconcat=True)
            dcube0 = tmp['cube']
            rcube0 = dcube0
        else:
            tmp = dir_cubeL(cfg['proj'], cfg['rdir'], **cfg['f_opts'],
                            **mdict, period=cfg['p_']['ref'], ifconcat=True)
            rcube0 = tmp['cube']
            pL = [i for i in cfg['p_'].values()]
            p0 = min(flt_l(pL[1:]))
            p1 = max(flt_l(pL[1:]))
            tmp = dir_cubeL(cfg['proj'], cfg['ddir'], **cfg['f_opts'],
                            **mdict, period=[p0, p1], ifconcat=True)
            dcube0 = tmp['cube']
    else:
        tmp = dir_cubeL(cfg['proj'], cfg['rdir'], **cfg['f_opts'], **mdict,
                        period=cfg['p_']['ref'], ifconcat=True)
        rcube0 = tmp['cube'] if tmp else tmp
        #tmp = dir_cubeL(cfg['proj'], cfg['ddir'], **cfg['f_opts'], **mdict)
        tmp = dir_cubeL(cfg['proj'], cfg['ddir'], **cfg['f_opts'], **mdict,
                        ifconcat=True)
        dcube0 = tmp['cube'] if tmp else tmp
    ll_('loading cube0', t0)
    return (rcube0, dcube0)


def inloop_rg_hwmi_m_(cfg, hORc, rcube0, dcube0, rn, odir, fn_, _d, *sftlf):
    """
    Purpose: in loop region-loop calculate and save results
    """
    kGet = k_get_(cfg, hORc)
    fn__ = '_'.join((fn_, 'ref{}'.format(rPeriod_(cfg['p_']['ref']))))
    if 'season' in cfg:
         fn__ += '_{}'.format(cfg['season'])
    t0 = l__(fn__)
    dd__ = dictdict_(cfg, fn_)
    if check_med_f_(dd__, _d):
        rCube = ()
    else:
        rCube = get_cube_m_(rcube0, rn, cfg['sub_r'], *sftlf)
    dd__.update({'rCube': rCube})
    if 'periods' in cfg:
        dCube = get_cube_m_(dcube0, rn, cfg['sub_r'], *sftlf, **kGet)
        dd__.update({'dCube': dCube})
        for pn in cfg['periods']:
            dd__.update({'pn': pn})
            out = inloop_func_(hORc, dd__, _d)
            tof_(out, odir, '_'.join((fn__, rPeriod_(cfg['p_'][pn]))),
                 hORc, _d, cfg['_sdi'])
    elif isinstance(dcube0, iris.cube.Cube):
        dCube = get_cube_m_(dcube0, rn, cfg['sub_r'], *sftlf, **kGet)
        dd__.update({'dCube': dCube})
        out = inloop_func_(hORc, dd__, _d)
        tof_(out, odir, fn__, hORc, _d, cfg['_sdi'])
    else:
        c_h = []
        c_w = []
        for i, cc in enumerate(dcube0):
            t00 = l__(prg_(i, len(dcube0)))
            dCube = get_cube_m_(cc, rn, cfg['sub_r'], *sftlf, **kGet)
            dd__.update({'dCube': dCube})
            tmp = inloop_func_(hORc, dd__, _d)
            c_h.append(tmp['hwmi'])
            c_w.append(tmp['wsdi'])
            ll_(prg_(i, len(dcube0)), t00)
        c_h = iris.cube.CubeList(c_h)
        c_h = concat_cube_(c_h)
        c_w = iris.cube.CubeList(c_w)
        c_w = concat_cube_(c_w)
        out = {'hwmi': c_h, 'wsdi': c_w}
        tof_(out, odir, fn__, hORc, _d, cfg['_sdi'])
    ll_(fn__, t0)

def hwmi_cmip5_(cfg):
    """
    Purpose: hwmi for cmip5 data
    """
    odir = mk_odir_(cfg)
    hORc = cfg['hORc'] if 'hORc' in cfg else 'heat'
    _d = cfg['_d'] if '_d' in cfg else False
    var = _var(cfg, hORc)
    warnings.filterwarnings("ignore", message="Missing CF-netCDF ")
    for gcm in cfg['gcms']:
        try:
            rips = cfg['rip'][gcm]
        except KeyError:
            rips = ['r1i1p1']
        rips = [rips] if isinstance(rips, str) else rips
        for rip in rips:
            logging.info(' >>>>>>> {}_{}'.format(gcm, rip))
            fn0 = _cmip5_dirs(cfg, gcm, rip)
            if fn0 is None:
                ll_(' XXX {}_{}'.format(gcm, rip))
                continue
            rcube0, dcube0 = get_cube0_m_(cfg, dict(gcm=gcm, realz=rip))
            if rcube0 is None or dcube0 is None:
                ll_(' XXX {}_{}'.format(gcm, rip))
                continue
            if isinstance(dcube0, iris.cube.CubeList):
                if len(dcube0) == 0:
                    ll_(' XXX {}_{}'.format(gcm, rip))
                    continue
            linfo = cmip5_dir_finfo(cfg['ldir'], var='sftlf', gcm=gcm)
            if len(linfo['fn']) != 0:
                sftlf = (iris.load_cube(linfo['fn'][0]),)
            else:
                sftlf = ()
            for rn in cfg['regions']:
                fn_ =  '_'.join((var, fn0, rn))
                inloop_rg_hwmi_m_(cfg, hORc, rcube0, dcube0, rn, odir, fn_,
                                  _d, *sftlf)
            logging.info(' {}_{} <<<<<<<'.format(gcm, rip))


def hwmi_cordex_(cfg):
    """
    Purpose: hwmi for cordex data
    """
    odir = mk_odir_(cfg)
    hORc = cfg['hORc'] if 'hORc' in cfg else 'heat'
    _d = cfg['_d'] if '_d' in cfg else False
    var = _var(cfg, hORc)
    warnings.filterwarnings("ignore", message="Missing CF-netCDF ")
    for rcm in cfg['rcms']:
        t0_ = l__(' >>>>>>>>> {}'.format(rcm))
        for gcm in cfg['gcms']:
            t0__ = l__(' >>>>>>> {}'.format(gcm))
            try:
                rips = cfg['rip'][gcm]
            except KeyError:
                rips = ['r1i1p1']
            rips = [rips] if isinstance(rips, str) else rips
            for rip in rips:
                t0___ = l__(' >>>>> {}'.format(rip))
                fn0 = _cordex_dirs(cfg, gcm, rcm, rip)
                if fn0 is None:
                    ll_(' XXX {}_{}_{}'.format(gcm, rip, rcm))
                    continue
                rcube0, dcube0 = get_cube0_m_(cfg, dict(gcm=gcm, rcm=rcm,
                                                        realz=rip))
                if rcube0 is None or dcube0 is None:
                    ll_(' XXX {}_{}'.format(gcm, rip))
                    continue
                if isinstance(dcube0, iris.cube.CubeList):
                    if len(dcube0) == 0:
                        ll_(' XXX {}_{}_{}'.format(gcm, rip, rcm))
                        continue
                linfo = cordex_dir_finfo(cfg['ldir'], var='sftlf', rcm=rcm)
                if len(linfo['fn']) != 0:
                    sftlf = (iris.load_cube(linfo['fn'][0]),)
                else:
                    sftlf = ()
                for rn in cfg['regions']:
                    fn_ =  '_'.join((var, fn0, rn))
                    inloop_rg_hwmi_m_(cfg, hORc, rcube0, dcube0, rn, odir,
                                      fn_, _d, *sftlf)
                ll_(' {} <<<<<'.format(rip), t0___)
            ll_(' {} <<<<<<<'.format(gcm), t0__)
        ll_(' {} <<<<<<<<<'.format(rcm), t0_)


def _realzL(cube):
    if 'realization' in [i.name() for i in cube.dim_coords]:
        nr = cube.shape[cube.coord_dims('realization')[0]]
        return [extract_byAxes_(cube,'realization', np.s_[i,])
                for i in range(nr)]
    else:
        return cube


def get_cube0_o_(cfg, dn):
    """
    Purpose: load 'other' data
    """
    warnings.filterwarnings("ignore", message="Missing CF-netCDF ")

    logging.debug('get_cube0_o_')
    t0 = l__('loading cube0')
    keys = (cfg['f_opts']['var'] + '_',)\
           if 'f_opts' in cfg and 'var' in cfg['f_opts'] else\
           (('tasmin_',) if cfg['hORc'] == 'cold' else ('tasmax_'))
    fnL = schF_keys_(cfg['idir'] + dn + '/', *keys, ext='.nc')
    o = iris.load(fnL, 'air_temperature')
    out0 = _realzL(o[0]) if len(o) == 1 else\
           [iris.load_cube(i, 'air_temperature') for i in fnL]

    if len(fnL) > 1:
        out2 = [pure_fn_(i) for i in fnL]
    elif len(fnL) == 1 and not isinstance(out0, iris.cube.Cube):
        out2 = ['{}'.format(i) for i in range(len(out0))]
    else:
        out2 = None

    tmp = schF_keys_('{}{}/'.format(cfg['idir'], dn), 'sftlf')
    if len(tmp) != 0:
        out1 = (iris.load_cube(tmp[0]),)
    else:
        out1 = ()

    ll_('loading cube0')

    return (out0, out1, out2)


def cubeORcubeL_hwmi_(cfg, odir, hORc, o0, rn, fn_, _d, pn='data'):
    from iris.experimental.equalise_cubes import equalise_attributes
    kGet = k_get_(cfg, hORc)
    if 'season' in cfg:
        fn__ = '{}_{}'.format(fn_, cfg['season'])
    else:
        fn__ = fn_
    t0 = l__(fn__)
    if isinstance(o0[0], iris.cube.Cube):
        dd__ = dictdict_(cfg, fn_)
        if check_med_f_(dd__, _d):
            rCube = ()
        else:
            rCube = get_cube_m_(o0[0], rn, cfg['sub_r'], *o0[1])
        dCube = get_cube_m_(o0[0], rn, cfg['sub_r'], *o0[1], **kGet)
        dd__.update({'dCube': dCube, 'rCube': rCube, 'pn': pn})
        out = inloop_func_(hORc, dd__, _d)
        tof_(out, odir, fn__, hORc, _d, cfg['_sdi'])
    else:
        c_h = []
        c_w = []
        for i, cc in enumerate(o0[0]):
            #if i :#QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
            #    continue
            t00 = l__(prg_(i, len(o0[0])))
            dd__ = dictdict_(cfg, '_'.join((fn_, o0[2][i])))
            if check_med_f_(dd__, _d):
                rCube = ()
            else:
                rCube = get_cube_m_(cc.copy(), rn, cfg['sub_r'], *o0[1])
            dCube = get_cube_m_(cc.copy(), rn, cfg['sub_r'], *o0[1], **kGet)
            dd__.update({'dCube': dCube, 'rCube': rCube, 'pn': pn})
            tmp = inloop_func_(hORc, dd__, _d)
            tof_(tmp, odir, '_'.join((fn__, o0[2][i])), hORc, _d, cfg['_sdi'])
            ll_(prg_(i, len(o0[0])), t00)
        frf_(odir, fn__, hORc, _d, cfg['_sdi'])
        #    c_h.append(tmp['hwmi'])
        #    c_w.append(tmp['wsdi'])
        #    ll_(prg_(i, len(o0[0])), t00)
        #c_h = iris.cube.CubeList(c_h)
        #c_w = iris.cube.CubeList(c_w)
        #purefy_cubeL_(c_h)
        #purefy_cubeL_(c_w)
        #try:
        #    c_h = c_h.merge_cube()
        #    c_w = c_w.merge_cube()
        #except:
        #    try:
        #        c_h = c_h.concatenate_cube()
        #        c_w = c_w.concatenate_cube()
        #    except:
        #        ll_(' merge_cube() or concatenate_cube() failure!')
        #out = {'hwmi': c_h, 'wsdi': c_w}
        #tof_(out, odir, fn__, hORc, _d, cfg['_sdi'])
    ll_(fn__)


def hwmi_other_(cfg):
    """
    Purpose: hwmi for cordex data
    """
    odir = mk_odir_(cfg)
    hORc = cfg['hORc'] if 'hORc' in cfg else 'heat'
    _d = cfg['_d'] if '_d' in cfg else False
    var = _var(cfg, hORc)
    for dn in cfg['datasets']:
        logging.info(' >>>>>>> {}'.format(dn))
        o0 = get_cube0_o_(cfg, dn)
        if o0[0] is None:
            continue
        for rn in cfg['regions']:
            fn_ = '_'.join((var, dn, rn, 'ref' + rPeriod_(cfg['p_']['ref'])))
            if 'periods' in cfg:
                for pn in cfg['periods']:
                    fn__ = '_'.join((fn_, rPeriod_(cfg['p_'][pn])))
                    cubeORcubeL_hwmi_(cfg, odir, hORc, o0, rn, fn__, _d,
                                      pn=pn)
            else:
                cubeORcubeL_hwmi_(cfg, odir, hORc, o0, rn, fn_, _d)
        logging.info(' {} <<<<<<<'.format(dn))


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser('derive hwmi')
    parser.add_argument("controlfile",
                        help="yaml file with metadata")
    args = parser.parse_args()
    with open(args.controlfile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    nlog = len(schF_keys_('', cfg['proj'], ext='.log'))
    logging.basicConfig(filename= cfg['proj'] + '_'*nlog + '.log',
                        filemode='w',
                        level=eval('logging.' + cfg['dbgl']))
    logging.info(' {:_^42}'.format('start of program'))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))
    logging.info(' ')
    if cfg['proj'] == 'obs':
        rrrr__ = hwmi_obs_
    elif cfg['proj'][:5] == 'cmip5':
        rrrr__ = hwmi_cmip5_
    elif cfg['proj'][:6] == 'cordex':
        rrrr__ = hwmi_cordex_
    elif cfg['proj'] == 'other':
        rrrr__ = hwmi_other_
    else:
        raise Exception("{!r} is unknown as 'proj'".format(cfg['proj']))
    rrrr__(cfg)

if __name__ == '__main__':
    start_time = time.time()
    main()
    logging.info(' ')
    logging.info(' {:_^42}'.format('end of program'))
    logging.info(' {:_^42}'.format('TOTAL'))
    logging.info(' ' + rTime_(time.time() - start_time))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))


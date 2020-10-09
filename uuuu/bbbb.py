"""
>--#¤&#¤%/%&(¤%¤#%"¤#"#¤%&¤#!"#%#%&()/=?=()/&%¤%&/()/£$€¥{[}]()=?=)(/&¤%&/()--<
>--------------------functions dealing with BI directories--------------------<
>--#¤&&()/=?=()/&%¤%&/()/(/&%&/(/()=?=)(/&¤%&/("%"¤#&#&¤&&/(%&"#¤"¤%&#%/)/*#--<
* cmip5_dir_finfo       : file info of a directory: cmip5
* cordex_dir_finfo      : file info of a directory: cordex
* min_fselect_          : select files according specified period
* cmip5_dir_cubeL       : load cube list from a directory: cmip5
* cordex_dir_cubeL      : load cube list from a directory: cordex
* load_res_             : load cube in hwmi result directory
* load_fx_              : load 'sftlf' & 'areacella'
* en_mm_                : ensemble (different models) cube
* bi_cordex_info_       : get cordex data achieve in bi
* path2cordex_info_     : cordex info from string of path
...

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 06.10.2019
Date last modified: 24.09.2020
          comments: some updates dealing with missing of EOBS or ERAI
"""


import warnings
import glob
import os
import iris
import numpy as np

from .ffff import intsect_, ouniqL_, schF_keys_, valueEqFront_, p_least_, \
                  p_deoverlap_, pure_fn_, latex_unit_, l_ind_
from .cccc import concat_cube_, intersection_, en_rip_, rgMean_cube, \
                  repair_lccs_, en_mm_cubeL_, extract_byAxes_, maskNaN_cube, \
                  rgMean_poly_cube, extract_period_cube, en_mean_, en_iqr_, \
                  rm_sc_cube


__all__ = ['bi_cmip5_info_',
           'bi_cordex_info_',
           'clmidx_finfo_',
           'clmidx_finfo_cmip5_',
           'clmidx_finfo_eval_',
           'cmip5_dir_cubeL',
           'cmip5_dir_finfo',
           'cordex_dir_cubeL',
           'cordex_dir_finfo',
           'en_mm_',
           'en_pmean_cubeL_',
           'get_clm_clmidx_',
           'get_clm_eval_',
           'get_ts_clmidx_',
           'get_ts_eval_',
           'load_clmidx_',
           'load_clmidx_eval_',
           'load_fx_',
           'load_res_',
           'map_sim_',
           'map_sim_cmip5_',
           'min_fselect_',
           'path2cmip5_info_',
           'path2cordex_info_',
           'pure_ts_dn_',
           'sc_unit_clmidx_',
           'slct_cubeLL_dnL_',
           'version_up_',
           'version_up_cubeLL_']


def cmip5_dir_finfo(idir, var='*', freq='*', gcm='*', exp='*', realz='*',
                    p=None, ext='.nc'):
    """
    Purpose: get data info from a cmip5 data directory
    """
    varL, freqL, gcmL, expL, realzL, pL = [[] for _ in range(6)]
    s = '_'
    if isinstance(idir, str):
        idir = [idir]
    files = []
    for i in idir:
        if p is None:
            tmp = glob.glob(i + s.join([var, freq, gcm, exp, realz]) +
                            '*' + ext)
        else:
            tmp = glob.glob(i + s.join([var, freq, gcm, exp, realz, p]) + ext)
        files += tmp
    files.sort()
    for f in files:
        for i in idir:
            f = f.replace(i, '')
        s_s = f.replace(ext, '').split(s)
        varL.append(s_s[0])
        freqL.append(s_s[1])
        gcmL.append(s_s[2])
        expL.append(s_s[3])
        realzL.append(s_s[4])
        if s_s[1].upper() != 'FX':
            pL.append(s_s[5])
    return {'var': ouniqL_(varL), 'freq': ouniqL_(freqL),
            'gcm': ouniqL_(gcmL), 'exp': ouniqL_(expL),
            'realz': ouniqL_(realzL), 'p': ouniqL_(pL),
            'fn': files}


def cordex_dir_finfo(idir, var='*', dm='*', gcm='*', exp='*', realz='*',
                     rcm='*', V='*', freq='*', p=None, ext='.nc'):
    """
    Purpose: get data info from a cordex data directory
    """
    varL, dmL, gcmL, expL, realzL, rcmL, VL, freqL, pL = [[]
                                                          for _ in range(9)]
    s = '_'
    if isinstance(idir, str):
        idir = [idir]
    files = []
    for i in idir:
        if p is None:
            tmp = glob.glob(i + s.join([var, dm, gcm, exp, realz, rcm, V,
                                        freq]) + '*' + ext)
        else:
            tmp = glob.glob(i + s.join([var, dm, gcm, exp, realz, rcm, V,
                                        freq, p]) + ext)
        files += tmp
    files.sort()
    for f in files:
        for i in idir:
            f = f.replace(i, '')
        s_s = f.replace(ext, '').split(s)
        varL.append(s_s[0])
        dmL.append(s_s[1])
        gcmL.append(s_s[2])
        expL.append(s_s[3])
        realzL.append(s_s[4])
        rcmL.append(s_s[5])
        VL.append(s_s[6])
        freqL.append(s_s[7])
        if s_s[7].upper() != 'FX':
            pL.append(s_s[8])
    return {'var': ouniqL_(varL), 'dm': ouniqL_(dmL),
            'gcm': ouniqL_(gcmL), 'exp': ouniqL_(expL),
            'realz': ouniqL_(realzL), 'rcm': ouniqL_(rcmL),
            'V': ouniqL_(VL), 'freq': ouniqL_(freqL),
            'p': ouniqL_(pL), 'fn': files}


def min_fselect_(dir_finfo, period=None):
    """
    Purpose: select fewest files from finfo of a cmip5/cordex data directory
    """
    pp = p_deoverlap_(p_least_(dir_finfo['p'], *period) if period else
                      dir_finfo['p'])
    fn = [f for f in dir_finfo['fn'] if any(i for i in pp if i in f)]
    dir_finfo.update({'p': pp, 'fn': fn})


def cmip5_dir_cubeL(idir, var='*', freq='*', gcm='*', exp='*', realz='*',
                    p='*', ext='.nc', period=None, ifconcat=False):
    """
    Purpose: load cube list from a cmip5 data directory
    """
    warnings.filterwarnings("ignore", message="Missing CF-netCDF measure")
    s = '_'
    info = cmip5_dir_finfo(idir, var=var, freq=freq, gcm=gcm, exp=exp,
                           realz=realz, p=p, ext=ext)
    if (len(info['gcm']) * len(info['var']) * len(info['freq'])
        * len(info['realz'])) not in [0, 1] and ifconcat:
        raise Exception("no idea how to organize kArgs!")
    min_fselect_(info, period)
    cubeL = iris.load(info['fn'])
    if var != '*':
        varCstr_ = iris.Constraint(cube_func=lambda c: c.var_name == var)
        cubeL = cubeL.extract(varCstr_)
    if len(cubeL) == 0:
        return None
    elif ifconcat:
        cube = concat_cube_(cubeL, thr=1e-5)
        p = sorted(info['p'])
        return {'cube': cube,
                'p': '-'.join((p[0].split('-')[0], p[-1].split('-')[-1]))}
    else:
        return {'cube': cubeL, 'p': info['p']}


def cordex_dir_cubeL(idir, var='*', dm='*', gcm='*', exp='*', rcm='*',
                     freq='*', V='*', realz='*', p='*', ext='.nc',
                     period=None, ifconcat=False):
    """
    Purpose: load cube list from a cordex data directory
    """
    warnings.filterwarnings("ignore", message="Missing CF-netCDF ")
    s = '_'
    info = cordex_dir_finfo(idir, var=var, dm=dm, gcm=gcm, exp=exp,
                            realz=realz, rcm=rcm, p=p, ext=ext)
    if (len(info['dm']) * len(info['gcm']) * len(info['var'])
        * len(info['rcm']) * len(info['freq'] * len(info['realz'])
        * len(info['V']))) not in [0, 1] and ifconcat:
        raise Exception("no idea how to organize kArgs!")
    min_fselect_(info, period)
    cubeL = iris.load(info['fn'])
    if var != '*':
        varCstr_ = iris.Constraint(cube_func=lambda c: c.var_name == var)
        cubeL = cubeL.extract(varCstr_)
    if len(cubeL) == 0:
        return None
    elif ifconcat:
        cube = concat_cube_(cubeL)
        p = sorted(info['p'])
        return {'cube': cube,
                'p': '-'.join((p[0].split('-')[0], p[-1].split('-')[-1]))}
    else:
        return {'cube': cubeL, 'p': info['p']}


def load_res_(res_d, v, dn, rn, rn_dict):
    """
    Purpose: load single cube of hwmi
    """
    fn = schF_keys_(res_d, dn, rn, v)
    if len(fn) == 1:
        cube = iris.load_cube(fn[0])
    elif len(fn) > 1:
        raise Exception('multiple files found!')
    else:
        fn = schF_keys_(res_d, dn, v)
        if len(fn) == 1:
            tmp = iris.load_cube(fn[0])
            cube = intersection_(tmp, **rn_dict[rn])
        elif len(fn) > 1:
            raise Exception('multiple files found!')
        else:
            raise Exception('no file found!')
    return cube


def load_fx_(fxdir, dn):
    """
    Purpose: load fx cubes 'sftlf' & 'areacella' required by rgMean_cube()
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    if not isinstance(dn, (set, tuple, list)):
        rgm_opts = {}
        fn = schF_keys_(fxdir, '_{}_'.format(dn), 'sftlf')
        if len(fn) > 0:
            o = iris.load_cube(fn[0])
            repair_lccs_(o)
            rgm_opts.update({'sftlf': o})
        fn = schF_keys_(fxdir, '_{}_'.format(dn), 'areacella')
        if len(fn) > 0:
            o = iris.load_cube(fn[0])
            repair_lccs_(o)
            rgm_opts.update({'areacella': o})
        return rgm_opts
    else:
        return [load_fx_(fxdir, i) for i in dn]


def en_mm_(ddir, dns, cfg, rmp=True, rmpm=None, ref=None):
    """
    Purpose: ensemble cube from a direcory over different datasets
    """
    if ref:
        dns = valueEqFront_(dns, ref)
    if rmp and rmpm is None:
        rmpm = iris.analysis.Linear(extrapolation_mode='mask')
    cubeL = []
    for i, dn in enumerate(dns):
        cube = load_res_(ddir, cfg['v'], dn, cfg['rn'], cfg['sub_r'])
        cube.attributes = {}
        if i == 0:
            tmp = cube.copy()
            new_coord = iris.coords.AuxCoord(np.int32(i),
                                             long_name='realization',
                                             units='no_unit')
            tmp.add_aux_coord(new_coord)
            a = tmp.copy()
        else:
            if tmp.shape != cube.shape:
                if rmp:
                    cube = cube.regrid(tmp, rmpm)
                else:
                    raise Exception('unmatched cubes!')
            a = tmp.copy(cube.data)
            a.coord('realization').points = np.int32(i)
        cubeL.append(a)
    cubeL = iris.cube.CubeList(cubeL)
    eCube = cubeL.merge_cube()
    return eCube


def bi_cmip5_info_(root, o_o='list'):
    if o_o == 'list':
        o = []
    elif o_o == 'dict':
        o = {}
    else:
        raise Exception("output option unknown!"
                        " Either 'list' (default) or 'dict' accepted!")
    gcms = os.listdir(root)
    for gg in gcms:
        if not os.path.isdir(root + gg):
            continue
        rcps = os.listdir(root + gg)
        for rcp in rcps:
            if not os.path.isdir(root + '/'.join((gg, rcp))):
                continue
            rips = os.listdir(root + '/'.join((gg, rcp)))
            for rip in rips:
                if not os.path.isdir(root + '/'.join((gg, rcp, rip))):
                    continue
                if o_o == 'list':
                    o.append(root + '/'.join((gg, rcp, rip, 'input/')))
                else:
                    if gg not in o:
                        o.update({gg: {rcp: {rip: root + '/'.join((gg, rcp,
                                 rip, 'input/'))}}})
                    elif rcp not in o[gg]:
                        o[gg].update({rcp: {rip: root + '/'.join((gg, rcp,
                                     rip, 'input/'))}})
                    elif rip not in o[gg][rcp]:
                        o[gg][rcp].update({rip: root + '/'.join((gg, rcp, rip,
                                          'input/'))})
    return o


def bi_cordex_info_(root, o_o='list'):
    if o_o == 'list':
        o = []
    elif o_o == 'dict':
        o = {}
    else:
        raise Exception("output option unknown!"
                        " Either 'list' (default) or 'dict' accepted!")
    rcms = os.listdir(root)
    for rr in rcms:
        if not os.path.isdir(root + rr):
            continue
        versions = os.listdir(root + rr)
        for vv in versions:
            if not os.path.isdir(root + '/'.join((rr, vv))):
                continue
            gcms = os.listdir(root + '/'.join((rr, vv)))
            for gg in gcms:
                if not os.path.isdir(root + '/'.join((rr, vv, gg))):
                    continue
                rips = os.listdir(root + '/'.join((rr, vv, gg)))
                for rip in rips:
                    if not os.path.isdir(root + '/'.join((rr, vv, gg, rip))):
                        continue
                    rcps = os.listdir(root + '/'.join((rr, vv, gg, rip)))
                    for rcp in rcps:
                        if not os.path.isdir(root + '/'.join((rr, vv, gg, rip,
                                                              rcp))):
                            continue
                        if o_o == 'list':
                            o.append(root + '/'.join((rr, vv, gg, rip, rcp,
                                                      'input/')))
                        else:
                            if rr not in o:
                                o.update({rr: {vv: {gg: {rip: {rcp:
                       root + '/'.join((rr, vv, gg, rip, rcp, 'input/'))}}}}})
                            elif vv not in o[rr]:
                                o[rr].update({vv: {gg: {rip: {rcp:
                       root + '/'.join((rr, vv, gg, rip, rcp, 'input/'))}}}})
                            elif gg not in o[rr][vv]:
                                o[rr][vv].update({gg: {rip: {rcp:
                       root + '/'.join((rr, vv, gg, rip, rcp, 'input/'))}}})
                            elif rip not in o[rr][vv][gg]:
                                o[rr][vv][gg].update({rip: {rcp:
                       root + '/'.join((rr, vv, gg, rip, rcp, 'input/'))}})
                            elif rcp not in o[rr][vv][gg][rip]:
                                o[rr][vv][gg][rip].update({rcp:
                       root + '/'.join((rr, vv, gg, rip, rcp, 'input/'))})
    return o


def path2cmip5_info_(fn):
    terms = [i for i in fn.split('/') if i != '']
    if terms[-1] in ['fx', 'fx_i', '1hr', '3hr', '6hr', 'day', 'mon', 'mon_i',
                     'sem', 'sem_i']:
        gcm, rcp, rip = terms[-5:-2]
        freq = terms[-1]
        return {'gcm': gcm, 'rcp': rcp, 'rip': rip, 'freq': freq}
    else:
        gcm, rcp, rip = terms[-4:-1]
        return {'gcm': gcm, 'rcp': rcp, 'rip': rip}


def path2cordex_info_(fn):
    terms = [i for i in fn.split('/') if i != '']
    if terms[-1] in ['fx', 'fx_i', '1hr', '3hr', '6hr', 'day', 'mon', 'mon_i',
                     'sem', 'sem_i']:
        rcm, version, gcm, rip, rcp = terms[-7:-2]
        freq = terms[-1]
        return {'rcm': rcm, 'version': version, 'gcm': gcm,
                'rip': rip, 'rcp': rcp, 'freq': freq}
    else:
        rcm, version, gcm, rip, rcp = terms[-6:-1]
        return {'rcm': rcm, 'version': version, 'gcm': gcm,
                'rip': rip, 'rcp': rcp}


def version_up_(dnL):
    dnL.sort()
    tmp_ = {}
    for ii in map(lambda i: {'_'.join(i.split('_')[:-1]): i.split('_')[-1]},
                  dnL):
        tmp_.update(ii)
    return list(map(lambda i: '{}_{}'.format(i, tmp_[i]), tmp_))


def slct_cubeLL_dnL_(cubeLL, dnL, func, *Args, **kwArgs):
    udnL = func(dnL, *Args, **kwArgs)
    return ([[i for i, ii in zip(cubeL, dnL) if ii in udnL]
             for cubeL in cubeLL],
            udnL)

def version_up_cubeLL_(cubeLL, dnL):
    udnL = version_up_(dnL)
    return ([[i for i, ii in zip(cubeL, dnL) if ii in udnL]
             for cubeL in cubeLL],
            udnL)


def _clmidx_f2dn(fn):
    tmp = pure_fn_(fn)
    return '_'.join(tmp.split('_')[1:6])


def _clmidx_f2dn_cmip5(fn):
    tmp = pure_fn_(fn)
    return '_'.join(tmp.split('_')[1:4])


def _clmidx_dn_rplrcp(dn, nrcp='historical'):
    tmp = dn.split('_')
    tmp[1] = nrcp
    return '_'.join(tmp)


def clmidx_finfo_(idir, var, gwls=['gwl15', 'gwl2'],
                   gcm='*', rcp='*', rip='*', rcm='*',
                   version='*', rn='EUR', freq='year', newestV=False):
    files = glob.glob(idir + '_'.join((var, gcm, '*', rip, rcm, version,
                                       rn, freq, '*.nc')))
    files.sort()
    fns = [[i for i in files if '{}.nc'.format(ii) in i] for ii in gwls]
    ois = [[_clmidx_f2dn(i) for i in fn] for fn in fns]
    oi = intsect_(*ois)
    if rcp != '*':
        oi = [i for i in oi if rcp in i]
    if newestV:
        oi = version_up_(oi)
    fnc = [idir + '_'.join((var, _clmidx_dn_rplrcp(i), rn, freq,
                            'current.nc')) for i in oi]
    fng = [[i for i, ii in zip(ff, oo) if ii in oi]
           for ff, oo in zip(fns, ois)]
    return ([fnc] + fng, oi)


def clmidx_finfo_cmip5_(idir, var, gwls=['gwl15', 'gwl2'],
                        gcm='*', rcp='*', rip='*', rn='GLB', freq='year'):
    files = glob.glob(idir + '_'.join((var, gcm, '*', rip, rn, freq, '*.nc')))
    files.sort()
    fns = [[i for i in files if '{}.nc'.format(ii) in i] for ii in gwls]
    ois = [[_clmidx_f2dn_cmip5(i) for i in fn] for fn in fns]
    oi = intsect_(*ois)
    if rcp != '*':
        oi = [i for i in oi if rcp in i]
    fnc = [idir + '_'.join((var, _clmidx_dn_rplrcp(i), rn, freq,
                            'current.nc')) for i in oi]
    fng = [[i for i, ii in zip(ff, oo) if ii in oi]
           for ff, oo in zip(fns, ois)]
    return ([fnc] + fng, oi)


def clmidx_finfo_eval_(idir, var, gcm='*', rcp='*', rip='*', rcm='*',
                        version='*', rn='EUR', freq='year'):
    files = glob.glob(idir + '_'.join((var, gcm, rcp, rip, rcm, version,
                                       rn, freq, '*.nc')))
    files.sort()
    oi = [_clmidx_f2dn(i) for i in files]
    dn = ['_'.join(i.split('_')[-2:]) for i in oi]
    if len(dn) > 0:
        feobs = schF_keys_(idir.replace('eval','obs'), var + '_',
                           'EOBS20', freq)
        ferai = schF_keys_(idir.replace('eval','obs'), var + '_',
                           'ERAI', freq)
        if len(feobs) != 1 and len(ferai) != 1:
            files, dn = [], []
        else:
            dn += ['EOBS20','ERA-Interim']
            if len(feobs) == 1:
                files += feobs
            else:
                files.append(None)
            if len(ferai) == 1:
                files += ferai
            else:
                files.append(None)
    return (files, dn)


def load_clmidx_eval_(idir, var, freq='year', rn='EUR', period=[1989, 2008]):
    from iris.fileformats.netcdf import UnknownCellMethodWarning
    warnings.filterwarnings("ignore", category=UnknownCellMethodWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    def _ilc(fn, *Args, **kwArgs):
        if fn:
            o = iris.load_cube(fn, *Args, **kwArgs)
            maskNaN_cube(o)
            repair_lccs_(o)
            if period:
                o = extract_period_cube(o, *period)
            return o
        else:
            return None
    if freq == 'year':
        a, d = clmidx_finfo_eval_(idir, var, freq=freq, rn=rn)
        cubeL0 = [_ilc(i) for i in a]
    elif freq in ['djf', 'mam', 'jja', 'son']:
        a, d = clmidx_finfo_eval_(idir, var, freq=freq, rn=rn)
        if len(d) !=0:
            cubeL0 = [_ilc(i) for i in a]
        else:
            a, d = clmidx_finfo_eval_(idir, var, freq='season', rn=rn)
            cubeL0 = [_ilc(i, iris.Constraint(season=freq)) for i in a]
            if freq == 'djf':
                if period is None:
                    cubeL0 = [extract_byAxes_(i, 'time', np.s_[1:-1])
                              if i else None for i in cubeL0]
                else:
                    cubeL0 = [extract_byAxes_(i, 'time', np.s_[1:])
                              if i else None for i in cubeL0]
    elif freq.capitalize() in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        a, d = clmidx_finfo_eval_(idir, var, freq='month', rn=rn)
        cubeL0 = [_ilc(i, iris.Constraint(month=freq.capitalize()))
                  for i in a]
    else:
        raise Exception('file not found!')
    return (cubeL0, d)


def load_clmidx_(idir, var, gwls=['gwl15', 'gwl2'], freq='year', rn='EUR',
                 folder=None, newestV=False):
    from iris.fileformats.netcdf import UnknownCellMethodWarning
    warnings.filterwarnings("ignore", category=UnknownCellMethodWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    finfo_ = (eval('clmidx_finfo_{}_'.format(folder)) if folder else
              clmidx_finfo_)
    aT = (idir, var)
    kaD = dict(gwls=gwls, freq=freq, rn=rn)
    if folder is None:
        kaD.update(dict(newestV=newestV))
    def _ilc(fn, *Args, **kwArgs):
        o = iris.load_cube(fn, *Args, **kwArgs)
        maskNaN_cube(o)
        repair_lccs_(o)
        return o
    if freq == 'year':
        a, d = finfo_(*aT, **kaD)
        cubeLL = [[_ilc(i) for i in aa] for aa in a]
    elif freq in ['djf', 'mam', 'jja', 'son']:
        a, d = finfo_(*aT, **kaD)
        if len(d) !=0:
            cubeLL = [[_ilc(i) for i in aa] for aa in a]
        else:
            kaD.update(dict(freq='season'))
            a, d = finfo_(*aT, **kaD)
            cubeLL = [[_ilc(i, iris.Constraint(season=freq)) for i in aa]
                      for aa in a]
            if freq == 'djf':
                cubeLL = [[extract_byAxes_(i, 'time', np.s_[1:-1])
                           for i in cubeL0] for cubeL0 in cubeLL]
    elif freq.capitalize() in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        kaD.update(dict(freq='month'))
        a, d = finfo_(*aT, **kaD)
        cubeLL = [[_ilc(i, iris.Constraint(month=freq.capitalize()))
                   for i in aa] for aa in a]
    else:
        raise Exception('file not found!')
    return (cubeLL, d)


def sc_unit_clmidx_(cube, var):
    if var in ('EffPR', 'ET', 'PR', 'PRRN', 'PRSN', 'NetRO'):
        unit = 'mm day$^{-1}$'
        sc = 3600 * 24
    elif var in ('PRmax', 'PRSNmax'):
        unit = 'mm hr$^{-1}$'
        sc = 3600
    elif var == 'PR7Dmax':
        unit = 'mm'
        sc = 3600 * 24 * 7
    elif var == 'SD':
        unit = 'hr'
        sc = 1. / 3600.
    else:
        cuo = 'days' if cube.units.origin == 'day' else cube.units.origin
        unit = latex_unit_(cuo)
        sc = 1.
    return (sc, unit)


def get_ts_clmidx_(cubeLL, dL, sc, fxdir=None, poly=None, rgD=None,
                   folder=None):
    if len(dL) > 0:
        d = ([i.split('_')[0] for i in dL] if folder else
             [i.split('_')[-2] for i in dL])
        if fxdir:
            o = load_fx_(fxdir, d)
        else:
            o = [{}] * len(dL)
        if poly:
            a = [np.ma.asarray([rgMean_poly_cube(i, poly, **ii).data * sc
                             for i, ii in zip(cubeL, o)]) for cubeL in cubeLL]
        else:
            a = [np.ma.asarray([rgMean_cube(i, rgD=rgD, **ii).data * sc
                             for i, ii in zip(cubeL, o)]) for cubeL in cubeLL]
        return a
    else:
        return None


def pure_ts_dn_(ts, dn):
    ind = [np.asarray([np.all(i.mask) for i in ii]) for ii in ts]
    ind_ = np.logical_or.reduce(ind)
    ts_ = [i[~ind_] for i in ts]
    dn_ = l_ind_(dn, np.arange(len(ind_), dtype=int)[~ind_])
    return (ts_, dn_)


def get_ts_eval_(cubeL, dL, sc, fxdir=None, poly=None, rgD=None):
    if len(dL) > 2:
        d = ['_'.join(i.split('_')[-2:]) for i in dL]
        if fxdir:
            o = load_fx_(fxdir, d)
        else:
            o = [{}] * len(dL)
        if poly:
            a = [rgMean_poly_cube(i, poly, **ii) if i else None
                 for i, ii in zip(cubeL, o)]
        else:
            a = [rgMean_cube(i, rgD=rgD, **ii) if i else None
                 for i, ii in zip(cubeL, o)]
        for i in a:
            if i :
                i.data *= sc
        return a
    else:
        return None


def get_clm_clmidx_(cubeLL, sc, cref=None):
    a = [en_pmean_cubeL_(i, cref=cref) for i in cubeLL]
    b = [i - a[0] for i in a[1:]]
    am = [en_mean_(i) for i in a]
    ai = [en_iqr_(i) for i in a]
    bm = [en_mean_(i) for i in b]
    bi = [en_iqr_(i) for i in b]
    for i in am + ai + bm + bi:
        i.data *= sc
    return (am, ai, bm, bi)


def get_clm_eval_(cubeL, sc):
    if len(cubeL) > 2:
        a = [i.collapsed('time', iris.analysis.MEAN) if i else None
             for i in cubeL]
        for i in a:
            if i:
                i.data *= sc
        ec = en_mm_cubeL_(a[:-2])
        aa = [en_mean_(ec), en_iqr_(ec)]
        return (a, aa)
    else:
        return None


def en_pmean_cubeL_(cubeL, cref=None):
    warnings.filterwarnings("ignore", category=UserWarning)
    cl = [c.collapsed('time', iris.analysis.MEAN) for c in cubeL]
    o = en_mm_cubeL_(cl, cref=cref)
    rm_sc_cube(o)
    return o


def map_sim_(dn):
    import re
    gd = {'CCCma-CanESM2': 'a',
          'CNRM-CERFACS-CNRM-CM5': 'b',
          'ICHEC-EC-EARTH': 'c',
          'IPSL-IPSL-CM5A-MR': 'd',
          'MIROC-MIROC5': 'e',
          'MOHC-HadGEM2-ES': 'f',
          'MPI-M-MPI-ESM-LR': 'g',
          'NCC-NorESM1-M': 'h'}
    rd = {'CLMcom-CCLM4-8-17': 'a',
          'CLMcom-ETH-COSMO-crCLIM-v1-1': 'b',
          'CNRM-ALADIN63': 'c',
          'DMI-HIRHAM5': 'd',
          'GERICS-REMO2015': 'e',
          'IPSL-INERIS-WRF331F': 'f',
          'IPSL-WRF381P': 'g',
          'KNMI-RACMO22E': 'h',
          'MPI-CSC-REMO2009': 'i',
          'SMHI-RCA4': 'j',
          'UHOH-WRF361H': 'k'}
    if isinstance(dn, str):
        a, b, c, d, e = dn.split('_')
        aa = gd[a] if b == 'rcp45' else gd[a].upper()
        rip = re.split('[rip]', c)[1]
        ndn = ''.join((aa, rip, rd[d], e[1:]))
        return ndn
    else:
        return [map_sim_(i) for i in dn]


def map_sim_cmip5_(dn):
    import re
    gd = {'ACCESS1-0': 'A',
          'ACCESS1-3': 'B',
          'BNU-ESM': 'C',
          'CCSM4': 'D',
          'CMCC-CM': 'E',
          'CMCC-CMS': 'F',
          'CNRM-CM5': 'G',
          'CSIRO-Mk3-6-0': 'H',
          'CanESM2': 'I',
          'EC-EARTH': 'J',
          'FGOALS-g2': 'K',
          'FIO-ESM': 'L',
          'GFDL-CM3': 'M',
          'GFDL-ESM2G': 'N',
          'GFDL-ESM2M': 'O',
          'GISS-E2-H': 'P',
          'GISS-E2-H-CC': 'Q',
          'GISS-E2-R-CC': 'R',
          'HadGEM2-AO': 'S',
          'HadGEM2-CC': 'T',
          'HadGEM2-ES': 'U',
          'IPSL-CM5A-LR': 'V',
          'IPSL-CM5B-LR': 'W',
          'MIROC-ESM': 'X',
          'MIROC-ESM-CHEM': 'Y',
          'MIROC5': 'Z',
          'MPI-ESM-LR': 'a',
          'MPI-ESM-MR': 'b',
          'MRI-CGCM3': 'c',
          'MRI-ESM1': 'd',
          'NorESM1-M': 'e',
          'NorESM1-ME': 'f',
          'bcc-csm1-1': 'g',
          'bcc-csm1-1-m': 'h',
          'inmcm4': 'i'}
    if isinstance(dn, str):
        a, b, c = dn.split('_')
        r_, i_, p_ = re.split('[rip]', c)[1:]
        r_ = '' if r_ == '1' else 'r' + r_
        i_ = '' if i_ == '1' else 'i' + i_
        p_ = '' if p_ == '1' else 'p' + p_
        ndn = ''.join((gd[a], b[3:], r_, i_, p_))
        return ndn
    else:
        return [map_sim_cmip5_(i) for i in dn]

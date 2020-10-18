import numpy as np
import geopandas as gpd
import matplotlib as mpl
mpl.use('Agg', force=True)
import matplotlib.pyplot as plt
import iris
import os
import time
import logging
import warnings
import argparse
import yaml

from time import localtime, strftime

from climi.uuuu import *
from climi.pppp import *


_here_ = get_path_(__file__)


def _shps(n):
    shp = gpd_read(_here_ + 'swe_districts/distriktsanalys_polygon.shp')
    shp_ = shp.to_crs(epsg=4326)
    idd = shp.ID.copy()
    idd.loc[:] = 0.
    for i in range(n):
        idd = idd.rename('mean{}'.format(i))
        shp = gpd.pd.concat((shp, idd), axis=1)
    for i in range(n):
        idd = idd.rename('iqr{}'.format(i))
        shp = gpd.pd.concat((shp, idd), axis=1)
    return (shp, shp_)


def _minmax_df(df):
    data_ = df.to_numpy()
    return (np.nanmin(data_), np.nanmax(data_))


def _cmp01(unit, var):
    if unit == 'K' or var in ('RLDS', 'RSDS', 'SD', 'TropicNights',
                              'ConWarmDays', 'HumiWarmDays', 'WarmDays'):
        cmp0 = 'coolwarm'
        cmp1 = 'coolwarm'
    elif var in ('DryDays', 'LnstDryDays'):
        cmp0 = 'YlGnBu_r'
        cmp1 = 'BrBG_r'
    elif unit == 'days':
        cmp0 = 'cubehelix'
        cmp1 = 'PRGn_r'
    else:
        cmp0 = 'YlGnBu'
        cmp1 = 'BrBG'
    return (cmp0, cmp1)


def _map(shp0, dn, gwls, unit, odir, var, freq, fig_dict=None):
    #################################################################parameters
    ngwls = gwls + [r'{} $-$ {}'.format(gwl, gwls[0]) for gwl in gwls[1:]]
    nrow = 2
    na, nb = len(gwls), len(ngwls) - len(gwls)
    if na > 4:
        dx = None
        nrow = 4
        ncol = na
        n0, n1, n2, n3 = 1, 2 + ncol, 1 + ncol * 2, 2 + ncol * 3
    else:
        dx = -.03 #rx = .85
        nrow = 2
        ncol = na + nb
        n0, n1, n2, n3 = 1, 1 + na, 1 + ncol, 1 + ncol + na
    pch_dict = {'cmap': 'YlGnBu'}
    cmp0, cmp1 = _cmp01(unit, var)

    min0, max0 = _minmax_df(shp0.loc[:, ['mean{}'.format(i)
                                         for i in range(na)]])
    min1, max1 = _minmax_df(shp0.loc[:, ['iqr{}'.format(i)
                                         for i in range(na)]])
    min2, max2 = _minmax_df(shp0.loc[:, ['mean{}'.format(i)
                                         for i in range(na, na + nb)]])
    min3, max3 = _minmax_df(shp0.loc[:, ['iqr{}'.format(i)
                                         for i in range(na, na + nb)]])
    mm2 = np.max(np.abs([min2, max2]))

    ########################################################################fig
    if fig_dict:
        fig = init_fig_(**fig_dict)
    else:
        fig = init_fig_(h=.02, w=.05, t=.95, b=.025, l=.01, r=.95)

    #########################################################################ax
    axs = []
    pch_dict.update({'vmin': min0, 'vmax': max0, 'cmap': cmp0})
    for i in range(na):
        ax = distri_swe_(fig, nrow, ncol, i + n0, shp0,
                         column='mean{}'.format(i), pcho=pch_dict, ti=gwls[i])
        axs.append(ax)
    pch_dict.update({'vmin': -mm2, 'vmax': mm2, 'cmap': cmp1})
    for i in range(nb):
        ax = distri_swe_(fig, nrow, ncol, i + n1, shp0,
                         column='mean{}'.format(i + na), pcho=pch_dict,
                         ti=ngwls[i + na])
        axs.append(ax)
    pch_dict.update({'vmin': min1, 'vmax': max1, 'cmap': cmp0})
    for i in range(na):
        ax = distri_swe_(fig, nrow, ncol, i + n2, shp0,
                         column='iqr{}'.format(i), pcho=pch_dict)
        axs.append(ax)
    pch_dict.update({'vmin': min3, 'vmax': max3, 'cmap': cmp0})
    for i in range(nb):
        ax = distri_swe_(fig, nrow, ncol, i + n3, shp0,
                         column='iqr{}'.format(i + na), pcho=pch_dict)
        axs.append(ax)

    ###################################################################colorbar
    if dx:
        axs_move_(axs[:na], dx)
        axs_move_(axs[na:(na + nb)], dx)
        axs_move_(axs[(na + nb):(na * 2 + nb)], dx)
        axs_move_(axs[(na * 2 + nb): (na + nb) * 2], dx)
    cb = aligned_cb_(fig, axs[na - 1], get_1st_patchCollection_(axs[na - 1]),
                     [.025, .0125], orientation='vertical')
    cb.set_label('{} ({})'.format(var, unit))
    cb2 = aligned_cb_(fig, axs[na + nb - 1],
                      get_1st_patchCollection_(axs[na + nb - 1]),
                      [.025, .0125], orientation='vertical')
    cb2.set_label('$\Delta${} ({})'.format(var, unit))
    cbz = aligned_cb_(fig, axs[na * 2 + nb - 1],
                      get_1st_patchCollection_(axs[na * 2 + nb - 1]),
                      [.025, .0125], orientation='vertical')
    cbz.set_label('{} IQR ({}) N={}'.format(var, unit, len(dn)))
    cbz2 = aligned_cb_(fig, axs[-1], get_1st_patchCollection_(axs[-1]),
                       [.025, .0125], orientation='vertical')
    cbz2.set_label('$\Delta${} IQR ({}) N={}'.format(var, unit, len(dn)))

    ####################################################################to file
    plt.savefig(odir + '_'.join(('mapX', var, freq)) + '.png')
    plt.close()


def bp__(ts, dn, unit, gwls, odir, var, freq, opt='0', ap=None):
    if len(dn[0].split('_')) == 3:
        ll = map_sim_cmip5_(dn)
    else:
        ll = map_sim_(dn)
    ########################################################################bp0
    if '0' in opt:
        fig = init_fig_(fx=16, l=.05, r=.95, t=0.8)
        ax_opt = {'ylabel': '{} ({})'.format(var, unit)}
        ax0 = fig.add_subplot(1, 1, 1, **ax_opt)
        h_ = bp_dataLL0_(ax0, ts, labels=gwls)
        ax0.legend(h_, ll + ['ENSEMBLE'], bbox_to_anchor=(0., 1.01, 1., 0.18),
                   loc='lower left',
                   borderaxespad=0, ncol=15, mode='expand')
        pn = 'bp0-{}'.format(ap) if ap else 'bp0'
        plt.savefig(odir + '_'.join((pn, var, freq)) + '.png')
        plt.close()

    ########################################################################bp1
    if '1' in opt:
        fig = init_fig_()
        ax_opt = {'ylabel': '{} ({})'.format(var, unit)}
        ax0 = fig.add_subplot(1, 1, 1, **ax_opt)
        h_ = bp_dataLL1_(ax0, ts, labels=gwls)

        ax0.legend(h_, ['ENSEMBLE'])

        pn = 'bp1-{}'.format(ap) if ap else 'bp1'
        plt.savefig(odir + '_'.join((pn, var, freq)) + '.png')
        plt.close()

    ########################################################################bp2
    if '2' in opt:
        fig = init_fig_(fx=16, l=.05, r=.975, b=.1)
        ax_opt = {'ylabel': '{} ({})'.format(var, unit)}
        ax0 = fig.add_subplot(2, 1, 1, **ax_opt)
        ax_opt.update({'ylabel': '$\Delta${} ({})'.format(var, unit)})
        ax1 = fig.add_subplot(2, 1, 2, **ax_opt)
        h0_ = bp_dataLL_(ax0, ts, labels=None)
        h1_ = bp_dataLL_(ax1, [di - ts[0] for di in ts[1:]],
                         labels=ll)
        lgo = dict(ncol=len(gwls)) if len(gwls) > 3 else dict()
        ax0.legend(h0_, gwls, **lgo)
        lgo = dict(ncol=len(gwls) - 1) if len(gwls) > 3 else dict()
        ax1.legend(h1_, [r'{} $-$ {}'.format(gwl, gwls[0])
                         for gwl in gwls[1:]], **lgo)

        pn = 'bp2-{}'.format(ap) if ap else 'bp2'
        plt.savefig(odir + '_'.join((pn, var, freq)) + '.png')
        plt.close()


def map__(am, ai, bm, bi, dn, gwls, unit, rg_dict, odir, var, freq,
          pch__=None, fig_dict=None, ap=None):
    ##############################################################plot function
    if pch__ is None:
        pch__ = pch_swe_

    #################################################################parameters
    dgwls = [r'{} $-$ {}'.format(gwl, gwls[0]) for gwl in gwls[1:]]
    na, nb = len(gwls), len(dgwls)
    if na > 4:
        dx = None
        nrow = 4
        ncol = na
        n0, n1, n2, n3 = 1, 2 + ncol, 1 + ncol * 2, 2 + ncol * 3
    else:
        dx = -.03 #rx = .85
        nrow = 2
        ncol = na + nb
        n0, n1, n2, n3 = 1, 1 + na, 1 + ncol, 1 + ncol + na
    pch_dict = {'cmap': 'YlGnBu'}
    cmp0, cmp1 = _cmp01(unit, var)

    min0, max0 = minmax_cube_(am, rg=rg_dict, p=5)
    min1, max1 = minmax_cube_(ai, rg=rg_dict, p=5)
    min2, max2 = minmax_cube_(bm, rg=rg_dict, p=5)
    min3, max3 = minmax_cube_(bi, rg=rg_dict, p=5)
    mm2 = np.max(np.abs([min2, max2]))

    ########################################################################fig
    if fig_dict:
        fig = init_fig_(**fig_dict)
    else:
        fig = init_fig_(h=.02, w=.05, t=.95, b=.025, l=.01, r=.95)

    #########################################################################ax
    axs, pchs = [], []
    pch_dict.update({'vmin': min0, 'vmax': max0, 'cmap': cmp0})
    for i in range(na):
        ax, pch = pch__(fig, nrow, ncol, i + n0, am[i],
                        rg=rg_dict, ti=gwls[i], pcho=pch_dict)
        axs.append(ax)
        pchs.append(pch)
    pch_dict.update({'vmin': -mm2, 'vmax': mm2, 'cmap': cmp1})
    for i in range(nb):
        ax, pch = pch__(fig, nrow, ncol, i + n1, bm[i],
                        rg=rg_dict, ti=dgwls[i], pcho=pch_dict)
        axs.append(ax)
        pchs.append(pch)
    pch_dict.update({'vmin': min1, 'vmax': max1, 'cmap': cmp0})
    for i in range(na):
        ax, pch = pch__(fig, nrow, ncol, i + n2, ai[i],
                        rg=rg_dict, pcho=pch_dict)
        axs.append(ax)
        pchs.append(pch)
    pch_dict.update({'vmin': min3, 'vmax': max3, 'cmap': cmp0})
    for i in range(nb):
        ax, pch = pch__(fig, nrow, ncol, i + n3, bi[i],
                        rg=rg_dict, pcho=pch_dict)
        axs.append(ax)
        pchs.append(pch)

    ###################################################################colorbar
    if dx:
        axs_move_(axs[:na], dx)
        axs_move_(axs[na:(na + nb)], dx)
        axs_move_(axs[(na + nb):(na * 2 + nb)], dx)
        axs_move_(axs[(na * 2 + nb): (na + nb) * 2], dx)
    cb = aligned_cb_(fig, axs[na - 1], pchs[na - 1], [.025, .0125],
                     orientation='vertical')
    cb.set_label('{} ({})'.format(var, unit))
    cb2 = aligned_cb_(fig, axs[na + nb - 1], pchs[na + nb - 1], [.025, .0125],
                      orientation='vertical')
    cb2.set_label('$\Delta${} ({})'.format(var, unit))
    cbz = aligned_cb_(fig, axs[na * 2 + nb - 1], pchs[na * 2 + nb - 1],
                      [.025, .0125], orientation='vertical')
    cbz.set_label('{} IQR ({}) N={}'.format(var, unit, len(dn)))
    cbz2 = aligned_cb_(fig, axs[-1], pchs[-1], [.025, .0125],
                       orientation='vertical')
    cbz2.set_label('$\Delta${} IQR ({}) N={}'.format(var, unit, len(dn)))

    ####################################################################to file
    fn = 'map-{}'.format(ap) if ap else 'map'
    plt.savefig(odir + '_'.join((fn, var, freq)) + '.png')
    plt.close()


def id__(cubeLL, dn, gwls, sc, unit, odir, var, freq, shp, shp_,
         fig_dict=None):
    shp0 = shp.copy()
    ids = set(shp0.ID)
    for i in ids:
        svd = poly_to_path_(list(shp_.loc[shp_.ID == i, 'geometry']))
        ##########################################################fixing SNWmax
        if var in ['SNWmax', 'R5OScw', 'R1OScw'] and i in [12, 15]:
            a_, dn_ = slct_cubeLL_dnL_(cubeLL, dn, slctStrL_, excl='KNMI')
            ts = get_ts_clmidx_(a_, dn_, sc, poly=svd)
            bp__(ts, dn_, unit, gwls, odir, var, freq, opt='0',
                 ap='ID{}'.format(i))
        ##########################################################fixing SNWmax
        else:
            ts = get_ts_clmidx_(cubeLL, dn, sc, poly=svd)
            bp__(ts, dn, unit, gwls, odir, var, freq, opt='0',
                 ap='ID{}'.format(i))
        ts0 = ts + [ii - ts[0] for ii in ts[1:]]
        ts1 = np.asarray([np.nanmean(ii, axis=-1) for ii in ts0])
        m_ = np.nanmean(ts1, axis=-1)
        iqr_ = (np.nanpercentile(ts1, 75, axis=-1) -
                np.nanpercentile(ts1, 25, axis=-1))
        for ii in range(len(m_)):
            shp0.loc[shp_.ID==i, 'mean{}'.format(ii)] = m_[ii]
            shp0.loc[shp_.ID==i, 'iqr{}'.format(ii)] = iqr_[ii]
    _map(shp0, dn, gwls, unit, odir, var, freq, fig_dict=fig_dict)


def _q_file(odir__, cubeLL, sc, freq, gwls, cref=None):
    fnam = ['{}am_{}_{}.nc'.format(odir__, freq, i)
            for i in ['current'] + gwls]
    fnai = ['{}ai_{}_{}.nc'.format(odir__, freq, i)
            for i in ['current'] + gwls]
    fnbm = ['{}bm_{}_{}.nc'.format(odir__, freq, i) for i in gwls]
    fnbi = ['{}bi_{}_{}.nc'.format(odir__, freq, i) for i in gwls]
    if all([os.path.isfile(i) for i in fnam + fnai + fnbm + fnbi]):
        am, ai, bm, bi = [[iris.load_cube(ii) for ii in i]
                          for i in (fnam, fnai, fnbm, fnbi)]
    else:
        os.makedirs(odir__, exist_ok=True)
        am, ai, bm, bi = get_clm_clmidx_(cubeLL, sc, cref=cref)
        for i, ii in zip((am, ai, bm, bi), (fnam, fnai, fnbm, fnbi)):
            for i_, ii_ in zip(i, ii):
                iris.save(i_, ii_)
    return (am, ai, bm, bi)


def _get_freq(vdict, var):
    freqs = []
    if 'year' in vdict['freq'][var]:
        freqs += ['year']
    if 'season' in vdict['freq'][var]:
        freqs += ['djf', 'mam', 'jja', 'son']
    if 'month' in  vdict['freq'][var]:
        freqs += ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return freqs


def main():
    mpl.style.use('seaborn')
    #####################################################################PARSER
    parser = argparse.ArgumentParser('PLOT CLIMIDX')
    parser.add_argument("opt", type=int, default=0,
                        help="options for dataset on BI")
    parser.add_argument("-s", "--start", type=int, help="")
    parser.add_argument("-e", "--end", type=int, help="")
    parser.add_argument("-l", "--log", type=str, help="")
    args = parser.parse_args()
    opt_, log_ = args.opt, args.log
    sss_, eee_ = args.start, args.end
    ####################################################################LOGGING
    lfn = 'plt{}-{}'.format(opt_, log_)
    nlog = len(schF_keys_('', lfn, ext='.log'))
    logging.basicConfig(filename='{}{}.log'.format(lfn, '_' * nlog),
                        filemode='w',
                        level=logging.INFO)
    logging.info(' {:_^42}'.format('start of program'))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))
    logging.info(' ')
    ##################################################################CONFIGURE
    yf = _here_ + 'cfg_plt_climidx_gwls.yml'
    with open(yf, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    version = cfg['version']
    gg, gwls = cfg['gg'], cfg['gwls']
    #gg = ['gwl15', 'gwl2']
    #gwls = ['current', '1.5 K warming', '2 K wariming']
    #gg = ['gwl15', 'gwl2', 'gwl25', 'gwl3', 'gwl35', 'gwl4']
    #gwls = ['current', '1.5 K warming', '2 K wariming', '2.5 K warming',
    #                    '3 K wariming', '3.5 K warming', '4 K warming']
    root = cfg['root']
    ddir, fxdir = cfg['ddir'], cfg['fxdir']
    ######################################################INDEPENDENT VARIABLES
    vf = _here_ + 'var_dict.yml'
    with open(vf, 'r') as ymlfile:
        vdict = yaml.safe_load(ymlfile)
    yf = _here_ + 'rg_dict.yml'
    with open(yf, 'r') as ymlfile:
        rdict = yaml.safe_load(ymlfile)
    #######################################VARIABLES VARY CORRISPONDING TO OPT_
    rn = 'EUR' if opt_ in (0, 1) else 'GLB'
    idir = '{}{}/'.format(ddir, rn)
    odir = '{}DATA/energi/{}/fig{}K/'.format(root, version, gg[-1][3:])
    _rn = 'SWE/' if opt_ == 0 else ('EUR/' if opt_ == 1 else 'SS/')
    _od0 = odir + '{}/' + _rn
    _od1 = odir.replace('/fig', '/nc') + '{}/'
    #r24 = os.environ.get('r24')
    #odir = '{}DATA/energi/2/fig{}K/'.format(r24, gg[-1][3:])
    #os.makedirs(odir, exist_ok=True)
    #idir = '/nobackup/rossby22/sm_chali/DATA/energi/res/gwls/EUR/'
    #fxdir = '/nobackup/rossby22/sm_chali/DATA/fx/'
    if opt_ == 0:
        ne50 = gpd_read(_here_ + 'ne_50m/ne_50m_admin_0_countries.shp')
        sv = poly_to_path_(list(ne50.loc[ne50.NAME=='Sweden', 'geometry']))
        shp, shp_ = _shps(len(gwls) * 2 - 1)
        rD_map = rdict[0]['SWE']
        if len(gwls) == 3:
            figD = None
        elif len(gwls) < 5:
            figD = dict(fx=len(gwls) * 4, h=.075, w=.075,
                        t=.98, b=.075, l=.075, r=.98)
        else:
            figD = dict(fx=len(gwls) * 2, fy=12,
                        h =.15, w=.015, t=.95, b=.025, l=.01, r=.925)
        mD = dict(fig_dict=figD)
    else:
        rD_map = rdict[0]['EUR']
        ix = 2.5 if opt_ == 1 else 2.75
        if len(gwls) < 5:
            figD = dict(fx=len(gwls) * ix * 2,
                        h=.02, w=.05, t=.95, b=.025, l=.01, r=.95)
            figD = dict(fx=len(gwls) * ix, fy=12,
                        h=.15, w=.015, t=.95, b=.025, l=.01, r=.925)
        mD = dict(pch__=pch_eur_, fig_dict=figD, ap='EUR') if opt_ == 1 else\
             dict(pch__=pch_ll_, fig_dict=figD, ap='EUR')
    if opt_ in (0, 1):
        vvv_ = vdict['vars'][sss_:eee_]
        lD = dict()
        tsD = dict(fxdir=fxdir)
        cref = iris.load_cube('{}{}'.format(idir,
               'PR_ICHEC-EC-EARTH_historical_r12i1p1_'
               'SMHI-RCA4_v1_EUR_year_current.nc'))[0, :, :]
    else:
        vvv_ = ['SST', 'SIC']
        tsD = lD = dict(folder='cmip5')
        cref = iris.load_cube('{}{}'.format(idir,
               'SST_GFDL-ESM2M_rcp85_r1i1p1_GLB_year_gwl2.nc'))[0, :, :]
    ############################################################DOING SOMETHING
    for var in vvv_:
        freqs = _get_freq(vdict, var)
        #freqs = ['Nov', 'Dec']
        odir_ = _od0.format(vdict['odir'][var])
        odir__ = _od1.format(vdict['odir'][var])
        os.makedirs(odir_, exist_ok=True)
        for freq in freqs:
            t000 = l__('{} >>>>>> {}'.format(var, freq))
            a, dn = load_clmidx_(idir, var, gwls=gg, freq=freq, rn=rn,
                                 newestV=True, **lD)
            ######################################################fixing SNWmax
            if var in ['SNWmax', 'R5OScw', 'R1OScw']:
                a, dn = slct_cubeLL_dnL_(a, dn, slctStrL_, excl='KNMI')
            ######################################################fixing SNWmax
            ll_('< load', t000)
            if len(dn) == 0:
                continue
            sc, unit = sc_unit_clmidx_(a[0][0], var)
            #id__(a, dn, gwls, sc, unit, odir_, var, freq, shp, shp_)
            if opt_ == 0:
                id__(a, dn, gwls, sc, unit, odir_, var, freq, shp, shp_,
                     fig_dict=figD)
                ll_('<< id', t000)
                ts = get_ts_clmidx_(a, dn, sc, fxdir=fxdir, poly=sv)
                ll_('< ts', t000)
                bp__(ts, dn, unit, gwls, odir_, var, freq, opt='02', ap=None)
                ll_('<< bp', t000)
            else:
                for i in list(rdict[opt_].keys()):
                    ts = get_ts_clmidx_(a, dn, sc, rgD=rdict[opt_][i], **tsD)
                    ll_('< ts {}'.format(i), t000)
                    bp__(ts, dn, unit, gwls, odir_, var, freq, opt='02', ap=i)
                    ll_('<< bp {}'.format(i), t000)
            am, ai, bm, bi = _q_file(odir__, a, sc, freq, gg, cref=cref)
            ll_('< clm', t000)
            #map__(am, ai, bm, bi, dn, gwls, unit, rg_dict, odir_, var, freq)
            map__(am, ai, bm, bi, dn, gwls, unit, rD_map, odir_, var, freq,
                  **mD)
            ll_('<< map', t000)


def main_():
    parser = argparse.ArgumentParser('PLOT CLIMIDX _')
    parser.add_argument("-s", "--start", type=int, help="")
    parser.add_argument("-e", "--end", type=int, help="")
    args = parser.parse_args()
    sss_, eee_ = args.start, args.end
    nlog = len(schF_keys_('', 'ploting2', ext='.log'))
    logging.basicConfig(filename='ploting2' + '_' * nlog + '.log',
                        filemode='w',
                        level=logging.INFO)
    logging.info(' {:_^42}'.format('start of program'))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))
    logging.info(' ')

    mpl.style.use('seaborn')
    #gg = ['gwl15', 'gwl2']
    #gwls = ['current', '1.5 K warming', '2 K wariming']
    gg = ['gwl15', 'gwl2', 'gwl25', 'gwl3', 'gwl35', 'gwl4']
    gwls = ['current', '1.5 K warming', '2 K wariming', '2.5 K warming',
                        '3 K wariming', '3.5 K warming', '4 K warming']

    r24 = os.environ.get('r24')
    odir = '{}DATA/energi/2/fig{}K/'.format(r24, gg[-1][3:])
    os.makedirs(odir, exist_ok=True)
    idir = '/nobackup/rossby22/sm_chali/DATA/energi/res/gwls/EUR/'
    vf = _here_ + 'var_dict.yml'
    with open(vf, 'r') as ymlfile:
        vdict = yaml.safe_load(ymlfile)
    rf = _here_ + 'rg_dict.yml'
    with open(rf, 'r') as ymlfile:
        rdict = yaml.safe_load(ymlfile)
    #figD = dict(fx=15, h=.02, w=.05,
    #            t=.95, b=.025,
    #            l=.01, r=.95)
    figD = dict(fx=20, fy=12, h=.15, w=.015,
                t=.95, b=.025,
                l=.01, r=.925)
    for var in vdict['vars'][sss_:eee_]:
        freqs = _get_freq(vdict, var)
        odir_ = '{}{}/EUR/'.format(odir, vdict['odir'][var])
        odir__ = odir_.replace('/fig', '/nc')[:-4]
        for freq in freqs:
            t000 = l__('{} >>>>>> {}'.format(var, freq))
            a, dn = load_clmidx_(idir, var, gwls=gg, freq=freq, rn='EUR',
                                 newestV=True)
            ll_('< load', t000)
            if len(dn) == 0:
                continue
            os.makedirs(odir_, exist_ok=True)
            sc, unit = sc_unit_clmidx_(a[0][0], var)
            ######################################################fixing SNWmax
            if var in ['SNWmax', 'R5OScw', 'R1OScw']:
                a, dn = slct_cubeLL_dnL_(a, dn, slctStrL_, excl='KNMI')
            ######################################################fixing SNWmax
            for i in list(rdict[1].keys()):
                ts = get_ts_clmidx_(a, dn, sc, rgD=rdict[1][i])
                ll_('< ts {}'.format(i), t000)
                bp__(ts, dn, unit, gwls, odir_, var, freq, opt='02', ap=i)
                ll_('<< bp {}'.format(i), t000)
            am, ai, bm, bi = _q_file(odir__, a, sc, freq, gg)
            ll_('< clm', t000)
            #map__(am, ai, bm, bi, dn, gwls, unit, rdict[0]['EUR'], odir_, var,
            #      freq, pch__=pch_eur_, ap='EUR')
            map__(am, ai, bm, bi, dn, gwls, unit, rdict[0]['EUR'], odir_, var,
                  freq, pch__=pch_eur_, fig_dict=figD, ap='EUR')
            ll_('<< map', t000)


def main__():
    nlog = len(schF_keys_('', 'ploting3', ext='.log'))
    logging.basicConfig(filename='ploting3' + '_' * nlog + '.log',
                        filemode='w',
                        level=logging.INFO)
    logging.info(' {:_^42}'.format('start of program'))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))
    logging.info(' ')

    mpl.style.use('seaborn')
    #gg = ['gwl15', 'gwl2']
    #gwls = ['current', '1.5 K warming', '2 K wariming']
    gg = ['gwl15', 'gwl2', 'gwl25', 'gwl3', 'gwl35', 'gwl4']
    gwls = ['current', '1.5 K warming', '2 K wariming', '2.5 K warming',
                        '3 K wariming', '3.5 K warming', '4 K warming']

    r24 = os.environ.get('r24')
    odir = '{}DATA/energi/2/fig{}K/'.format(r24, gg[-1][3:])
    os.makedirs(odir, exist_ok=True)
    idir = '/nobackup/rossby22/sm_chali/DATA/energi/res/cmip5/GLB/'
    #fxdir = '/nobackup/rossby22/sm_chali/DATA/fx/'
    vf = _here_ + 'var_dict.yml'
    with open(vf, 'r') as ymlfile:
        vdict = yaml.safe_load(ymlfile)
    rf = _here_ + 'rg_dict.yml'
    with open(rf, 'r') as ymlfile:
        rdict = yaml.safe_load(ymlfile)
    figD = dict(fx=24, fy=12, h=.15, w=.015,
                t=.95, b=.025,
                l=.01, r=.925)
    for var in ['SST', 'SIC']:
        cref = iris.load_cube('{}{}{}'.format(idir, var,
               '_GFDL-ESM2M_rcp85_r1i1p1_GLB_year_gwl2.nc'))
        freqs = _get_freq(vdict, var)
        odir_ = '{}{}/SS/'.format(odir, vdict['odir'][var])
        odir__ = odir_.replace('/fig', '/nc')[:-3]
        for freq in freqs:
            t000 = l__('{} >>>>>> {}'.format(var, freq))
            a, dn = load_clmidx_(idir, var, gwls=gg, freq=freq, rn='GLB',
                                 folder='cmip5')
            ll_('< load', t000)
            if len(dn) == 0:
                continue
            os.makedirs(odir_, exist_ok=True)
            sc, unit = sc_unit_clmidx_(a[0][0], var)
            for i in list(rdict[2].keys()):
                ts = get_ts_clmidx_(a, dn, sc, rgD=rdict[2][i],
                                    folder='cmip5')
                ll_('< ts {}'.format(i), t000)
                ts_, dn_ = pure_ts_dn_(ts, dn)
                bp__(ts_, dn_, unit, gwls, odir_, var, freq, opt='02', ap=i)
                ll_('<< bp {}'.format(i), t000)
            #a = [indFront_(i, dn.index('GFDL-ESM2M_rcp85_r1i1p1')) for i in a]
            am, ai, bm, bi = _q_file(odir__, a, sc, freq, gg)
            ll_('< clm', t000)
            map__(am, ai, bm, bi, dn, gwls, unit, rdict[0]['EUR'], odir_, var,
                  freq, pch__=pch_ll_, fig_dict=figD, ap='EUR')
            ll_('<< map', t000)


if __name__ == '__main__':
    start_time = time.time()
    main()
    logging.info(' ')
    logging.info(' {:_^42}'.format('end of program'))
    logging.info(' {:_^42}'.format('TOTAL'))
    logging.info(' ' + rTime_(time.time() - start_time))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))

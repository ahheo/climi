import numpy as np
import matplotlib as mpl
mpl.use('Agg', warn=False, force=True)
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import os
import warnings
import logging
from ffff import rPeriod_, schF_keys_
from cccc import extract_period_cube, guessBnds_cube, load_res_, en_mm_, \
                 en_mean_, en_iqr_
from pppp import getAggrArg_, pch_swe_, aligned_cb_, aligned_tx_


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser('map plot (hwmi)')
    parser.add_argument("controlfile",
                        help="yaml file with metadata")
    args = parser.parse_args()
    with open(args.controlfile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    warnings.filterwarnings("ignore",
                            message="Collapsing a non-contiguous coordinate.")

    #map options
    cmap = mpl.colors.ListedColormap(cfg['cm'])
    cmap.set_over(cfg['cmo'])
    colorb = np.array(cfg['cbd_' + cfg['v'][0]])
    norm = mpl.colors.BoundaryNorm(colorb, cmap.N)

    #periods
    p0s = np.arange(1951, 1977, 5)

    #rg_dict
    rg_dict = {'lon': cfg['sub_r']['lon'][cfg['rn']],
               'lat': cfg['sub_r']['lat'][cfg['rn']]}

    #pch_dict
    pch_dict = {'cmap': cmap,
                'norm': norm}

    #directory options
    odir = cfg['root'] + cfg['experiment'] + '/' + cfg['fig']
    os.makedirs(odir, exist_ok=True)
    idir = cfg['root'] + cfg['experiment'] + '/' + cfg['res']

    ##############################hist

    for i in cfg['aggr_' + cfg['v'][0]]:
        ti = 'evl_CMIP5vsCORDEXvsEOBS-' + i
        fig = plt.figure(figsize = (16, 10))
        fig.subplots_adjust(hspace=0.1, wspace=0.075,
                            top=0.95, bottom=0.05,
                            left=0.05, right=0.95)
        fnf = odir + cfg['v'] + '_' + ti + cfg['fn_map']

        #data analysis options
        arg0, karg0 = getAggrArg_(i)

        ##obs
        ddir = idir + 'obs/'
        dn = cfg['dn_obs'][0]
        cube = load_res_(ddir, cfg['v'], dn, cfg['rn'], cfg['sub_r'])
        guessBnds_cube(cube)
        ax = []
        pch = []
        for i, p in enumerate(p0s):
            c1 = extract_period_cube(cube, p, p + 29)
            c0 = c1.collapsed('time', *arg0, **karg0)
            ax0, pch0 = pch_swe_(fig, 3, 7, i + 1, c0, rg_dict, pch_dict,
                                 ti=rPeriod_([p, p+29], True))
            ax.append(ax0)
            pch.append(pch0)
        c0 = extract_period_cube(cube, 2018, 2018)
        pch_swe_(fig, 3, 7, 7, c0, rg_dict, pch_dict, ti='2018')

        ##cmip5
        ddir = idir + 'cmip5/hist/'
        tmp = en_mm_(ddir, cfg['dn_cmp'], cfg, ref='NorESM1-M')
        guessBnds_cube(tmp)
        for i, p in enumerate(p0s):
            c1 = extract_period_cube(tmp, p, p + 29)
            ec = c1.collapsed('time', *arg0, **karg0)
            c0 = en_mean_(ec)
            ax0, pch0 = pch_swe_(fig, 3, 7, i + 8, c0, rg_dict, pch_dict)
            ax.append(ax0)
            pch.append(pch0)

        ##cordex
        ddir = idir + 'cordex/hist/'
        tmp = en_mm_(ddir, cfg['dn_cdx'], cfg)
        guessBnds_cube(tmp)
        for i, p in enumerate(p0s):
            c1 = extract_period_cube(tmp, p, p + 29)
            ec = c1.collapsed('time', *arg0, **karg0)
            c0 = en_mean_(ec)
            ax0, pch0 = pch_swe_(fig, 3, 7, i + 15, c0, rg_dict, pch_dict)
            ax.append(ax0)
            pch.append(pch0)

        ##cb
        cb = aligned_cb_(fig, ax[6:], pch[0], [.05, .025],
                         orientation='vertical', ticks=colorb, shrink=0.75,
                         extend='max')
        cb.set_label(cfg['cbti_' + cfg['v'][0]])

       ##ylabel_subplot_array
        aligned_tx_(fig, ax[0], 'EOBS', 'lc')
        aligned_tx_(fig, ax[6], 'CMIP5 (ensemble mean)', 'lc')
        aligned_tx_(fig, ax[12], 'CORDEX (ensemble mean)', 'lc')

        #fig.tight_layout()
        plt.savefig(fnf, **cfg['sv_opts'])
        plt.close()


if __name__ == '__main__':
    main()

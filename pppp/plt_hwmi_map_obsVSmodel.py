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
from cccc import extract_period_cube, guessBnds_cube
from pppp import getAggrArg_, load_res_, pch_swe_, aligned_cb_, en_mm_, \
                 en_mean_, en_iqr_


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

    colorbz = colorb / 5.
    normz = mpl.colors.BoundaryNorm(colorbz, cmap.N)
    #periods
    periods = [[1986, 2005]]

    #rg_dict
    rg_dict = {'lon': cfg['sub_r']['lon'][cfg['rn']],
               'lat': cfg['sub_r']['lat'][cfg['rn']]}

    #pch_dict
    pch_dict = {'cmap': cmap,
                'norm': norm}
    pchz_dict = {'cmap': cmap,
                'norm': normz}

    #directory options
    odir = cfg['root'] + cfg['experiment'] + '/' + cfg['fig']
    os.makedirs(odir, exist_ok=True)
    idir = cfg['root'] + cfg['experiment'] + '/' + cfg['res']

    ##############################hist

    for i in cfg['aggr_' + cfg['v'][0]]:
        ti = 'CMIP5vsCORDEXvsEOBS-' + i
        for p in periods:
            fig = plt.figure(figsize = (10, 8))
            fig.subplots_adjust(hspace=0.1, wspace=0.075,
                                top=0.95, bottom=0.05,
                                left=0.05, right=0.8)
            fnf = odir + cfg['v'] + '_' + ti + '_' + rPeriod_(p) \
                  + cfg['fn_map']

            #data analysis options
            arg0, karg0 = getAggrArg_(i)

            ##obs
            ddir = idir + 'obs/'
            dn = cfg['dn_obs'][0]
            cube = load_res_(ddir, cfg['v'], dn, cfg['rn'], cfg['sub_r'])
            cube = extract_period_cube(cube, p[0], p[1])
            guessBnds_cube(cube)
            c0 = cube.collapsed('time', *arg0, **karg0)
            ax0, pch0 = pch_swe_(fig, 2, 3, 1, c0, rg_dict, pch_dict,
                                 ti='EOBS')

            ##cmip5
            ddir = idir + 'cmip5/hist/'
            tmp = en_mm_(ddir, cfg['dn_cmp'], cfg, ref='NorESM1-M')
            tmp = extract_period_cube(tmp, p[0], p[1])
            guessBnds_cube(tmp)
            ec = tmp.collapsed('time', *arg0, **karg0)
            c1 = en_mean_(ec)
            ax1, pch1 = pch_swe_(fig, 2, 3, 2, c1, rg_dict, pch_dict,
                                 ti='CMIP5 (MEAN)')
            c4 = en_iqr_(ec)
            ax4, pch4 = pch_swe_(fig, 2, 3, 5, c4, rg_dict, pchz_dict,
                                 ti='CMIP5 (IQR)')

            ##cordex
            ddir = idir + 'cordex/hist/'
            tmp = en_mm_(ddir, cfg['dn_cdx'], cfg)
            tmp = extract_period_cube(tmp, p[0], p[1])
            guessBnds_cube(tmp)
            ec = tmp.collapsed('time', *arg0, **karg0)
            c2 = en_mean_(ec)
            ax2, pch2 = pch_swe_(fig, 2, 3, 3, c2, rg_dict, pch_dict,
                                 ti='CORDEX (MEAN)')
            c5 = en_iqr_(ec)
            ax5, pch5 = pch_swe_(fig, 2, 3, 6, c5, rg_dict, pchz_dict,
                                 ti='CORDEX (IQR)')

            ##cb
            cb = aligned_cb_(fig, ax2, pch2, [.05, .025],
                             orientation='vertical', ticks=colorb)
            cb.set_label(cfg['cbti_' + cfg['v'][0]])
            cbz = aligned_cb_(fig, ax5, pch5, [.05, .025],
                             orientation='vertical', ticks=colorbz)
            cbz.set_label(cfg['cbti_' + cfg['v'][0]] + ' (IQR)')

            #fig.tight_layout()
            plt.savefig(fnf, **cfg['sv_opts'])
            plt.close()


if __name__ == '__main__':
    main()

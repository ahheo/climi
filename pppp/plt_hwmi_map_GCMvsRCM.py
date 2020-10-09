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
from pppp import getAggrArg_, load_res_, pch_swe_, aligned_cb_, aligned_tx_


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

    #gcms & rcms
    gcms = ['EC-EARTH', 'CNRM-CM5', 'HadGEM2-ES', 'NorESM1-M']
    rcms = ['CLMcom-CCLM4-8-17', 'DMI-HIRHAM5', 'KNMI-RACMO22E']

    ##############################hist

    for i in cfg['aggr_' + cfg['v'][0]]:
        ti = 'GCMvsRCM-' + i
        #data analysis options
        arg0, karg0 = getAggrArg_(i)

        for p in periods:
            fig = plt.figure(figsize = (8, 10))
            fig.subplots_adjust(hspace=0.1, wspace=0.075,
                                top=0.95, bottom=0.05,
                                left=0.05, right=0.95)
            fnf = odir + cfg['v'] + '_' + ti + '_' + rPeriod_(p) \
                  + cfg['fn_map']

            #obs
            ddir = idir + 'obs/'
            dn = cfg['dn_obs'][0]
            cube = load_res_(ddir, cfg['v'], dn, cfg['rn'], cfg['sub_r'])
            cube = extract_period_cube(cube, p[0], p[1])
            guessBnds_cube(cube)
            c0 = cube.collapsed('time', *arg0, **karg0)
            pch_swe_(fig, 4, 5, 5, c0, rg_dict, pch_dict, ti='EOBS')

            ##models
            for ig, gcm in enumerate(gcms):
                ddir = idir + 'cmip5/hist/'
                tmp = load_res_(ddir, cfg['v'], gcm, cfg['rn'], cfg['sub_r'])
                guessBnds_cube(tmp)
                c = extract_period_cube(tmp, p[0], p[1])
                c1 = c.collapsed('time', *arg0, **karg0)
                pch_swe_(fig, 4, 5, ig + 1, c1, rg_dict, pch_dict, ti=gcm)
                ddir = idir + 'cordex/hist/'
                for ir, rcm in enumerate(rcms):
                    dn = rcm + '_*' + gcm
                    try:
                        tmp = load_res_(ddir, cfg['v'], dn, cfg['rn'],
                                        cfg['sub_r'])
                    except:
                        continue
                    guessBnds_cube(tmp)
                    c = extract_period_cube(tmp, p[0], p[1])
                    c1 = c.collapsed('time', *arg0, **karg0)
                    ax, pch = pch_swe_(fig, 4, 5, (ir + 1) * 5 + ig + 1, c1,
                                       rg_dict, pch_dict)
                    if ig == 0:
                        aligned_tx_(fig, ax, rcm, 'lc')

            ##cb
            cb = aligned_cb_(fig, ax, pch, [.05, .025],
                             orientation='vertical', ticks=colorb)
            cb.set_label(cfg['cbti_' + cfg['v'][0]])

            #fig.tight_layout()
            plt.savefig(fnf, **cfg['sv_opts'])
            plt.close()


if __name__ == '__main__':
    main()

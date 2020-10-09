import numpy as np
import matplotlib as mpl
mpl.use('Agg', warn=False, force=True)
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import os
import warnings
import logging
from ffff import rPeriod_, schF_keys_, iter_str_
from cccc import extract_period_cube, guessBnds_cube, load_res_
from pppp import pdf_iANDe_, axColor_, _flt_cube


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

    #periods
    p = [1986, 2005]

    #directory options
    odir = cfg['root'] + cfg['experiment'] + '/' + cfg['fig']
    os.makedirs(odir, exist_ok=True)
    fnf = odir + cfg['v'] + '_GCMvsRCM_' + rPeriod_(p) + cfg['fn_pdf']
    idir = cfg['root'] + cfg['experiment'] + '/' + cfg['res']

    #gcms & rcms
    gcms = ['EC-EARTH', 'CNRM-CM5', 'HadGEM2-ES', 'NorESM1-M']
    rcms = ['CLMcom-CCLM4-8-17', 'DMI-HIRHAM5', 'KNMI-RACMO22E']

    #colors
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:blue',
              'tab:purple','tab:gray']

    ##############################hist
    fig = plt.figure(figsize = (5, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.075,
                        top=0.95, bottom=0.075,
                        left=0.075, right=0.95)
    #axes option
    xtk = [.125, .25, .5, 1, 2, 4, 8]
    xtkl = iter_str_(xtk)
    ax_opt = {'ylim': [0, .8],
              'xlim': [np.log(.125), np.log(8)],
              'xticks': np.log(xtk),
              'xticklabels': xtkl,
              'facecolor': 'lightgray'}
    for ig, gcm in enumerate(gcms):
        ax_opt.update({'title': gcm})
        if ig == 3:
            ax_opt.update({'xlabel': 'HWMI'})
        ax = fig.add_subplot(4, 1, ig + 1, **ax_opt)
        ##cmp
        ddir = idir + 'cmip5/hist/'
        color = 'tab:gray'
        il0, el0 = pdf_iANDe_(ax, color, cfg, ddir, [gcm], p, True)
        el = [el0]
        ell = ['forcing GCM']
        ddir = idir + 'cordex/hist/'
        for ir, rcm in enumerate(rcms):
            dn = rcm + '_*' + gcm
            try:
                il1, el1 = pdf_iANDe_(ax, colors[ir], cfg, ddir, [dn], p,
                                      True)
                el.append(el1)
                ell.append(rcm)
            except:
                continue
        ax.grid(True, color='w', zorder=-5)
        ax.tick_params(length=0.)
        axColor_(ax, None)
        if ig == 0:
            ax.legend(el, ell)

    plt.savefig(fnf, **cfg['sv_opts'])
    plt.close()

if __name__ == '__main__':
    main()

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
from pppp import pdf_iANDe_, axColor_, aligned_tx_


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
    p0s = [2020, 2070]

    #directory options
    odir = cfg['root'] + cfg['experiment'] + '/' + cfg['fig']
    os.makedirs(odir, exist_ok=True)
    fnf = odir + cfg['v'] + '_HISTvsRCP' + cfg['fn_pdf']
    idir = cfg['root'] + cfg['experiment'] + '/' + cfg['res']

    #############################

    fig = plt.figure(figsize = (8, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.075,
                        top=0.95, bottom=0.075,
                        left=0.075, right=0.95)
    #colors
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']
    colors = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:orange', 'tab:red',
              'tab:purple','tab:gray']

    #axes option
    xtk = [.125, .25, .5, 1, 2, 4, 8, 16, 32]
    xtkl = iter_str_(xtk)
    ax_opt = {'ylim': [0, .8],
              'xlim': [np.log(.125), np.log(32)],
              'xticks': np.log(xtk),
              'xticklabels': xtkl,
              'facecolor': 'lightgray'}
    ##cmp
    ax1 = fig.add_subplot(2, 1, 1, **ax_opt)
    aligned_tx_(fig, ax1, 'CMIP5', rpo='tc', itv=-0.005)
    ##cdx
    ax_opt.update({'xlabel': 'HWMI'})
    ax2 = fig.add_subplot(2, 1, 2, **ax_opt)
    aligned_tx_(fig, ax2, 'CORDEX', rpo='tc', itv=-0.005)

    el = []
    lg = []
    ##obs
    ddir = idir + 'obs/'
    color = 'black'
    [il0, el0] = pdf_iANDe_(ax1, color, cfg, ddir, cfg['dn_obs'],
                            [1986, 2005], True)
    pdf_iANDe_(ax2, color, cfg, ddir, cfg['dn_obs'], [1986, 2005], True)
    el.append(el0)
    lg.append(rPeriod_([1986, 2005], True) + ' (EOBS)')

    ##cmp&cdx
    #hist
    ddir = idir + 'cmip5/hist/'
    color = colors[0]
    [il0, el0] = pdf_iANDe_(ax1, color, cfg, ddir, cfg['dn_cmp'],
                            [1986, 2005], True)
    el.append(el0)
    lg.append(rPeriod_([1986, 2005], True) + ' (Historical)')
    ddir = idir + 'cordex/hist/'
    pdf_iANDe_(ax2, color, cfg, ddir, cfg['dn_cdx'], [1986, 2005], True)
    for i, p0 in enumerate(p0s):
        #rcp45
        ddir = idir + 'cmip5/rcp45/'
        color = colors[i + 1]
        p = [p0, p0 + 29]
        [il0, el0] = pdf_iANDe_(ax1, color, cfg, ddir, cfg['dn_cmp'], p, True)
        el.append(el0)
        lg.append(rPeriod_(p, True) + ' (RCP45)')
        ddir = idir + 'cordex/rcp45/'
        pdf_iANDe_(ax2, color, cfg, ddir, cfg['dn_cdx'], p, True)
        #rcp85
        ddir = idir + 'cmip5/rcp85/'
        color = colors[i + 3]
        [il0, el0] = pdf_iANDe_(ax1, color, cfg, ddir, cfg['dn_cmp'], p, True)
        el.append(el0)
        lg.append(rPeriod_(p, True) + ' (RCP85)')
        ddir = idir + 'cordex/rcp85/'
        pdf_iANDe_(ax2, color, cfg, ddir, cfg['dn_cdx'], p, True)

    #more settings
    ax1.grid(True, color='w', zorder=-5)
    ax1.tick_params(length=0.)
    axColor_(ax1, None)
    ax1.legend(el, lg)
    ax2.grid(True, color='w', zorder=-5)
    ax2.tick_params(length=0.)
    axColor_(ax2, None)

    plt.savefig(fnf, **cfg['sv_opts'])
    plt.close()

if __name__ == '__main__':
    main()

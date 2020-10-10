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
from pppp import cdf_iANDe_, axColor_, aligned_tx_


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
    p0s = np.arange(1951, 1977, 5)

    #directory options
    odir = cfg['root'] + cfg['experiment'] + '/' + cfg['fig']
    os.makedirs(odir, exist_ok=True)
    fnf = odir + cfg['v'] + '_evl_CMIP5vsCORDEXvsEOBS' + cfg['fn_cdf']
    idir = cfg['root'] + cfg['experiment'] + '/' + cfg['res']

    ##############################hist

    fig = plt.figure(figsize = (5, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.075,
                        top=0.95, bottom=0.075,
                        left=0.075, right=0.95)
    #colors
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:blue',
              'tab:purple','tab:gray']

    #axes option
    ax_opt = {'ylim': [0, 1],
              'xlim': [0, 5],
              'facecolor': 'lightgray'}
    ##obs
    ax0 = fig.add_subplot(3, 1, 1, **ax_opt)
    aligned_tx_(fig, ax0, 'EOBS', rpo='tc', itv=-0.005)
    ##cmp
    ax1 = fig.add_subplot(3, 1, 2, **ax_opt)
    aligned_tx_(fig, ax1, 'CMIP5', rpo='tc', itv=-0.005)
    ##cdx
    ax_opt.update({'xlabel': 'HWMI'})
    ax2 = fig.add_subplot(3, 1, 3, **ax_opt)
    aligned_tx_(fig, ax2, 'CORDEX', rpo='tc', itv=-0.005)

    el = []
    lg = []
    for i, p0 in enumerate(p0s):
        color = colors[i]
        p = [p0, p0 + 29]
        #obs
        ddir = idir + 'obs/'
        [il0, el0] = cdf_iANDe_(ax0, color, cfg, ddir, cfg['dn_obs'], p, True)
        el.append(el0)
        lg.append(rPeriod_(p, True))
        ##cmp
        ddir = idir + 'cmip5/hist/'
        cdf_iANDe_(ax1, color, cfg, ddir, cfg['dn_cmp'], p, True)
        ##cdx
        ddir = idir + 'cordex/hist/'
        cdf_iANDe_(ax2, color, cfg, ddir, cfg['dn_cdx'], p, True)

    #more settings
    ax0.grid(True, color='w', zorder=-5)
    ax0.tick_params(length=0.)
    axColor_(ax0, None)
    ax0.legend(el, lg)
    ax1.grid(True, color='w', zorder=-5)
    ax1.tick_params(length=0.)
    axColor_(ax1, None)
    ax2.grid(True, color='w', zorder=-5)
    ax2.tick_params(length=0.)
    axColor_(ax2, None)

    plt.savefig(fnf, **cfg['sv_opts'])
    plt.close()

if __name__ == '__main__':
    main()

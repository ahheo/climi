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
    fnf = odir + cfg['v'] + '_CMIP5vsCORDEXvsEOBS_' + rPeriod_(p) \
               + cfg['fn_pdf']
    idir = cfg['root'] + cfg['experiment'] + '/' + cfg['res']

    ##############################hist
    ti = 'pdf-plot (CMIP5 vs CORDEX vs EOBS ' + rPeriod_(p, True)

    fig = plt.figure(figsize = (8, 5))

    #axes option
    xtk = [.125, .25, .5, 1, 2, 4, 8]
    xtkl = iter_str_(xtk)
    ax_opt = {'ylim': [0, .8],
              'xlim': [np.log(.125), np.log(8)],
              'xticks': np.log(xtk),
              'xticklabels': xtkl,
              'facecolor': 'lightgray',
              'xlabel': 'HWMI',
              'title': ti}

    ax0 = fig.add_subplot(1, 1, 1, **ax_opt)

    ##cmp
    #hist
    ddir = idir + 'cmip5/hist/'
    color = 'tab:blue'
    [il0, el0] = pdf_iANDe_(ax0, color, cfg, ddir, cfg['dn_cmp'], p, True)

    ##cdx
    #hist
    ddir = idir + 'cordex/hist/'
    color = 'tab:red'
    [il1, el1] = pdf_iANDe_(ax0, color, cfg, ddir, cfg['dn_cdx'], p, True)

    ##obs
    ddir = idir + 'obs/'
    color= 'black'
    [il2, el2] = pdf_iANDe_(ax0, color, cfg, ddir, cfg['dn_obs'], p, True)

    #more settings
    ax0.grid(True, color='w', zorder=-5)
    ax0.tick_params(length=0.)
    axColor_(ax0, None)
    ax0.legend([el0, el1, el2],
               ('CMIP5', 'CORDEX', 'EOBS'))

    plt.savefig(fnf, **cfg['sv_opts'])
    plt.close()

if __name__ == '__main__':
    main()

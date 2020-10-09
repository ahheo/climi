import numpy as np
import matplotlib as mpl
mpl.use('Agg', warn=False, force=True)
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import os
import warnings
from iris.experimental.equalise_cubes import equalise_attributes
from .pppp import axColor_, getAggrArg_, ts_iANDe_
from hwmi.ffff import schF_keys_
from hwmi.cccc import rgMean_cube, load_res_, load_fx_


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser('plot time series of hwmi')
    parser.add_argument("controlfile",
                        help="yaml file with metadata")
    args = parser.parse_args()
    with open(args.controlfile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    warnings.filterwarnings("ignore",
                            message="Collapsing a non-contiguous coordinate")
    warnings.filterwarnings("ignore",
                            message="Using DEFAULT_SPHERICAL_EARTH_RADIUS")

    odir = cfg['root'] + cfg['experiment'] + '/' + cfg['fig']
    os.makedirs(odir, exist_ok=True)
    idir = cfg['root'] + cfg['experiment'] + '/' + cfg['res']

    fig = plt.figure(figsize = (8, 8))
    fig.subplots_adjust(hspace=0.2, wspace=0.075,
                        top=0.95, bottom=0.075,
                        left=0.1, right=0.95)

    #axes option
    ax_opt = {'ylim': [0, 20],
              'facecolor': 'lightgray',
              'ylabel': 'HWMI',
              'title': 'CMIP5'}

    ax0 = fig.add_subplot(2, 1, 1, **ax_opt)

    ##cmp
    #hist
    ddir = idir + 'cmip5/hist/'
    color = 'tab:blue'
    [il0, el0] = ts_iANDe_(ax0, color, cfg, ddir,
                           cfg['fx_cmp'], cfg['dn_cmp'])

    #rcp45
    ddir = idir + 'cmip5/rcp45/'
    color = 'tab:green'
    [il1, el1] = ts_iANDe_(ax0, color, cfg, ddir,
                           cfg['fx_cmp'], cfg['dn_cmp'])

    #rcp85
    ddir = idir + 'cmip5/rcp85/'
    color = 'tab:red'
    [il2, el2] = ts_iANDe_(ax0, color, cfg, ddir,
                           cfg['fx_cmp'], cfg['dn_cmp'])

    ##obs
    ddir = idir + 'obs/'
    color= 'black'
    dn = cfg['dn_obs'][0]
    cube = load_res_(ddir, cfg['v'], dn, cfg['rn'], cfg['sub_r'])
    ts = rgMean_cube(cube)
    ol, = iplt.plot(ts, axes=ax0, lw=2, color=color, alpha=.85)

    #more settings
    ax0.grid(True, color='w', zorder=-5)
    ax0.tick_params(length=0.)
    axColor_(ax0, None)
    ax0.legend([el0, el1, el2, ol],
               ('Historical', 'RCP45', 'RCP85', 'EOBS'))
    ax0.xaxis.set_major_locator(mpl.dates.YearLocator(20))

    ax_opt.update({'title': 'CORDEX',
                   'xlabel': 'year'})

    ax1 = fig.add_subplot(2, 1, 2, **ax_opt)
    ##cdx
    #hist
    ddir = idir + 'cordex/hist/'
    color = 'tab:blue'
    [il0, el0] = ts_iANDe_(ax1, color, cfg, ddir,
                           cfg['fx_cdx'], cfg['dn_cdx'])

    #rcp45
    ddir = idir + 'cordex/rcp45/'
    color = 'tab:green'
    [il1, el1] = ts_iANDe_(ax1, color, cfg, ddir,
                           cfg['fx_cdx'], cfg['dn_cdx'])

    #rcp85
    ddir = idir + 'cordex/rcp85/'
    color = 'tab:red'
    [il2, el2] = ts_iANDe_(ax1, color, cfg, ddir,
                           cfg['fx_cdx'], cfg['dn_cdx'])
    ##obs
    color= 'black'
    ol, = iplt.plot(ts, axes=ax1, lw=2, color=color, alpha=.85)

    #more settings
    ax1.grid(True, color='w', zorder=-5)
    ax1.set_xlim(ax0.get_xlim())
    ax1.tick_params(length=0.)
    axColor_(ax1, None)
    ax1.xaxis.set_major_locator(mpl.dates.YearLocator(20))

    #fig.tight_layout()
    plt.savefig(odir + cfg['v'] + cfg['fn_ts'], **cfg['sv_opts'])
    plt.close()


if __name__ == '__main__':
    main()

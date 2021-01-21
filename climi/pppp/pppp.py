import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import cartopy.crs as ccrs
import os
import warnings
from climi.uuuu import *


__all__ = ['aligned_cb_',
           'aligned_tx_',
           'annotate_heatmap',
           'axColor_',
           'ax_move_',
           'axs_move_',
           'axs_rct_',
           'axs_shrink_',
           'bp_cubeL_eval_',
           'bp_dataLL0_',
           'bp_dataLL1_',
           'bp_dataLL_',
           'cdf_iANDe_',
           'distri_swe_',
           'getAggrArg_',
           'get_1st_patchCollection_',
           'heatmap',
           'hspace_ax_',
           'init_fig_',
           'pch_',
           'pch_eur_',
           'pch_ll_',
           'pch_swe_',
           'pdf_iANDe_',
           'ts_eCube_',
           'wspace_ax_']


def init_fig_(fx=12, fy=6,
              h=0.075, w=0.075,
              t=0.98, b=0.075, l=0.075, r=0.98):
    fig = plt.figure(figsize=(fx, fy))
    fig.subplots_adjust(hspace=h, wspace=w,
                        top=t, bottom=b,
                        left=l, right=r)
    return fig


def axColor_(ax, color):
    for child in ax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color(color)


def getAggrArg_(aggr):
    args, kwargs = (), {}
    if aggr.upper() in ['MEAN', 'MEDIAN', 'HMEAN']:
        args = (eval('iris.analysis.' + aggr.upper()),)
    elif aggr[:4].upper() == 'PCTL':
        args = (iris.analysis.PERCENTILE,)
        kwargs.update({'percent': float(aggr[4:])})
    elif aggr[:4].upper() == 'PROP>':
        args = (iris.analysis.PROPORTION,)
        kwargs.update({'function':
                       eval('lambda x: x > {:s}'.format(aggr[4:]))})
    else:
        raise Exception("not prepared for Aggregator '" + aggr + "'")
    return args, kwargs


def _get_clo(cube):
    cs = cube.coord_system()
    if isinstance(cs, (iris.coord_systems.LambertConformal,
                       iris.coord_systems.Stereographic)):
        clo = cs.central_lon
    elif isinstance(cs, iris.coord_systems.RotatedGeogCS):
        clo = cyl_(180 + cs.grid_north_pole_longitude, 180, -180)
    elif isinstance(cs, (iris.coord_systems.Orthographic,
                         iris.coord_systems.VerticalPerspective)):
        clo = cs.longitude_of_projection_origin
    elif isinstance(cs, iris.coord_systems.TransverseMercator):
        clo = cs.longitude_of_central_meridian
    else:
        clo = np.floor(np.mean(cube.coord('longitude').points) / 5) * 5
    return clo


def pch_swe_(fig, nrow, ncol, n, cube,
             rg='data', clo_=None, ti=None, pcho={}, fr_on=False):
    ext = _mapext(cube, rg=rg)
    if isinstance(clo_, (int, float)):
        clo = clo_
    elif clo_ == 'cs':
        clo = _get_clo(cube)
    else:
        clo = _clo_ext(ext, h_=clo_)
    proj = ccrs.NorthPolarStereo(central_longitude=clo)
    return pch_(fig, nrow, ncol, n, cube, proj, ext=ext, ti=ti, pcho=pcho,
                fr_on=fr_on)


def pch_eur_(fig, nrow, ncol, n, cube, rg=None, ti=None, pcho={}, fr_on=False):
    ext = _mapext(cube, rg=rg)
    proj = ccrs.EuroPP()
    return pch_(fig, nrow, ncol, n, cube, proj, ext=ext, ti=ti, pcho=pcho,
                fr_on=fr_on)


def pch_ll_(fig, nrow, ncol, n, cube, rg=None, ti=None, pcho={}, fr_on=False):
    ext = _mapext(cube, rg=rg)
    proj = ccrs.PlateCarree()
    return pch_(fig, nrow, ncol, n, cube, proj, ext=ext, ti=ti, pcho=pcho,
                fr_on=fr_on)


def pch_(fig, nrow, ncol, n, cube, proj, ext=None, ti=None, pcho={},
         fr_on=False):
    ax = fig.add_subplot(nrow, ncol, n, projection=proj)
    if ext:
        ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.coastlines('50m', linewidth=0.5) #coastlines
    ax.outline_patch.set_visible(fr_on)
    pch = _pcolormesh(cube, axes=ax, **pcho)
    if ti is not None:
        ax.set_title(ti)
    return (ax, pch)


def _clo_ext(ext, h_=None):
    if h_ == 'human':
        clo = np.floor(np.mean(ext[:2]) / 5) * 5
    else:
        clo = np.floor(np.mean(ext[:2]))
    return clo


def _mapext(cube, rg='data'):
    if isinstance(rg, dict):
        ext = flt_l([rg['longitude'], rg['latitude']])
    elif rg == 'data':
        ext = [np.min(cube.coord('longitude').points),
               np.max(cube.coord('longitude').points),
               np.min(cube.coord('latitude').points),
               np.max(cube.coord('latitude').points)]
    else:
        ext = None
    return ext


def _pcolormesh(cube, axes=None, **kwArgs):
    if axes is None:
        axes = plt.gca()
    lo0, la0 = cube.coord('longitude'), cube.coord('latitude')
    if lo0.ndim == 1:
        pch = iplt.pcolormesh(cube, axes=axes, **kwArgs)
    else:
        if hasattr(lo0, 'has_bounds') and lo0.has_bounds():
            x, y = lo0.contiguous_bounds(), la0.contiguous_bounds()
        else:
            x, y = _2d_bounds(lo0.points, la0.points)
        pch = axes.pcolormesh(x, y, cube.data,
                              transform=ccrs.PlateCarree(),
                              **kwArgs)
    return pch


def _2d_bounds(x, y):
    def _extx(x2d):
        dx2d = np.diff(x2d, axis=-1)
        return np.hstack((x2d, x2d[:, -1:] + dx2d[:, -1:]))
    def _exty(y2d):
        dy2d = np.diff(y2d, axis=0)
        return np.vstack((y2d, y2d[-1:, :] + dy2d[-1:, :]))
    dx0 = _extx(np.diff(x, axis=-1))
    dx1 = _exty(np.diff(x, axis=0))
    x00 = x - .5 * dx0 - .5 * dx1
    x01 = _extx(x00)
    xx = _exty(x01)
    dy0 = _extx(np.diff(y, axis=-1))
    dy1 = _exty(np.diff(y, axis=0))
    y00 = y - .5 * dy0 - .5 * dy1
    y01 = _extx(y00)
    yy = _exty(y01)
    return (xx, yy)


def ax_move_(ax, dx=0., dy=0.):
    axp = ax.get_position()
    axp.x0 += dx
    axp.x1 += dx
    axp.y0 += dy
    axp.y1 += dy
    ax.set_position(axp)


def axs_move_(axs, dx, d_='x'):
    for i, ax in enumerate(axs):
        if 'x' in d_:
            ax_move_(ax, dx=dx * i)
        elif 'y' in d_:
            ax_move_(ax, dy=dx * i)


def axs_shrink_(axs, rx=1., ry=1., anc='tl'):
    if anc[0] not in 'tbm':
        raise ValueError("anc[0] must be one of 't' ,'m', 'b'!")
    if anc[1] not in 'lcr':
        raise ValueError("anc[1] must be one of 'l' ,'c', 'r'!")
    x0, x1, y0, y1 = _minmaxXYlm(axs)
    for i in axs:
        x00, x11, y00, y11 = _minmaxXYlm(i)
        if anc[1] == 'l':
            dx = (x0 - x11) * (1 - rx) if x0 != x00 else 0.
        elif anc[1] == 'c':
            dx = (x0 + x1 - x00 - x11) * .5 * (1 - rx)
        else:
            dx = (x1 - x00) * (1 - rx) if x1 != x11 else 0.
        if anc[0] == 't':
            dy = (y1 - y00) * (1 - ry) if y1 != y11 else 0.
        elif anc[0] == 'm':
            dy = (y0 + y1 - y00 - y11) * .5 * (1 - ry)
        else:
            dy = (y0 - y11) * (1 - ry) if y0 != y00 else 0.
        ax_move_(i, dx, dy)


def _minmaxXYlm(ax):
    if isinstance(ax, list):
        xmin = min([i.get_position().x0 for i in ax])
        ymin = min([i.get_position().y0 for i in ax])
        xmax = max([i.get_position().x1 for i in ax])
        ymax = max([i.get_position().y1 for i in ax])
    else:
        xmin, ymin = ax.get_position().p0
        xmax, ymax = ax.get_position().p1
    return (xmin, xmax, ymin, ymax)


def axs_rct_(fig, ax, dx=.005, **kwarg):
    xmin, xmax, ymin, ymax = _minmaxXYlm(ax)
    kD = dict(fill=False, color='k', zorder=1000,
              transform=fig.transFigure, figure=fig)
    kD.update(kwarg)
    fig.patches.extend([plt.Rectangle((xmin - dx, ymin -dx),
                                      xmax - xmin + 2*dx, ymax - ymin + 2*dx,
                                      **kD)])


def wspace_ax_(ax0, ax1):
    return ax1.get_position().x0 - ax0.get_position().x1


def hspace_ax_(ax0, ax1):
    return ax0.get_position().y0 - ax1.get_position().y1


def aligned_cb_(fig, ax, ppp, iw, orientation='vertical', shrink=1.,
                **cb_dict):
    xmin, xmax, ymin, ymax = _minmaxXYlm(ax)
    if orientation == 'vertical':
        caxb = [xmax + iw[0], ymin + (ymax - ymin) * (1. - shrink) * 0.5,
                iw[1], (ymax - ymin) * shrink]
    elif orientation == 'horizontal':
        caxb = [xmin + (xmax - xmin) * (1. - shrink) * 0.5,
                ymin - iw[0] - iw[1], (xmax - xmin) * shrink, iw[1]]
    cb = plt.colorbar(ppp, fig.add_axes(caxb), orientation=orientation,
                      **cb_dict)
    return cb


def aligned_tx_(fig, ax, s, rpo='tl', itv=0.005,
                fontdict=None, **kwArgs):
    xmin, xmax, ymin, ymax = _minmaxXYlm(ax)
    if rpo[0].upper() in 'TB':
        xlm = [xmin, xmax]
    elif rpo[0].upper() in 'LR':
        xlm = [ymin, ymax]
    else:
        raise Exception('uninterpretable rpo!')

    if rpo[0].upper() == 'T':
        y = ymax + itv
        if itv >= 0:
            kwArgs.update({'verticalalignment': 'bottom'})
        else:
            kwArgs.update({'verticalalignment': 'top'})
    elif rpo[0].upper() == 'B':
        y = ymin - itv
        if itv >= 0:
            kwArgs.update({'verticalalignment': 'top'})
        else:
            kwArgs.update({'verticalalignment': 'bottom'})
    elif rpo[0].upper() == 'R':
        y = xmax + itv
        if itv >= 0:
            kwArgs.update({'verticalalignment': 'top'})
        else:
            kwArgs.update({'verticalalignment': 'bottom'})
    elif rpo[0].upper() == 'L':
        y = xmin - itv
        if itv >= 0:
            kwArgs.update({'verticalalignment': 'bottom'})
        else:
            kwArgs.update({'verticalalignment': 'top'})

    if rpo[1].upper() == 'L':
        x = xlm[0]
        kwArgs.update({'horizontalalignment': 'left'})
    elif rpo[1].upper() == 'C':
        x = np.mean(xlm)
        kwArgs.update({'horizontalalignment': 'center'})
    elif rpo[1].upper() == 'R':
        x = xlm[1]
        kwArgs.update({'horizontalalignment': 'right'})
    else:
        raise Exception('uninterpretable rpo!')

    if rpo[0].upper() in 'LR':
       x, y = y, x
       kwArgs.update({'rotation': 'vertical', 'rotation_mode': 'anchor'})

    tx = fig.text(x, y, s, fontdict=fontdict, **kwArgs)
    return tx


def _flt_cube(cube):
    data = nanMask_(cube.data).flatten()
    return data[~np.isnan(data)]


def pdf_iANDe_(ax, eCube, color, log_it=False, kopt={}):
    if 'clip' in kopt:
        clip = np.array(kopt['clip'], dtype=np.float64)
        kopt.update({'clip': clip})
    ils = []
    if 'realization' in (i.name() for i in eCube.dim_coords):
        for c in eCube.slices_over('realization'):
            obs = _flt_cube(c)
            _, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
            #plot
            il, = ax.plot(kdeo.support, kdeo.density, lw=0.75, color=color,
                          alpha=.25)
            ils.append(il)
    obs = _flt_cube(eCube)
    _, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
    el, = ax.plot(kdeo.support, kdeo.density, lw=1.5, color=color, alpha=.85)
    return (ils, el)


def cdf_iANDe_(ax, eCube, color, log_it=False, kopt={}):
    if 'clip' in kopt:
        clip = np.array(kopt['clip'], dtype=np.float64)
        kopt.update({'clip': clip})
    ils = []
    if 'realization' in (i.name() for i in eCube.dim_coords):
        for c in eCube.slices_over('realization'):
            obs = _flt_cube(c)
            x, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
            #plot
            il, = ax.plot(x, kdeo.cdf, lw=0.75, color=color, alpha=.25)
            ils.append(il)
    obs = _flt_cube(eCube)
    x, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
    el, = ax.plot(x, kdeo.cdf, lw=1.5, color=color, alpha=.85)
    return (ils, el)


def ts_eCube_(ax, eCube, color):
    y0y1 = y0y1_of_cube(eCube)
    cl = []
    ils = []
    if isinstance(eCube, iris.cube.Cube):
        if 'realization' in (i.name() for i in eCube.coords()):
            cubes = eCube.slices_over('realization')
            #ax_r = eCube.coord_dims('realization')[0]
            #crd_r = eCube.coord('realization').points
            #cubes = [extract_byAxes_(eCube, ax_r, np.where(crd_r == i)[0][0])
            #         for i in crd_r]
            cut = False
        else:
            cubes = []
    else:
        cubes, cut = eCube, True
    for c in cubes:
        tmp = extract_period_cube(c, *y0y1) if cut else c
        cl.append(tmp.data)
        #plot
        il, = iplt.plot(tmp, axes=ax, lw=.5, color=color, alpha=.25, zorder=0)
        ils.append(il)
    ets = tmp.copy(np.mean(np.array(cl), axis=0)) if cubes else eCube
    el, = iplt.plot(ets, axes=ax, lw=1.75, color=color, alpha=.8, zorder=9)
    return (ils, el)


def bp_dataLL_(ax, dataLL, labels=None):
    gn = len(dataLL)
    ng = len(dataLL[0])
    ax.set_xlim(.5, ng + .5)
    ww = .001
    wd = (.6 - (gn - 1) * ww) / gn
    p0s = np.arange(ng) + .7 + wd / 2

    cs = plt.get_cmap('Set2').colors
    bp_dict = {'notch': True,
               'sym': '+',
               'positions': p0s,
               'widths': wd,
               'patch_artist': True,
               'medianprops': {'color': 'lightgray',
                               'linewidth': 1.5}}

    hgn = []
    for i, ii in enumerate(dataLL):
        ts_ = [np.ma.compressed(iii) for iii in ii]
        h_ = ax.boxplot(ts_, **bp_dict)
        for patch in h_['boxes']:
            patch.set_facecolor(cs[cyl_(i, len(cs))] + (.667,))
        hgn.append(h_['boxes'][0])
        p0s += ww + wd
    ax.set_xticks(np.arange(ng) + 1)
    if labels:
        ax.set_xticklabels(labels, rotation=60, ha='right',
                           rotation_mode='anchor')
    else:
        ax.set_xticklabels([None] * ng)
    return hgn


def bp_dataLL0_(ax, dataLL, labels=None):
    gn = len(dataLL)
    ng = len(dataLL[0])
    dd0 = [np.ma.compressed(i) for i in dataLL]
    dd1 = [[np.ma.compressed(dd[i]) for dd in dataLL] for i in range(ng)]
    ax.set_xlim(.5, gn + .5)
    ww = .001
    wd = (.6 - (ng - 1) * ww) / ng
    p0s = np.arange(gn) + .7 + wd / 2
    wd0 = .667
    p0 = np.arange(gn) + 1.

    cs = plt.get_cmap('Set2').colors
    if gn <= 3:
        cs0 = ['b', 'g', 'r']
    else:
        cs0 = plt.get_cmap('tab10').colors
    bp_dict = {'notch': True,
               'sym': '+',
               'zorder': 15,
               'positions': p0s,
               'widths': wd,
               'patch_artist': True,
               'medianprops': {'color': 'lightgrey',
                               'linewidth': 1.5}}
    bp0_dict= {'positions': p0,
               'widths': wd0,
               'sym': '',
               'zorder': 2,
               'capprops': {'color': '#555555dd',
                            'linewidth': 3},
               'boxprops': {'color': '#555555dd'},
               'whiskerprops': {'color': '#555555dd',
                                'linewidth': 3},
               'flierprops': {'color': '#555555dd'}}

    hgn = []
    for i, ii in enumerate(dd1):
        h_ = ax.boxplot(ii, **bp_dict)
        for patch in h_['boxes']:
            patch.set_facecolor(cs[cyl_(i, len(cs))] + (.667,))
        hgn.append(h_['boxes'][0])
        p0s += ww + wd

    bp_dict.update(bp0_dict)
    h_ = ax.boxplot(dd0, **bp_dict)

    for i, patch in enumerate(h_['boxes']):
        patch.set_facecolor('#555555dd')
        patch.set_zorder(bp_dict['zorder'] + 2 * i)
    hgn.append(h_['boxes'][0])
    eps = {}

    for i, md in enumerate(h_['medians']):
        if i == 0:
            y0 = md.get_ydata()[1]
        xd = md.get_xdata()
        xd[1] = ax.get_xlim()[1]
        md.set_xdata(xd)
        md.set_color(cs0[i])
        md.set_zorder(bp_dict['zorder'] + 1 + 2 * i)
        if i > 0:
            s = '${:+.2g}$'.format(md.get_ydata()[1] - y0)
            ax.text(xd[1], md.get_ydata()[1], s, va='center', color=cs0[i])

    ax.set_xticks(p0)
    if labels is not None:
        ax.set_xticklabels(labels, ha='center')
    for i, xtl in enumerate(ax.get_xticklabels()):
        xtl.set_color(cs0[i])

    return hgn


def bp_dataLL1_(ax, dataLL, labels=None):
    gn = len(dataLL)
    ng = len(dataLL[0])
    dd0 = [np.ma.compressed(i) for i in dataLL]
    dd1 = [[dd[i] for dd in dataLL] for i in range(ng)]
    ax.set_xlim(.5, gn + .5)
    wd0 = .667
    p0 = np.arange(gn) + 1.

    if gn <= 3:
        cs0 = ['b', 'g', 'r']
    else:
        cs0 = plt.get_cmap('tab10').colors
    bp_dict = {'notch': True,
               'positions': p0,
               'widths': wd0,
               'sym': '',
               'zorder': 5,
               'patch_artist': True,
               'capprops': {'color': '#555555dd',
                            'linewidth': 3},
               'boxprops': {'color': '#555555dd'},
               'whiskerprops': {'color': '#555555dd',
                                'linewidth': 3},
               'flierprops': {'color': '#555555dd'},
               'medianprops': {'color': 'lightgray',
                               'linewidth': 1.5}}

    hgn = []

    h_ = ax.boxplot(dd0, **bp_dict)
    for i, patch in enumerate(h_['boxes']):
        patch.set_facecolor('#555555dd')
        patch.set_zorder(bp_dict['zorder'] + 2 * i)
    hgn.append(h_['boxes'][0])

    for i, md in enumerate(h_['medians']):
        if i == 0:
            y0 = md.get_ydata()[1]
        xd = md.get_xdata()
        xd[1] = ax.get_xlim()[1]
        md.set_xdata(xd)
        md.set_color(cs0[i])
        md.set_zorder(bp_dict['zorder'] + 1 + 2 * i)
        if i > 0:
            s = '{:+.2g}'.format(md.get_ydata()[1] - y0)
            ax.text(xd[1], md.get_ydata()[1], s, va='center', color=cs0[i])

    ax.set_xticks(p0)
    if labels is not None:
        ax.set_xticklabels(labels, ha='center')
    for i, xtl in enumerate(ax.get_xticklabels()):
        xtl.set_color(cs0[i])

    return hgn


def bp_cubeL_eval_(ax, cubeL):
    XL = ['Simulations']
    dd0 = [flt_l([np.ma.compressed(i.data) for i in cubeL[:-2]])]
    if cubeL[-2]:
        XL.append('EOBS')
        dd0.append(np.ma.compressed(cubeL[-2].data))
    if cubeL[-1]:
        XL.append('ERA-Interim')
        dd0.append(np.ma.compressed(cubeL[-1].data))
    gn = len(XL)
    ng = len(cubeL) - 2
    dd1 = [i.data for i in cubeL[:-2]]
    ax.set_xlim(.5, gn + .5)
    ww = .001
    wd = (.6 - (ng - 1) * ww) / ng
    p0s = np.asarray([.7]) + wd / 2
    wd0 = .667
    p0 = np.arange(gn) + 1.

    cs = plt.get_cmap('Set2').colors
    bp_dict = {'notch': True,
               'sym': '+',
               'zorder': 15,
               'positions': p0s,
               'widths': wd,
               'patch_artist': True,
               'medianprops': {'color': 'lightgrey',
                               'linewidth': 1.5}}
    bp0_dict= {'positions': p0,
               'widths': wd0,
               'sym': '',
               'zorder': 2,
               'capprops': {'color': '#555555dd',
                            'linewidth': 3},
               'boxprops': {'color': '#555555dd'},
               'whiskerprops': {'color': '#555555dd',
                                'linewidth': 3},
               'flierprops': {'color': '#555555dd'}}

    hgn = []
    for i, ii in enumerate(dd1):
        h_ = ax.boxplot(ii, **bp_dict)
        for patch in h_['boxes']:
            patch.set_facecolor(cs[cyl_(i, len(cs))] + (.667,))
        hgn.append(h_['boxes'][0])
        p0s += ww + wd

    bp_dict.update(bp0_dict)
    h_ = ax.boxplot(dd0, **bp_dict)

    for i, patch in enumerate(h_['boxes']):
        patch.set_facecolor('#555555dd')
        patch.set_hatch('x')
        patch.set_zorder(bp_dict['zorder'] + 2 * i)
    hgn.append(h_['boxes'][0])

    ax.set_xticks(p0)
    ax.set_xticklabels(XL, ha='center')

    return hgn


def distri_swe_(fig, nrow, ncol, n, df, pcho={}, ti=None, **kwArgs):
    ax = fig.add_subplot(nrow, ncol, n)
    df.plot(ax=ax, **kwArgs, **pcho)
    ax.set_axis_off()
    if ti is not None:
        ax.set_title(ti)
    return ax


def get_1st_patchCollection_(ax):
    pc_ = None
    for i in ax.get_children():
        if isinstance(i, mpl.collections.PatchCollection):
             pc_ = i
             break
    return pc_


#def heatmap(data, row_labels, col_labels, ax=None,
#            cbar_kw={}, cbarlabel="", **kwargs):
def heatmap(data, row_labels, col_labels, ax=None, tkD=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    if tkD:
        ax.tick_params(**tkD)
        rot=-45
    else:
        rot=45
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    #return im, cbar
    return im


def annotate_heatmap(im, data=None, valfmt="{:.2f}", data_=None,
                     textcolors=("black", "white"),
                     threshold=None, middle_0=False, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    #if isinstance(valfmt, str):
    #    valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    def _ccc(v):
        if middle_0:
            kw.update(color=textcolors[int(im.norm(abs(v)) > threshold)])
        else:
            kw.update(color=textcolors[int(im.norm(v) > threshold)])
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            _ccc(data[i, j])
            #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if data_ is None:
                #text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                text = im.axes.text(j, i, valfmt.format(data[i, j]), **kw)
            else:
                text = im.axes.text(j, i,
                                    valfmt.format(data[i, j], data_[i, j]),
                                    **kw)
            texts.append(text)

    return texts

"""
>--#########################################################################--<
>------------------------------whoknows functions-----------------------------<
>--#########################################################################--<
* b2l_endian_           : return little endian copy
* cyl_                  : values in cylinder axis           --> extract_win_
* dgt_                  : digits of the int part of a number
* ext_                  : get extension of file name
* find_patt_            : extract list from items matching pattern
* flt_                  : flatten (out: generator)          --> flt_l
* flt_l                 : flatten (out: list)               --> (o)uniqL_
* get_path_             : get path from filename str
* haversine_            : distance between geo points in radians
* indFront_             : move element with specified index to front
* ind_inRange_          : indices of values in a range
* ind_s_                : extraction indices (1 axis)       --> inds_ss_
* ind_shape_i_          : slice indices                     --> slice_back_
* ind_win_              : indices of a window in cylinder axis
* inds_ss_              : extraction indices (n axes)
* intsect_              : intersection of lists
* isGI_                 : if Iterator
* isIter_               : if Iterable but not str or bytes
* ismono_               : check if ismononic
* iter_str_             : string elements
* kde_                  : kernel distribution estimate
* kde__                 : tranform before kde_
* l__                   : start logging
* l_ind_                : extract list by providing indices
* l2b_endian_           : return big endian copy
* latex_unit_           : m s-1 -> m s$^{-1}$
* ll_                   : end logging
* nSlice_               : total number of slices
* nanMask_              : masked array -> array with NaN
* nli_                  : if item is list flatten it
* ouniqL_               : ordered unique elements of list
* p_deoverlap_          : remove overlaping period from a list of periods
* p_least_              : extract aimed period from a list of periods
* prg_                  : string indicating progress status (e.g., '#002/999')
* pure_fn_              : filename excluding path and extension
* rMEAN1d_              : rolling window mean
* rPeriod_              : [1985, 2019] -> '1985-2019'
* rSUM1d_               : rolling window sum
* rTime_                : a string of passing time
* rest_mns_             : rest season named with month abbreviation
* robust_bc2_           : robust alternative for numpy.broadcast_to
* schF_keys_            : find files by key words
* slctStrL_             : select string list include or exclude substr(s)
* ss_fr_sl_             : subgroups that without intersections
* uniqL_                : unique elements of list
* valueEqFront_         : move elements equal specified value to front
...

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 02.09.2019
Date last modified: 24.09.2020
          comments: add func dgt_, prg_;
                    fix rMEAN1d_ issue with mode 'full' and 'same'
"""


import numpy as np
import statsmodels.api as sm
import time
import logging
import re
import glob
import warnings
from itertools import permutations, combinations
from typing import Iterable, Iterator
#import math


__all__ = ['b2l_endian_',
           'cyl_',
           'dgt_',
           'ext_',
           'find_patt_',
           'flt_',
           'flt_l',
           'get_path_',
           'haversine_',
           'indFront_',
           'ind_inRange_',
           'ind_s_',
           'ind_shape_i_',
           'ind_win_',
           'inds_ss_',
           'intsect_',
           'isGI_',
           'isIter_',
           'ismono_',
           'iter_str_',
           'kde_',
           'kde__',
           'l__',
           'l_ind_',
           'l2b_endian_',
           'latex_unit_',
           'll_',
           'nSlice_',
           'nanMask_',
           'nli_',
           'ouniqL_',
           'p_deoverlap_',
           'p_least_',
           'prg_',
           'pure_fn_',
           'rMEAN1d_',
           'rPeriod_',
           'rSUM1d_',
           'rTime_',
           'rest_mns_',
           'robust_bc2_',
           'schF_keys_',
           'slctStrL_',
           'ss_fr_sl_',
           'uniqL_',
           'valueEqFront_']


def cyl_(x, rb=2*np.pi, lb=0):
    """
    ... map to value(s) in a cylinder/period axis ...

    Parsed arguments:
         x: to be mapped (numeric array_like)
        rb: right bound of a cylinder/period axis (default 2*pi)
        lb: left bound of a cylinder/period axis (default 0)
    Returns:
        normal value in a cylinder/period axis
    Notes:
        list, tuple transfered as np.ndarray

    Examples:
        >>>cyl_(-1)
        5.283185307179586 #(2*np.pi-1)
        >>>cyl_(32,10)
        2
        >>>cyl_(355,180,-180)
        -5
    """

    if lb >= rb:
         raise ValueError('left bound should not greater than right bound!')

    if isIter_(x):
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.number):
            raise Exception('data not interpretable')

    return (x-lb) % (rb-lb) + lb


def ss_fr_sl_(sl):
    uv = set(flt_l(sl))
    o = []
    def _sssl(ss):
        return set(flt_l([i for i in sl if any([ii in i for ii in ss])]))
    def _ssl(vv):
        si = set(list(vv)[:1])
        while True:
            si_ = _sssl(si)
            if si_ == si:
                break
            else:
                si = si_
        o.append(si)
        rvv = vv - si
        if len(rvv) != 0:
            _ssl(rvv)
    _ssl(uv)
    return o


def _not_list_iter(l):
    for el in l:
        if not isinstance(el, list):
            yield el
        else:
            yield from _not_list_iter(el)


def nli_(l):
    """
    ... flatten a nested List deeply (basic item as not of list type) ...
    """
    return list(_not_list_iter(l))


def flt_(l):
    """
    ... flatten a nested List deeply (generator) ...
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flt_(el)
        else:
            yield el


def flt_l(l):
    """
    ... flatten a nested List deeply (list) ...
    """

    return list(flt_(l))


def kde_(obs, **kde_opts):
    """
    ... derive estimated distribution using kde from statsmodels.api ...

    Notes:
        default option values as documented
        >>>help(sm.nonparametric.KDEUnivariate) #for more options
    """
    o = sm.nonparametric.KDEUnivariate(obs)
    o.fit(**kde_opts)

    return o


def kde__(obs, log_it=False, **kde_opts):
    """
    ...  similar to kde_ but accept log transform of observations ...

    Returns:
            x: true x
            y: true pdf
        kde_o: kde class for log(x) if log_it is True

    Notes:
        default option values as documented
        >>>help(sm.nonparametric.KDEUnivariate) #for more options
    """
    if log_it:
        kde_o = kde_(np.log(obs[obs > 0]), **kde_opts)
        x = np.exp(kde_o.support)
        y = kde_o.density / x
    else:
        kde_o = kde_(obs, **kde_opts)
        x, y = kde_o.support, kde_o.density
    return x, y, kde_o


def ismono_(x, axis=-1):
    """
    ... check if an array is monotonic along axis (default -1) ...
    """

    return (np.all(np.diff(x, axis=axis) > 0)
            or np.all(np.diff(x, axis=axis) < 0))


def nSlice_(shape, axis=-1):
    """
    ... get total number of slices of a CUBE/ARRAY along axis ...

    Parsed arguments:
        shape: shape of parent CUBE/ARRAY that has multiple dimensions
         axis: axis along which the slice is
    Returns:
        total number of slices
    """

    if axis is None:
        shp = shape
    else:
        axis = cyl_(axis, len(shape))
        axis = (axis,) if not isIter_(axis) else axis
        shp = [ii for i, ii in enumerate(shape) if i not in axis]
    return int(np.prod(shp))


def ind_shape_i_(shape, i, axis=-1, sl_=np.s_[:]):
    """
    ... get indices of CUBE/ARRAY for #i slice along axis ...

    Parsed arguments:
        shape: shape of parent CUBE/ARRAY that has multiple dimensions
            i: slice # of all of parent CUBE/ARRAY in C ORDER
         axis: axis along which the slice is
    Returns:
        indices associated with #i slice
    """

    if axis is None:
        return np.unravel_index(i, shape)
    else:
        axis = cyl_(axis, len(shape))
        axis = (axis,) if not isIter_(axis) else axis
        shp = [ii for i, ii in enumerate(shape) if i not in axis]
        tmp = [i for i in range(len(shape)) if i not in axis]
        shpa = {ii: i for i, ii in enumerate(tmp)}
        iikk = np.unravel_index(i, shp)
        return tuple(iikk[shpa[i]] if i in tmp else sl_
                     for i in range(len(shape)))


def ind_s_(ndim, axis, sl_i):
    """
    ... creat indices for extraction along axis ...

    Parsed arguments:
        ndim: number of dimensions in data
        axis: along which for the extraction
        sl_i: slice, list, or 1d array of selected indices along axis
    Returns:
        indices of ndim datan for extraction
    """

    axis = cyl_(axis, ndim)
    return np.s_[:,] * axis + (sl_i,) + np.s_[:,] * (ndim - axis - 1)


def inds_ss_(ndim, axis, sl_i, *vArg):
    """
    ... creat indices for extraction, similar to ind_s_(...) but works for
        multiple axes ...

    Parsed arguments:
        ndim: number of dimensions in data
        axis: along which for the extraction
        sl_i: slice, list, or 1d array of selected indices along axis
        vArg: any pairs of (axis, sl_i)
    Returns:
        indices of ndim datan for extraction (tuple)
    """

    if len(vArg)%2 != 0:
        raise Exception('Arguments not interpretable!')

    inds = list(ind_s_(ndim, axis, sl_i))

    if len(vArg) > 0:
        ax, sl = list(vArg[::2]), list(vArg[1::2])
        if (any(cyl_(ii, ndim) == cyl_(axis, ndim) for ii in ax)
            or len(np.unique(cyl_(ax, ndim))) != len(ax)):
            raise ValueError('duplicated axis provided!')
        else:
            for ii, ss in zip(ax, sl):
                inds[cyl_(ii, ndim)] = ss

    return tuple(inds)


def ind_inRange_(y, y0, y1, side='both', i_=False, r_=None):
    """
    ... boolen as y0 <= y <= y1 if included is true (defalut)
        otherwise y0 < y < y1 ...
    """
    if r_ is None:
        if side in (0, 'i', 'inner'):
            ind = np.logical_and((y > y0), (y < y1))
        elif side in (-1, 'l', 'left'):
            ind = np.logical_and((y >= y0), (y < y1))
        elif side in (1, 'r', 'right'):
            ind = np.logical_and((y > y0), (y <= y1))
        elif side in (2, 'b', 'both'):
            ind = np.logical_and((y >= y0), (y <= y1))
        else:
            raise ValueError("unknow value of side!")
        return np.where(ind) if i_ else ind
    else:
        if y0 > y1 and y0 - y1 < r_ / 2:
            y0, y1 = y1, y0 
        else:
            y1 = cyl_(y1, y0 + r_, y0)
        return ind_inRange_(cyl_(y, y0 + r_, y0), y0, y1, side=side, i_=i_)


def ind_win_(doy, d, w, rb=366, lb=1):
    dw = cyl_(np.arange(d - w, d + 1 + w), rb=rb, lb=lb)
    return np.isin(doy, dw)


def nanMask_(data):
    """
    ... give nan where masked ...
    """
    if np.ma.isMaskedArray(data):
        if np.ma.is_masked(data):
            data.data[data.mask] = np.nan
        data = data.data
    return data


def rPeriod_(p_bounds, TeX_on=False):
    """
    ... return readable style of period from period bounds ...
    """
    if p_bounds[0] != p_bounds[-1]:
        if TeX_on:
            return r'{:d}$-${:d}'.format(p_bounds[0], p_bounds[-1])
        else:
            return r'{:d}-{:d}'.format(p_bounds[0], p_bounds[-1])
    else:
        return r'{:d}'.format(p_bounds[0])


def rTime_(t):
    """
    ... return readable style of time interval ...
    """
    d = t / (60 * 60 * 24)
    if d >= 1:
        return r'passing :t::i:::m::::e::::: {:.2f} day(s)'.format(d)
    else:
        return time.strftime('passing :t::i:::m::::e::::: %H:%M:%S +0000',
                             time.gmtime(t))


def uniqL_(l):
    """
    ... return sorted unique elements of list l ...
    """
    return list(np.unique(np.array(flt_l(l))))


def ouniqL_(l):
    """
    ... return ordered unique elements of list l ...
    """
    return list(dict.fromkeys(flt_l(l)).keys())


def schF_keys_(idir, *keys, ext='*', ordered=False, h_=False):
    """
    ... find files that contain specified keyword(s) in the directory ...
    """
    s = '*'
    if ordered:
        pm = [s.join(keys)]
    else:
        a = set(permutations(keys))
        pm = [s.join(i) for i in a]
    fn = []
    for i in pm:
        if h_:
            fn += glob.iglob(idir + '.*' + s.join([i, ext]))
        fn += glob.glob(s.join([idir, i, ext]))
    fn = list(set(fn))
    fn.sort()
    return fn


def valueEqFront_(l, v):
    """
    ... move element(s) equal to the specified value v in the list l to front
    """
    l0 = [i for i in l if i == v]
    l1 = [i for i in l if i != v]
    return l0 + l1


def indFront_(l, v):
    ind = valueEqFront_(list(range(len(l))), v)
    return l_ind_(l, ind)


def iter_str_(iterable):
    """
    ... transform elements to string ...
    """
    tmp = flt_l(iterable)
    return [str(i) for i in tmp]


def ext_(s):
    """
    ... get extension from filename (str) ...
    """
    tmp = re.search(r'\.\w+$', s)
    return tmp.group() if tmp else tmp


def find_patt_(p, s):
    """
    ... return s or list of items in s that match the given pattern ...
    """
    if isinstance(s, str):
        return s if re.search(p, s) else None
    elif isIter_(s, xi=str):
        return [i for i in s if find_patt_(p, i)]


def pure_fn_(s):
    """
    ... get pure filename without path to and without extension ...
    """
    if isinstance(s, str):
        tmp = re.search(r'((?<=[\\/])[^\\/]*$)|(^[^\\/]+$)', s)
        fn = tmp.group() if tmp else tmp
        return re.sub(r'\.\w+$', '', fn) if fn else fn
    elif isIter_(s, str):
        return [pure_fn_(i) for i in s]


def get_path_(s):
    """
    ... get path from filename ...
    """
    tmp = re.sub(r'[^\/]+$', '', s)
    return tmp if tmp else './'


def rest_mns_(mmm):
    """
    ... get rest season named with months' 1st letter ...
    """
    mns = 'jfmamjjasond' * 2
    n = mns.find(mmm)
    if n == -1:
        raise Exception('unknown season provided!')
    else:
        return mns[n + len(mmm):n + 12]


def rSUM1d_(y, n, mode='valid'):
    """
    ... sum over a n-point rolling_window ...
    """
    if hasattr(y, 'mask'):
        msk = np.ma.getmaskarray(y)
    else:
        msk = np.isnan(y)
    return np.convolve(np.where(msk, 0, y), np.ones((n,)), mode)


def rMEAN1d_(y, n, mode='valid'):
    """
    ... mean over a n-point rolling_window ...
    """
    if hasattr(y, 'mask'):
        msk = np.ma.getmaskarray(y)
    else:
        msk = np.isnan(y)
    uu = np.convolve(np.where(msk, 0, y), np.ones((n,)), mode)
    dd = np.convolve(~msk, np.ones((n,)), mode)
    dd[dd == 0] = np.nan
    out = uu / dd
    return np.ma.masked_where(np.isnan(out), out) if hasattr(y, 'mask') else\
           out


def l__(msg, out=True):
    """
    ... starting logging msg giving a time stamp ...
    """
    logging.info(' ' + msg + ' -->')
    if out:
        return time.time()


def ll_(msg, t0=None):
    """
    ... ending logging msg giving a time lapse if starting time stamp given
    """
    logging.info(' {}{}'.format(msg, ' <--' if t0 else ''))
    if t0:
        logging.info(' ' + rTime_(time.time() - t0))
        logging.info(' ')


def slctStrL_(strl, incl=None, excl=None, incl_or=False, excl_and=False):
    """
    ... select items including/excluding sts(s) for a list of str ...
    """
    if incl:
        incl = [incl] if isinstance(incl, str) else incl
        if incl_or:
            strl = [i for i in strl if any([ii in i for ii in incl])]
        else:
            strl = [i for i in strl if all([ii in i for ii in incl])]
    if excl:
        excl = [excl] if isinstance(excl, str) else excl
        if excl_and:
            strl = [i for i in strl if not all([ii in i for ii in excl])]
        else:
            strl = [i for i in strl if not any([ii in i for ii in excl])]
    return strl


def latex_unit_(unit):
    """
    ... turn unit str into latex style ...
    """
    def r__(m):
        return '$^{' + m.group(0) + '}$'
    return re.sub(r'(?<=[a-zA-Z])-?\d+', r__, unit)


def p_least_(pl, y0, y1):
    """
    ... select periods within [y0, y1] from a list of periods ...
    """
    pl.sort()
    a = lambda x: y0 <= int(x) <= y1
    b = lambda x, y: a(x) or a(y)
    c = lambda x, y: int(x) <= y0 and int(y) >= y1
    return [i for i in pl if b(i.split('-')[0][:4], i.split('-')[-1][:4])
            or c(i.split('-')[0][:4], i.split('-')[-1][:4])]


def p_deoverlap_(pl):
    """
    ... des-overlap period list ...
    """
    pl = np.asarray(pl)
    pi = np.arange(len(pl))
    ii_ = []
    iii_ = []
    a_ = lambda p: [int(i) for i in p.split('-')]
    b_ = lambda x, p: a_(p)[0] <= x <= a_(p)[-1]
    c_ = lambda p0, p1: b_(a_(p0)[0], p1) and b_(a_(p0)[-1], p1)
    for i in pi:
        if any([c_(pl[i], pl[ii])
                for ii in pi if ii != i and ii not in ii_]):
            ii_.append(i)
            iii_.append(False)
        else:
            iii_.append(True)
    return list(pl[iii_])


def _match2shps(shape0, shape1, fw=False):
    if len(shape1) < len(shape0):
        raise Exception('len(shape1) must >= len(shape0)!')
    cbs = list(combinations(shape1, len(shape0)))
    if shape0 not in cbs:
        raise Exception("unmatched shapes!")
    cbi = list(combinations(np.arange(len(shape1)), len(shape0)))
    if not fw:
        cbs.reverse()
        cbi.reverse()
    o = cbi[0]
    for ss, ii in zip(cbs, cbi):
        if shape0 == ss:
            o = ii
            break
    return o


def robust_bc2_(data, shape, axes=None, fw=False):
    """
    ... broadcast data to shape according to axes given or direction ...
    """
    data = data.squeeze()
    dshp = data.shape
    if axes:
        axes = cyl_(axes, len(shape))
        axes = (axes,) if not isIter_(axes) else axes
        if len(axes) != len(dshp):
            raise ValueError("len(axes) != len(data.squeeze().shape)")
        if (len(np.unique(axes)) != len(axes) or
            any([i not in range(len(shape)) for i in axes])):
            raise ValueError("one or more axes exceed target shape of data!")
    else:
        axes = _match2shps(dshp, shape, fw=fw)
    shp_ = tuple(ii if i in axes else 1 for i, ii in enumerate(shape))
    return np.broadcast_to(data.reshape(shp_), shape)


def intsect_(*l):
    if len(l) > 1:
        ll = list(set(l[0]).intersection(*l[1:]))
        ll.sort()
        return ll
    elif len(l) == 1:
        return l[0]


def l_ind_(l, ind):
    if isIter_(ind, xi=(bool, np.bool, np.bool_)):
        return [i for i, ii in zip(l, ind) if ii]
    elif isIter_(ind, xi=(int, np.integer)):
        ind = cyl_(ind, len(l))
        return [l[i] for i in ind]


def dgt_(n):
    return int(np.floor(np.log10(n)) + 1)


def prg_(i, n=None):
    ss = '#{:0' + r'{:d}'.format(dgt_(n)) + r'd}/{:d}' if n else '#{:d}/--'
    return ss.format(i, n)


def b2l_endian_(x):
    return x.astype(np.dtype(x.dtype.str.replace('>', '<')))


def l2b_endian_(x):
    return x.astype(np.dtype(x.dtype.str.replace('<', '>')))


def isGI_(x):
    return isinstance(x, Iterator)


def isIter_(x, xi=None, XI=(str, bytes)):
    o = isinstance(x, Iterable) and not isinstance(x, XI)
    if o and xi is not None:
        if not isGI_(x):
            o = o and all([isinstance(i, xi) or i is None for i in x])
        else:
            warnings.warn("xi ignored for Iterator or Generator!")
    return o


def haversine_(x0, y0, x1, y1):
    lat0 = np.radians(y0)
    lon0 = np.radians(x0)
    lat1 = np.radians(y1)
    lon1 = np.radians(x1)

    dlon = lon1 - lon0
    dlat = lat1 - lat0

    a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(dlon/2)**2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

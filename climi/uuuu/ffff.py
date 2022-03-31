"""
>--#########################################################################--<
>------------------------------whoknows functions-----------------------------<
>--#########################################################################--<
* aggr_func_            : aggregate function over ndarray
* b2l_endian_           : return little endian copy
* compressLL_           : compress 2D list
* consecutive_          : consecutive numbers
* cyl_                  : values in cylinder axis           --> extract_win_
* dgt_                  : digits of the int part of a number
* el_join_              : element-wise join
* ext_                  : get extension of file name
* find_patt_            : extract list from items matching pattern
* flt_                  : flatten (out: generator)          --> flt_l
* flt_ndim_             : flatten number of consecutive dims
* flt_l                 : flatten (out: list)               --> (o)uniqL_
* get_path_             : get path from filename str
* haversine_            : distance between geo points in radians
* iind_                 : rebuild extraction indices (n axes)
* indFront_             : move element with specified index to front
* ind_inRange_          : indices of values in a range
* ind_s_                : extraction indices (1 axis)       --> inds_ss_
* ind_shape_i_          : slice indices                     --> slice_back_
* ind_win_              : indices of a window in cylinder axis
* inds_ss_              : extraction indices (n axes)
* intsect_              : intersection of lists
* isGI_                 : if Iterator
* isIter_               : if Iterable but not str or bytes
* isMonth_              : string if a month
* ismono_               : check if ismononic
* isSeason_             : string if is a season
* iter_str_             : string elements
* kde_                  : kernel distribution estimate
* kde__                 : tranform before kde_
* l__                   : start logging
* l_ind_                : extract list by providing indices
* l2b_endian_           : return big endian copy
* latex_unit_           : m s-1 -> m s$^{-1}$
* ll_                   : end logging
* m2s_                  : season
* m2sm_                 : season membership
* m2sya_                : season year adjust
* mmmN_                 : months in season
* mnN_                  : month order in the calendar
* nSlice_               : total number of slices
* nanMask_              : masked array -> array with NaN
* ndigits_              : number of digits
* nli_                  : if item is list flatten it
* ouniqL_               : ordered unique elements of list
* p_deoverlap_          : remove overlaping period from a list of periods
* p_least_              : extract aimed period from a list of periods
* prg_                  : string indicating progress status (e.g., '#002/999')
* pure_fn_              : filename excluding path (& also extension by default)
* rMEAN1d_              : rolling window mean
* rPeriod_              : [1985, 2019] -> '1985-2019'
* rSUM1d_               : rolling window sum
* rTime_                : a string of passing time
* rest_mns_             : rest season named with month abbreviation
* robust_bc2_           : robust alternative for numpy.broadcast_to
* schF_keys_            : find files by key words
* shp_drop_             : drop dims specified (and replace if desired)
* slctStrL_             : select string list include or exclude substr(s)
* ss_fr_sl_             : subgroups that without intersections
* uniqL_                : unique elements of list
* valid_seasons_        : if provided seasons valid
* valueEqFront_         : move elements equal specified value to front
* windds2uv_            : u v from wind speed and direction
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
import pandas as pd
import warnings
#import math


__all__ = ['aggr_func_',
           'b2l_endian_',
           'compressLL_',
           'consecutive_',
           'cyl_',
           'dgt_',
           'el_join_',
           'ext_',
           'find_patt_',
           'flt_',
           'flt_ndim_',
           'flt_l',
           'get_path_',
           'haversine_',
           'iind_',
           'indFront_',
           'ind_inRange_',
           'ind_s_',
           'ind_shape_i_',
           'ind_win_',
           'inds_ss_',
           'intsect_',
           'isGI_',
           'isIter_',
           'isMonth_',
           'ismono_',
           'isSeason_',
           'iter_str_',
           'kde_',
           'kde__',
           'l__',
           'l_ind_',
           'l2b_endian_',
           'latex_unit_',
           'll_',
           'm2s_',
           'm2sm_',
           'm2sya_',
           'mmmN_',
           'mnN_',
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
           'shp_drop_',
           'slctStrL_',
           'ss_fr_sl_',
           'uniqL_',
           'valid_seasons_',
           'valueEqFront_',
           'windds2uv_']


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

    assert lb < rb, 'left bound should not greater than right bound!'

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
    from typing import Iterable
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
    import statsmodels.api as sm
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


def inds_ss_(ndim, axis, sl_i, *vArg, _safely=True):
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

    assert len(vArg)%2 == 0, 'arguments not paired!'

    inds = list(ind_s_(ndim, axis, sl_i))

    if len(vArg) > 0:
        ax, sl = list(vArg[::2]), list(vArg[1::2])
        if (any(cyl_(ii, ndim) == cyl_(axis, ndim) for ii in ax)
            or len(pd.unique(cyl_(ax, ndim))) != len(ax)):
            raise ValueError('duplicated axis provided!')
        else:
            for ii, ss in zip(ax, sl):
                inds[cyl_(ii, ndim)] = ss

    return iind_(tuple(inds)) if _safely else tuple(inds)


def iind_(inds):
    x = [ii for ii, i in enumerate(inds) if isIter_(i)]
    if len(x) < 2:
        return inds
    else:
        inds_ = list(inds)
        y = [i for ii, i in enumerate(inds) if isIter_(i)]
        z = np.ix_(*y)
        for ii, i in zip(x, z):
            inds_[ii] = i
        return tuple(inds_)


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
    import time
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


def schF_keys_(idir, *keys, s_='*',  ext='*', ordered=False, h_=False):
    """
    ... find files that contain specified keyword(s) in the directory ...
    """
    import glob
    import os
    from itertools import permutations
    s = '*'
    if ordered:
        pm = [s.join(keys)]
    else:
        a = set(permutations(keys))
        pm = [s.join(i) for i in a]
    fn = []
    for i in pm:
        if h_:
            fn += glob.iglob(os.path.join(idir, '.' + s_ + s.join([i, ext])))
        fn += glob.glob(os.path.join(idir, s_ + s.join([i, ext])))
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
    import os
    return os.path.splitext(s)[1]
    #import re
    #tmp = re.search(r'(?<=\w)\.\w+$', s)
    #return tmp.group() if tmp else ''


def find_patt_(p, s):
    """
    ... return s or list of items in s that match the given pattern ...
    """
    import re
    if isinstance(s, str):
        return s if re.search(p, s) else None
    elif isIter_(s, xi=str):
        return [i for i in s if find_patt_(p, i)]


def pure_fn_(s, no_etc=True):
    """
    ... get pure filename without path to and without extension ...
    """
    #import re
    import os
    def _rm_etc(s):
        #return re.sub(r'\.\w+$', '', s) if no_etc else s
        return os.path.splitext(s)[0] if no_etc else s
    if isinstance(s, str):
        #tmp = re.search(r'((?<=[\\/])[^\\/]*$)|(^[^\\/]+$)', s)
        #fn = tmp.group() if tmp else tmp
        fn = os.path.basename(s)
        return _rm_etc(fn) #if fn else ''
    elif isIter_(s, str):
        return [pure_fn_(i) for i in s]


def get_path_(s):
    """
    ... get path from filename ...
    """
    import re
    tmp = re.sub(r'[^\/]+$', '', s)
    return tmp if tmp else './'


def isMonth_(mn, short_=True, nm=3):
    """
    ...  if input string is name of a month ...
    """
    mns = ['january', 'february', 'march', 'april', 'may', 'june',
           'july', 'august', 'september', 'october', 'november', 'december']
    n = len(mn)
    if n < 3:
        warnings.warn("month string shorter than 3 letters; return 'False'!")
        return False
    mn3s = [i[:n] for i in mns]
    if short_:
        return mn.lower() in mn3s
    else:
        return mn.lower() in mns or mn.lower() in mn3s


def mnN_(mn):
    """
    ...  month order in calendar...
    """
    mns = ['january', 'february', 'march', 'april', 'may', 'june',
           'july', 'august', 'september', 'october', 'november', 'december']
    n = len(mn)
    if n < 3:
        warnings.warn("month string short than 3 letters; 1st guess used!")
    mn3s = [i[:n] for i in mns]
    return mn3s.index(mn.lower()) + 1


def isSeason_(mmm, ismmm_=True):
    """
    ...  if input string is a season named with 1st letters of composing
         months ...
    """
    mns = 'jfmamjjasond' * 2
    n = mns.find(mmm.lower())
    s4 = {'spring', 'summer', 'autumn', 'winter'}
    if ismmm_:
        return (1 < len(mmm) < 12 and n != -1)
    else:
        return (1 < len(mmm) < 12 and n != -1) or mmm.lower() in s4


def valid_seasons_(seasons, ismmm_=True):
    o = all(isSeason_(season, ismmm_=ismmm_) for season in seasons)
    if o:
        o_ = sorted(flt_l(mmmN_(season) for season in seasons))
        return np.array_equal(o_, np.arange(12) + 1)
    else:
        return False


def _month_season_numbers(seasons):
    month_season_numbers = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for season_number, season in enumerate(seasons):
        for month in mmmN_(season):
            month_season_numbers[month] = season_number
    return month_season_numbers


def _month_year_adjust(seasons):
    month_year_adjusts = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for season in seasons:
        months_in_season = mmmN_(season)
        for month in months_in_season:
            if month > months_in_season[-1]:
                month_year_adjusts[month] = 1
    return month_year_adjusts


def _m2sm(month, season):
    return month in mmmN_(season)


def _m2sya(month, seasons=('djf', 'mam', 'jja', 'son')):
    sya = _month_year_adjust(seasons)
    return sya[month]


def _m2s(month, seasons=('djf', 'mam', 'jja', 'son')):
    ssm = _month_season_numbers(seasons)
    return seasons[ssm[month]]


m2s_ = np.vectorize(_m2s, excluded=['seasons'])
m2sm_ = np.vectorize(_m2sm, excluded=['season'])
m2sya_ = np.vectorize(_m2sya, excluded=['seasons'])


def mmmN_(mmm):
    """
    ... months in season  ...
    """
    ss = {'spring': 'mam',
          'summer': 'jja',
          'autumn': 'son',
          'winter': 'djf'}
    mmm = ss[mmm] if mmm in ss.keys() else mmm

    mns = 'jfmamjjasond' * 2
    n = mns.find(mmm.lower())
    if n != -1:
        return cyl_(range(n+1, n+1+len(mmm)), 13, 1)
    else:
        raise ValueError("{!r} unrecognised as a season!".format(mmm))


def rest_mns_(mmm):
    """
    ... get rest season named with months' 1st letter ...
    """
    mns = 'jfmamjjasond' * 2
    n = mns.find(mmm.lower())
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


def l__(msg, out=True, _p=False):
    """
    ... starting logging msg giving a time stamp ...
    """
    import time
    import logging
    msg0 = ' {} -->'.format(msg)
    if _p:
        print(msg0)
    else:
        logging.info(msg0)
    if out:
        return time.time()


def ll_(msg, t0=None, _p=False):
    """
    ... ending logging msg giving a time lapse if starting time stamp given
    """
    import time
    import logging
    msg0 = ' {}{}'.format(msg, ' <--' if t0 else '')
    if _p:
        print(msg0)
    else:
        logging.info(msg0)
    if t0:
        msg1 = ' {}'.format(rTime_(time.time() - t0))
        if _p:
            print(msg1)
            print(' ')
        else:
            logging.info(msg1)
            logging.info(' ')


def slctStrL_(strl, incl=None, excl=None): #, incl_or=False, excl_and=False):
    """
    ... select items including/excluding sts(s) for a list of str ...
    """
    def _in(s, L):
        if isinstance(L, str):
            return L in s
        else:
            return _inc(s, L)
    def _inc(s, L):
        return all([i in s if isinstance(i, str) else _incl(s, i) for i in L])
    def _incl(s, L):
        return any([i in s if isinstance(i, str) else _inc(s, i) for i in L])
    def _ex(s, L):
        if isinstance(L, str):
            return L not in s
        else:
            return _exc(s, L)
    def _exc(s, L):
        return any([i not in s if isinstance(i, str) else _excl(s, i)
                    for i in L])
    def _excl(s, L):
        return all([i not in s if isinstance(i, str) else _exc(s, i)
                    for i in L])
    if incl:
        strl = [i for i in strl if _in(i, incl)]
    if excl:
        strl = [i for i in strl if _ex(i, excl)]
    #if incl:
    #    incl = [incl] if isinstance(incl, str) else incl
    #    if incl_or:
    #        strl = [i for i in strl if any([ii in i for ii in incl])]
    #    else:
    #        strl = [i for i in strl if all([ii in i for ii in incl])]
    #if excl:
    #    excl = [excl] if isinstance(excl, str) else excl
    #    if excl_and:
    #        strl = [i for i in strl if not all([ii in i for ii in excl])]
    #    else:
    #        strl = [i for i in strl if not any([ii in i for ii in excl])]
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
    y0_, y1_ = str(y0), str(y1)
    def _cmp(x0, x1):
        n = min(len(x0), len(x1))
        return x0[:n] <= x1[:n]
    a = lambda x: _cmp(y0_, x) and _cmp(x, y1_)
    b = lambda x, y: a(x) or a(y)
    c = lambda x, y: _cmp(x, y0_) and _cmp(y1_, y)
    return [i for i in pl if b(i.split('-')[0], i.split('-')[-1])
            or c(i.split('-')[0], i.split('-')[-1])]
    #a = lambda x: y0 <= int(x) <= y1
    #b = lambda x, y: a(x) or a(y)
    #c = lambda x, y: int(x) <= y0 and int(y) >= y1
    #return [i for i in pl if b(i.split('-')[0][:4], i.split('-')[-1][:4])
    #        or c(i.split('-')[0][:4], i.split('-')[-1][:4])]


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
    from itertools import combinations
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
        if (len(pd.unique(axes)) != len(axes) or
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
    return ss.format(i + 1, n)


def b2l_endian_(x):
    return x.astype(np.dtype(x.dtype.str.replace('>', '<')))


def l2b_endian_(x):
    return x.astype(np.dtype(x.dtype.str.replace('<', '>')))


def isGI_(x):
    from typing import Iterator
    return isinstance(x, Iterator)


def isIter_(x, xi=None, XI=(str, bytes)):
    from typing import Iterable
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


def compressLL_(LL):
    TF = np.asarray([[i is None for i in L] for L in LL])
    TFx = np.where(~np.all(TF, axis=0))[0]
    TFy = np.where(~np.all(TF, axis=1))[0]
    LL_ = l_ind_([l_ind_(L, TFx) for L in LL], TFy)
    return (LL_, TFx, TFy)


def consecutive_(x1d, func_,
                 nn_=3,
                 ffunc_=np.max,
                 efunc_=lambda x: len(x[1:])):
    ts = np.split(np.concatenate(([0], x1d)),
                  np.concatenate(([1], np.where(func_(x1d))[0] + 1)))
    ts = [efunc_(its) for its in ts if len(its) > nn_]
    return ffunc_(ts) if ts else 0.


def _sz(xnd, axis=None):
    if axis is None:
        return xnd.size
    elif isinstance(axis, int):
        return xnd.shape[cyl_(axis, xnd.ndim)]
    elif isIter_(axis, xi=int):
        ind = pd.unique(cyl_(axis, xnd.ndim))
        return np.prod(np.asarray(xnd.shape)[ind])
    else:
        raise Exception(f"I don't understand axis={axis!r}")


def shp_drop_(shp, axis=None, replace=None):
    if axis is not None:
        axis = sorted(cyl_((axis,) if not isIter_(axis) else axis, len(shp)))
        if replace is None:
            return tuple(ii for i, ii in enumerate(shp) if i not in axis)
        else:
            tmp = (replace if i == axis[0] else ii for i, ii in enumerate(shp)
                   if i not in axis[1:])
            return tuple(flt_(tmp))
    else:
        return shp


def flt_ndim_(xnd, dim0, ndim):
    dim0 = cyl_(dim0, xnd.ndim)
    tmp_ = tuple(-1 if i == dim0 else ii for i, ii in enumerate(xnd.shape)
                 if i not in range(dim0 + 1, dim0 + ndim))
    return xnd.reshape(tmp_)


def aggr_func_(xnd, *V, axis=None, func_=np.ma.mean, uniqV=False):
    #0 checking input arguments
    arr = np.asarray(xnd)
    if len(V) == 1:
        lbl = np.asarray(V[0]).ravel()
    elif len(V) > 1:
        lbl = np.asarray(el_join_([np.asarray(i).ravel() for i in V]))
    else:
        raise Exception("at least one label array is required, "
                        "but none is provided!")
    arr, axis = (arr.ravel(), -1) if axis is None else (arr, axis)
    print(lbl)
    if lbl.size != _sz(arr, axis=axis):
        raise Exception("input arguments not matching!")
    uV = pd.unique(lbl)
    nshp = shp_drop_(arr.shape, axis=axis, replace=uV.size)
    if isIter_(axis, xi=int):
        axis = np.unique(cyl_(axis, arr.ndim))
        if not all(np.diff(axis) == 1):
            tmp = tuple(flt_((axis if i == axis[0] else i
                              for i in range(arr.ndim)
                              if i not in axis[1:])))
            arr = arr.transpose(tmp)
        arr = flt_ndim_(arr, axis[0], len(axis))
        naxi = axis[0]
    else:
        naxi = axis
    o = np.empty(nshp)
    for i, ii in enumerate(uV):
        ind_l = ind_s_(len(nshp), naxi, i)
        ind_r = ind_s_(arr.ndim, naxi, lbl==ii)
        o[ind_l] = func_(arr[ind_r], axis=naxi)
    return (o, uV) if uniqV else o


def el_join_(caL, jointer='.'):
    return [jointer.join(iter_str_(i)) for i in zip(*caL)]


def windds2uv_(winds, windd):
    tmp = np.deg2rad(windd)
    return (- np.sin(tmp) * winds,
            - np.cos(tmp) * winds)

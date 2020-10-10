import iris
from iris.coords import DimCoord
import numpy as np
import glob
import warnings
from uuuu.ffff import pure_fn_, robust_bc2_
from uuuu.bbbb import load_fx_
from uuuu.cccc import maskLS_cube, get_xyd_cube


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    idir = '/nobackup/rossby22/sm_chali/DATA/energi/res/cmip5/GLB/'
    fxdir = '/nobackup/rossby22/sm_chali/DATA/fx/'
    fn = glob.glob(idir + 'SIC_CSI*.nc')
    #o0 = iris.load_cube('/nobackup/rossby22/sm_chali/DATA/energi/res/cmip5/'
    #                    'GLB/SST_MRI-CGCM3_rcp85_r1i1p1_GLB_year_gwl2.nc')
    o0 = iris.load_cube('/nobackup/rossby22/sm_chali/DATA/'
                        'fx/sftlf_fx_CSIRO-Mk3-6-0_historical_r0i0p0.nc')
    m = o0.data > 50
    for i, ifn in enumerate(fn):
        o = iris.load_cube(ifn)
        o = iris.util.mask_cube(o, robust_bc2_(m, o.shape, get_xyd_cube(o)))
        iris.save(o, ifn.replace(idir, idir.replace('cmip5/GLB/','')))


if __name__ == '__main__':
    main()

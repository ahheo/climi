import iris
import iris.coord_categorisation as cat
import numpy as np
import glob


def main():
    idir = '/nobackup/rossby22/sm_chali/DATA/energi/res/obs/SWE/'
    fn = glob.glob(idir + 'DTR_*month*.nc')
    for ifn in fn:
        print(ifn)
        o = iris.load_cube(ifn)
        try:                                                                    
            cat.add_month(o, 'time', name='month')                           
        except ValueError:                                                      
            continue
        iris.save(o, ifn.replace(idir, idir.replace('SWE/','')))


if __name__ == '__main__':
    main()

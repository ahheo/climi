import iris
from iris.coords import DimCoord
import numpy as np
import glob


def main():
    idir = '/nobackup/rossby24/users/sm_chali/DATA/energi/res/h248/'\
           'cordex/EUR11/grp1/'
    fn = glob.glob(idir + '*_*_CNRM-ALADIN53_*.nc')
    epochs = {}
    for i, ifn in enumerate(fn):
        o = iris.load_cube(ifn)
        if i == 0:
            cx = o.coord('projection_x_coordinate')
            cy = o.coord('projection_y_coordinate')
            cxd = o.coord_dims(cx)
            cyd = o.coord_dims(cy)
            cs = iris.coord_systems.LambertConformal(central_lat=49.5,
                                                     central_lon=10.5,
                                                     false_easting=0.0,
                                                     false_northing=0.0,
                                                     secant_latitudes=(49.5,),
                                                     ellipsoid=None)
            cxx = DimCoord(np.linspace(cx.points[0], cx.points[-1],
                                       len(cx.points)),
                           standard_name=cx.standard_name,
                           units=cx.units,
                           long_name=cx.long_name,
                           var_name=cx.var_name,
                           coord_system=cs)
            cyy = DimCoord(np.linspace(cy.points[0], cy.points[-1],
                                       len(cy.points)),
                           standard_name=cy.standard_name,
                           units=cy.units,
                           long_name=cy.long_name,
                           var_name=cy.var_name,
                           coord_system=cs)
        o.remove_coord('projection_x_coordinate')
        o.remove_coord('projection_y_coordinate')
        o.add_dim_coord(cxx, cxd)
        o.add_dim_coord(cyy, cyd)
        iris.save(o, ifn.replace(idir, idir.replace('grp1/','')))


if __name__ == '__main__':
    main()

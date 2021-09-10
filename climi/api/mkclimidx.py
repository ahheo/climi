from climi.uuuu import *
from climi.climidx import *

import numpy as np
import iris
import iris.coord_categorisation as ica
import os
import re
import yaml
import time
import warnings
import logging
import argparse

from time import localtime, strftime


_here_ = get_path_(__file__)


_djn = os.path.join


# INDEX DICT: FORMAT:
#     NAME: (#, TR_i, VARIABLES, FUNCTION, TR_o, METADATA, GROUPS)
# where TR_i, TR_o are input and output temporal resolution.
# acceptable values and desciption:
#     TR_i: ('mon', 'day', '6hr', '3hr', '1hr')
#     VARIABLES: input variables (using the same names as CORDEX/CMIPX)
#     FUNCTION (most fr. climidx module):
#         None or string: using pSTAT_cube() fr. module uuuu;
#                         see _mm() or _dd()
#         function type1: with cube(s) as input(s); see _d0()
#         function type2: with 1d array (np) as input(s); see _d1()
#     TR_o: ('month', 'season', 'year', 'hour-month', 'hour-season', 'hour')
#         PS: 'month' is exactly 'month-year'; similar for 'season',
#             'hour-month', 'hour-season', and 'hour'
#     METADATA:
#         a diction:
#             name: try standard name first; if failed long_name instead
#             units:
#             attrU: some notes
#         None: nothing modified but following the input nc files
#     GROUPS (for batch application;
#             one or more of the following;
#             free to add more):
#         'p': precipitation related
#         't': temperature related
#         'w': wind related
#         'r': radiation related
#         'c': consecutive days
i__ = {
    'ET': (1, 'mon', ['evspsbl'],
        None, ['year'], None, 'p'),
    'HumiWarmDays': (2, 'day', ['hurs', 'tas'],
        dHumiWarmDays_cube, ['season'],
        dict(name='humid warm days', units='days',
             attrU={'CLIMI': 'days with hurs > 90% & tas > 10 degree C'}),
        'ht'),
    'EffPR': (3, 'mon', ['evspsbl', 'pr'],
        None, ['season', 'year'],
        dict(name='effective precipitation',
             attrU={'CLIMI': 'pr - evspsbl'}),
        'p'),
    'PR7Dmax': (4, 'day', ['pr'],
        dPr7_, ['year'],
        dict(name='max 7-day precipitation', units='days',
             attrU={'CLIMI': 'maximum of rolling 7-day mean precipitation'}),
        'p'),
    'LnstDryDays': (5, 'day', ['pr'],
        dLongestDryDays_, ['season'],
        dict(name='longest dry days', units='days',
             attrU={'CLIMI': 'longest dry spell defined as consecutive days '
                             'with daily pr < 1 mm'}),
        'p'),
    'DryDays': (6, 'day', ['pr'],
        dDryDays_, ['year'],
        dict(name='dry days', units='days',
             attrU={'CLIMI': 'total days with daily pr < 1 mm'}),
        'p'),
    'PR': (7, 'mon', ['pr'],
        None, ['season', 'year'], None, 'p'),
    'PRmax': (8, 'day', ['pr'],
        'MAX', ['month', 'year'],
        dict(name='maximum precipitation rate'),
        'p'),
    'PRRN': (9, 'mon', ['pr', 'prsn'],
        None, ['season', 'year'], dict(name='rainfall flux'), 'p'),
    'PRSN': (10, 'mon', ['pr', 'prsn'],
        None, ['season', 'year'], None, 'p'),
    'NetRO': (11, 'mon', ['mrro'],
        None, ['year'], dict(attrU={'CLIMI': 'season: amjjas'}), 'p'),
    'SD': (12, 'mon', ['sund'],
        None, ['year'], None, 'r'),
    'RLDS': (13, 'mon', ['rlds'],
        None, ['season'], None, 'r'),
    'RSDS': (14, 'mon', ['rsds'],
        None, ['season'], None, 'r'),
    'CoolingDegDay': (15, 'day', ['tasmax'],
        dDegDay_, ['month', 'year'],
        dict(name='degree day for cooling',
             attrU={'CLIMI': 'degree day for tasmax > 20 degree C'}),
        't'),
    'ConWarmDays': (16, 'day', ['tasmax'],
        dConWarmDays_, ['year'],
        dict(name='consecutive warm days', units='days',
             attrU={'CLIMI': 'longest warm spell defined as consecutive days '
                             'with tasmax > 20 degree C'}),
        'tc'),
    'TX': (17, 'mon', ['tasmax'],
        None, ['year'], None, 't'),
    'WarmDays': (18, 'day', ['tasmax'],
        dWarmDays_, ['season', 'year'],
        dict(name='warm days', units='days',
             attrU={'CLIMI': 'total days with tasmax > 20 degree C'}),
        't'),
    'ColdDays': (19, 'day', ['tasmax'],
        dColdDays_, ['season', 'year'],
        dict(name='cold days', units='days',
             attrU={'CLIMI': 'total days with tasmax < -7 degree C'}),
        't'),
    'DegDay20': (20, 'day', ['tas'],
        dDegDay_, ['year'],
        dict(name='degree day warmer than 20 degree C',
             attrU={'CLIMI': 'degree day for tas > 20 degree C'}),
        't'),
    'DegDay8': (21, 'day', ['tas'],
        dDegDay8_vegSeason_, ['year'],
        dict(name='degree day under vegetation season',
             attrU={'CLIMI': 'degree day for tas > 8 degree C under '
                             'vegetation season (tas > 5 degree C)'}),
        't'),
    'DegDay17': (22, 'day', ['tas'],
        dDegDay_, ['year'],
        dict(name='degree day cooler than 17 degree C',
             attrU={'CLIMI': 'degree day for tas < 17 degree C'}),
        't'),
    'TN': (23, 'mon', ['tasmin'],
        None, ['year'], None, 't'),
    'SpringFrostDayEnd': (24, 'day', ['tasmin'],
        dEndSpringFrost_, ['year'],
        dict(name='end of spring frost', units=1,
             attrU={'CLIMI': 'last day with tasmin < 0 degree C; '
                             'no later than 213.'}),
        't'),
    'FrostDays': (25, 'day', ['tasmin'],
        dFrostDays_, ['season', 'year'],
        dict(name='frost days', units='days',
             attrU={'CLIMI': 'total days with tasmin < 0 degree C'}),
        't'),
    'TropicNights': (26, 'day', ['tasmin'],
        dTropicNights_, ['year'],
        dict(name='tropical nights', units='days',
             attrU={'CLIMI': 'total days with tasmin > 17 degree C'}),
        't'),
    'ZeroCrossingDays': (27, 'day', ['tasmax', 'tasmin'],
        dZeroCrossingDays_cube, ['season'],
        dict(name='zero-crossing days', units='days',
             attrU={'CLIMI': 'days with tasmin < 0 degree C and '
                             'tasmax > 0 degree C'}),
        't'),
    'VegSeasonDayEnd-5': (28, 'day', ['tas'],
        dStartEndVegSeason_, ['year'],
        dict(name='vegetation season (5) end', units=1,
             attrU={'CLIMI': 'vegetation season: from first day to last day '
                             'with4-day rolling mean tas > 5 degree C'}),
        't'),
    'VegSeasonDayEnd-2': (29, 'day', ['tas'],
        dStartEndVegSeason_, ['year'],
        dict(name='vegetation season (2) end', units=1,
             attrU={'CLIMI': 'vegetation season: from first day to last day '
                             'with4-day rolling mean tas > 2 degree C'}),
        't'),
    'VegSeasonDayStart-5': (30, 'day', ['tas'],
        dStartEndVegSeason_, ['year'],
        dict(name='vegetation season (5) start', units=1,
             attrU={'CLIMI': 'vegetation season: from first day to last day '
                             'with4-day rolling mean tas > 5 degree C'}),
        't'),
    'VegSeasonDayStart-2': (31, 'day', ['tas'],
        dStartEndVegSeason_, ['year'],
        dict(name='vegetation season (2) start', units=1,
             attrU={'CLIMI': 'vegetation season: from first day to last day '
                             'with4-day rolling mean tas > 2 degree C'}),
        't'),
    'VegSeasonLength-5': (32, 'day', ['tas'],
        dStartEndVegSeason_, ['year'],
        dict(name='vegetation season (5) length', units=1,
             attrU={'CLIMI': 'vegetation season: from first day to last day '
                             'with4-day rolling mean tas > 5 degree C'}),
        't'),
    'VegSeasonLength-2': (33, 'day', ['tas'],
        dStartEndVegSeason_, ['year'],
        dict(name='vegetation season (2) length', units=1,
             attrU={'CLIMI': 'vegetation season: from first day to last day '
                             'with4-day rolling mean tas > 2 degree C'}),
        't'),
    'SfcWind': (34, 'day', ['sfcWind', 'uas', 'vas'],
        None, ['month', 'season', 'year'], None, 'w'),
    'WindGustMax': (35, 'day', ['wsgsmax'],
        'MAX', ['year'], dict(name='maximum wind gust'), 'w'),
    'WindyDays': (36, 'day', ['wsgsmax'],
        dWindyDays_, ['season', 'year'],
        dict(name='windy days', units='days',
             attrU={'CLIMI': 'total days with wsfsmax > 21 m s-1'}),
        'w'),
    'PRgt10Days': (37, 'day', ['pr'],
        dExtrPrDays_, ['season', 'year'],
        dict(name='days with heavy precipitation', units='days',
             attrU={'CLIMI': 'total days with daily pr > 10 mm'}),
        'p'),
    'PRgt25Days': (38, 'day', ['pr'],
        dExtrPrDays_, ['season', 'year'],
        dict(name='days with extreme precipitation', units='days',
             attrU={'CLIMI': 'total days with daily pr > 25 mm'}),
        'p'),
    'SncDays': (39, 'day', ['snc'],
        dSncDays_, ['year'],
        dict(name='days with snow cover', units='days',
             attrU={'CLIMI': 'total days with snc > 0'}),
        'p'),
    'Snd10Days': (40, 'day', ['snd'],
        dSndLE10Days_, ['year'],
        dict(name='days with 0-10cm snow-depth', units='days',
             attrU={'CLIMI': 'total days with 0 < snd <= 10 cm'}),
        'p'),
    'Snd20Days': (41, 'day', ['snd'],
        dSndGT10LE20Days_, ['year'],
        dict(name='days with 10-20cm snow-depth', units='days',
             attrU={'CLIMI': 'total days with 10 < snd <= 20 cm'}),
       'p'),
    'SNWmax': (42, 'day', ['snw'],
        'MAX', ['year'], dict(name='maximum surface snow amount'), 'p'),
    'TAS': (43, 'mon', ['tas'],
        None, ['year'], None, 't'),
    'DTR': (44, 'mon', ['tasmax', 'tasmin'],
        mDTR_, ['month'],
        dict(name='daily temperature range',
             attrU={'CLIMI': 'tasmax - tasmin'}),
        't'),
    'Rho925': (45, 'day', ['ta925', 'hus925', 'ps'],
        None, ['month'],
        dict(name='air density at 925 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
       'w'),
    'RhoS': (46, 'day', ['tas', 'huss', 'ps'],
        None, ['month'],
        dict(name='near-surface air density', units='kg m-3'),
        'w'),
    'SuperCooledPR': (47, 'day', ['tas', 'pr', 'ps', 'ta925', 'ta850', 'ta700',
                                  'hus925', 'hus850', 'hus700'],
        dFreezRainDays_, ['year'],
        dict(name='day with super cooled precipitation', units='days'),
        'p'),
    'Snc25Days': (48, 'day', ['snc'],
        dSncDays_, ['year'],
        dict(name='days with snow cover', units='days',
             attrU={'CLIMI': 'total days with snc > 25%'}),
        'p'),
    'R5OScw': (49, 'day', ['pr', 'tas', 'snc', 'snw'],
        dRainOnSnow_, ['year'],
        dict(name='days with rain on snow', units='days',
             attrU={'CLIMI': 'total days with snc > 25% and daily prrn > 5 mm '
                             ' and snw > 3 mm'}),
        'p'),
    'R1OScw': (50, 'day', ['pr', 'tas', 'snc', 'snw'],
        dRainOnSnow_, ['year'],
        dict(name='days with rain on snow', units='days',
             attrU={'CLIMI': 'total days with snc > 25% and daily prrn > 1 mm '
                             ' and snw > 3 mm'}),
        'p'),
    'R5OSc': (51, 'day', ['pr', 'tas', 'snc'],
        dRainOnSnow_, ['year'],
        dict(name='days with rain on snow', units='days',
             attrU={'CLIMI': 'total days with snc > 25% '
                             'and daily prrn > 5 mm'}),
        'p'),
    'R1OSc': (52, 'day', ['pr', 'tas', 'snc'],
        dRainOnSnow_, ['year'],
        dict(name='days with rain on snow', units='days',
             attrU={'CLIMI': 'total days with snc > 25% '
                             'and daily prrn > 1 mm'}),
        'p'),
    'PRSNmax': (53, 'day', ['prsn'],
        'MAX', ['year'], dict(name='maximum snowfall flux'), 'p'),
    'CalmDays': (54, 'day', ['sfcWind', 'uas', 'vas'],
        dCalmDays_, ['season', 'year'],
        dict(name='calm days', units='days',
             attrU={'CLIMI': 'total days with sfcWind < 2 m s-1'}),
        'w'),
    'ConCalmDays': (55, 'day', ['sfcWind', 'uas', 'vas'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with sfcWind < 2 m s-1'}),
        'wc'),
    'Wind975': (56, 'day', ['ua975', 'va975', 'ps'],
        None, ['month', 'season', 'year'],
        dict(name='wind speed at 975 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'w'),
    'Wind975toSfc': (57, 'day', ['ua975', 'va975', 'sfcWind', 'ps',
                                 'uas', 'vas'],
        None, ['season', 'year'],
        dict(name='ratio of wind speed at 975 hPa to surface wind speed',
             units=1,
             attrU={'CLIMI': 'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'w'),
    'ColdRainDays': (58, 'day', ['pr', 'tas'],
        dColdRainDays_, ['year'],
        dict(name='days with cold rain', units='days',
             attrU={'CLIMI': 'total days with daily prrn > 0 mm and '
                             '0.58 degree C < tas < 2 degree C'}),
        'p'),
    'ColdRainGT10Days': (59, 'day', ['pr', 'tas'],
        dColdRainDays_, ['year'],
        dict(name='days with cold rain', units='days',
             attrU={'CLIMI': 'total days with daily prrn > 10 mm and '
                             '0.58 degree C < tas < 2 degree C'}),
        'p'),
    'ColdRainGT20Days': (60, 'day', ['pr', 'tas'],
        dColdRainDays_, ['year'],
        dict(name='days with cold rain', units='days',
             attrU={'CLIMI': 'total days with daily prrn > 20 mm and '
                             '0.58 degree C < tas < 2 degree C'}),
        'p'),
    'WarmSnowDays': (61, 'day', ['pr', 'tas'],
        dWarmSnowDays_, ['year'],
        dict(name='days with warm snow', units='days',
             attrU={'CLIMI': 'total days with daily prsn > 0 mm and '
                             '-2 degree C < tas < 0.58 degree C'}),
        'p'),
    'WarmSnowGT10Days': (62, 'day', ['pr', 'tas'],
        dWarmSnowDays_, ['year'],
        dict(name='days with warm snow', units='days',
             attrU={'CLIMI': 'total days with daily prsn > 10 mm and '
                             '-2 degree C < tas < 0.58 degree C'}),
        'p'),
    'WarmSnowGT20Days': (63, 'day', ['pr', 'tas'],
        dWarmSnowDays_, ['year'],
        dict(name='days with warm snow', units='days',
             attrU={'CLIMI': 'total days with daily prsn > 20 mm and '
                             '-2 degree C < tas < 0.58 degree C'}),
        'p'),
    'ColdPRRNdays': (64, 'day', ['pr', 'tas', 'prsn'],
        dColdPRRNdays_, ['year'],
        dict(name='days with cold rain', units='days',
             attrU={'CLIMI': 'total days with daily prrn > 0 mm and '
                             '0.58 degree C < tas < 2 degree C'}),
        'p'),
    'ColdPRRNgt10Days': (65, 'day', ['pr', 'tas', 'prsn'],
        dColdPRRNdays_, ['year'],
        dict(name='days with cold rain', units='days',
             attrU={'CLIMI': 'total days with daily prrn > 10 mm and '
                             '0.58 degree C < tas < 2 degree C'}),
        'p'),
    'ColdPRRNgt20Days': (66, 'day', ['pr', 'tas', 'prsn'],
        dColdPRRNdays_, ['year'],
        dict(name='days with cold rain', units='days',
             attrU={'CLIMI': 'total days with daily prrn > 20 mm and '
                             '0.58 degree C < tas < 2 degree C'}),
        'p'),
    'WarmPRSNdays': (67, 'day', ['tas', 'prsn'],
        dWarmPRSNdays_, ['year'],
        dict(name='days with warm snow', units='days',
             attrU={'CLIMI': 'total days with daily prsn > 0 mm and '
                             '-2 degree C < tas < 0.58 degree C'}),
        'p'),
    'WarmPRSNgt10Days': (68, 'day', ['tas', 'prsn'],
        dWarmPRSNdays_, ['year'],
        dict(name='days with warm snow', units='days',
             attrU={'CLIMI': 'total days with daily prsn > 10 mm and '
                             '-2 degree C < tas < 0.58 degree C'}),
        'p'),
    'WarmPRSNgt20Days': (69, 'day', ['tas', 'prsn'],
        dWarmPRSNdays_, ['year'],
        dict(name='days with warm snow', units='days',
             attrU={'CLIMI': 'total days with daily prsn > 20 mm and '
                             '-2 degree C < tas < 0.58 degree C'}),
        'p'),
    'SST': (70, 'mon', ['tos'],
        None, ['season', 'year'], None, 't'),
    'SIC': (71, 'mon', ['sic'],
        None, ['season', 'year'], None, 'tp'),
    'Rho975': (72, 'day', ['ta975', 'hus975', 'ps'],
        None, ['month'],
        dict(name='air density at 975 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'w'),
    'h6SuperCooledPR': (73, '6hr', ['tas', 'pr', 'ps',
                                    'ta925', 'ta850', 'ta700',
                                    'hus925', 'hus850', 'hus700'],
        dFreezRainDays_, ['year'],
        dict(name='number of super cooled precipitation events', units=1,
             attrU={'CLIMI': 'interval of 6hr'}),
        'p'),
    'Wind925': (74, 'day', ['ua925', 'va925', 'ps'],
        None, ['month', 'season', 'year'],
        dict(name='wind speed at 925 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 925 hPa masked if ps '
                             'is available'}),
        'w'),
    'Wind925toSfc': (75, 'day', ['ua925', 'va925', 'sfcWind', 'ps',
                                 'uas', 'vas'],
        None, ['season', 'year'],
        dict(name='ratio of wind speed at 925 hPa to surface wind speed',
             units=1,
             attrU={'CLIMI': 'with grid cells where ps < 925 hPa masked if ps '
                             'is available'}),
        'w'),
    'FirstDayWithoutFrost': (76, 'day', ['tasmin'],
        dFirstDayWithoutFrost_, ['year'],
        dict(name='first day without frost', units=1,
             attrU={'CLIMI': 'first day with tasmin > 0 degree C'}),
        't'),
    'CalmDays925': (77, 'day', ['ua925', 'va925', 'ps'],
        dCalmDays_, ['year'],
        dict(name='calm days at 925 hPa', units='days',
             attrU={'CLIMI': 'total days with wind speed at 925 hPa '
                             '< 2 m s-1; '
                             'with grid cells where ps < 925 hPa masked if ps '
                             'is available'}),
        'w'),
    'ConCalmDays925': (78, 'day', ['ua925', 'va925', 'ps'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days at 925 hPa', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with wind speed at 925 hPa < 2 m s-1; '
                             'with grid cells where ps < 925 hPa masked if ps '
                             'is available'}),
        'wc'),
    'CalmDays975': (79, 'day', ['ua975', 'va975', 'ps'],
        dCalmDays_, ['year'],
        dict(name='calm days at 975 hPa', units='days',
             attrU={'CLIMI': 'total days with wind speed at 975 hPa '
                             '< 2 m s-1; '
                             'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'w'),
    'ConCalmDays975': (80, 'day', ['ua975', 'va975', 'ps'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days at 975 hPa', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with wind speed at 975 hPa < 2 m s-1; '
                             'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'wc'),
    'MinusDays': (81, 'day', ['tas'],
        dMinusDays_, ['year'],
        dict(name='minus days', units='days',
             attrU={'CLIMI': 'total days with tas < 0 degree C'}),
        't'),
    'FreezingDays': (82, 'day', ['tasmax'],
        dFreezingDays_, ['year'],
        dict(name='freezing days', units='days',
             attrU={'CLIMI': 'total days with tasmax < 0 degree C'}),
       't'),
    'ColdRainWarmSnowDays': (83, 'day', ['pr', 'tas'],
        dColdRainWarmSnowDays_, ['year'],
        dict(name='days with cold rain or warm snow', units='days',
             attrU={'CLIMI': 'total days with daily pr > 0 mm and '
                             '-2 degree C < tas < 2 degree C'}),
        'p'),
    'Wind50m': (84, 'day', ['ua50m', 'va50m'],
        None, ['month', 'season', 'year'],
        dict(name='wind speed at 50 m height'),
        'w'),
    'Rho50m': (85, 'day', ['ta50m', 'hus50m', 'p50m'],
        None, ['month'],
        dict(name='air density at 50 m height', units='kg m-3'),
        'w'),
    'Wind100m': (86, 'day', ['ua100m', 'va100m'],
        None, ['month', 'season', 'year'],
        dict(name='wind speed at 100 m height'),
        'w'),
    'Rho100m': (87, 'day', ['ta100m', 'hus100m', 'p100m'],
        None, ['month'],
        dict(name='air density at 100 m height', units='kg m-3'),
        'w'),
    'Wind200m': (88, 'day', ['ua200m', 'va200m'],
        None, ['month', 'season', 'year'],
        dict(name='wind speed at 200 m height'),
        'w'),
    'Rho200m': (89, 'day', ['ta200m', 'hus200m', 'p200m'],
        None, ['month'],
        dict(name='air density at 200 m height', units='kg m-3'),
        'w'),
    'Wind950': (90, 'day', ['ua950', 'va950'],
        None, ['month', 'season', 'year'],
        dict(name='wind speed at 950 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 950 hPa masked if ps '
                             'is available'}),
        'w'),
    'Rho950': (91, 'day', ['ta950', 'hus950', 'ps'],
        None, ['month'],
        dict(name='air density at 950 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 950 hPa masked if ps '
                             'is available'}),
        'w'),
    'Wind950toSfc': (92, 'day', ['ua950', 'va950', 'sfcWind', 'ps',
                                 'uas', 'vas'],
        None, ['season', 'year'],
        dict(name='ratio of wind speed at 950 hPa to surface wind speed',
             units=1,
             attrU={'CLIMI': 'with grid cells where ps < 950 hPa masked if ps '
                             'is available'}),
        'w'),
    'Wind900': (93, 'day', ['ua900', 'va900'],
        None, ['month', 'season', 'year'],
        dict(name='wind speed at 900 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 900 hPa masked if ps '
                             'is available'}),
        'w'),
    'Rho900': (94, 'day', ['ta900', 'hus900', 'ps'],
        None, ['month'],
        dict(name='air density at 900 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 900 hPa masked if ps '
                             'is available'}),
        'w'),
    'Wind900toSfc': (95, 'day', ['ua900', 'va900', 'sfcWind', 'ps',
                                 'uas', 'vas'],
        None, ['season', 'year'],
        dict(name='ratio of wind speed at 900 hPa to surface wind speed',
             units=1,
             attrU={'CLIMI': 'with grid cells where ps < 900 hPa masked if ps '
                             'is available'}),
        'w'),
    'CalmDays50m': (96, 'day', ['ua50m', 'va50m'],
        dCalmDays_, ['year'],
        dict(name='calm days at 50 m height', units='days',
             attrU={'CLIMI': 'total days with wind speed at 50 m height '
                             '< 2 m s-1'}),
        'w'),
    'ConCalmDays50m': (97, 'day', ['ua50m', 'va50m'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days at 50 m height', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with wind speed at 50 m height < 2 m s-1'}),
        'wc'),
    'CalmDays100m': (98, 'day', ['ua100m', 'va100m'],
        dCalmDays_, ['year'],
        dict(name='calm days at 100 m height', units='days',
             attrU={'CLIMI': 'total days with wind speed at 100 m height '
                             '< 2 m s-1'}),
        'w'),
    'ConCalmDays100m': (99, 'day', ['ua100m', 'va100m'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days at 100 m height', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with wind speed at 100 m height < 2 m s-1'}),
        'wc'),
    'CalmDays200m': (100, 'day', ['ua200m', 'va200m'],
        dCalmDays_, ['year'],
        dict(name='calm days at 200 m height', units='days',
             attrU={'CLIMI': 'total days with wind speed at 200 m height '
                             '< 2 m s-1'}),
        'w'),
    'ConCalmDays200m': (101, 'day', ['ua200m', 'va200m'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days at 200 m height', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with wind speed at 200 m height < 2 m s-1'}),
        'wc'),
    'CalmDays950': (102, 'day', ['ua950', 'va950', 'ps'],
        dCalmDays_, ['year'],
        dict(name='calm days at 950 hPa', units='days',
             attrU={'CLIMI': 'total days with wind speed at 950 hPa '
                             '< 2 m s-1; '
                             'with grid cells where ps < 950 hPa masked if ps '
                             'is available'}),
        'w'),
    'ConCalmDays950': (103, 'day', ['ua950', 'va950'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days at 950 hPa', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with wind speed at 950 hPa < 2 m s-1; '
                             'with grid cells where ps < 950 hPa masked if ps '
                             'is available'}),
        'wc'),
    'CalmDays900': (104, 'day', ['ua900', 'va900', 'ps'],
        dCalmDays_, ['year'],
        dict(name='calm days at 900 hPa', units='days',
             attrU={'CLIMI': 'total days with wind speed at 900 hPa '
                             '< 2 m s-1; '
                             'with grid cells where ps < 900 hPa masked if ps '
                             'is available'}),
        'w'),
    'ConCalmDays900': (105, 'day', ['ua900', 'va900'],
        dConCalmDays_, ['year'],
        dict(name='consecutive calm days at 900 hPa', units='days',
             attrU={'CLIMI': 'longest calm spell defined as consecutive days '
                             'with wind speed at 900 hPa < 2 m s-1; '
                             'with grid cells where ps < 900 hPa masked if ps '
                             'is available'}),
        'wc'),
    'h3SfcWind': (106, '3hr', ['sfcWind', 'uas', 'vas'],
        None, ['hour-month', 'hour-season', 'hour'], None, 'w'),
    'h3RhoS': (107, '3hr', ['tas', 'huss', 'ps'],
        None, ['hour-month'],
        dict(name='near-surface air density', units='kg m-3'),
        'w'),
    'h3Wind975': (108, '3hr', ['ua975', 'va975', 'ps'],
        None, ['hour-month', 'hour-season', 'hour'],
        dict(name='wind speed at 975 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Rho975': (109, '3hr', ['ta975', 'hus975', 'ps'],
        None, ['hour-month'],
        dict(name='air density at 975 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Wind950': (110, '3hr', ['ua950', 'va950', 'ps'],
        None, ['hour-month', 'hour-season', 'hour'],
        dict(name='wind speed at 950 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 950 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Rho950': (111, '3hr', ['ta950', 'hus950', 'ps'],
        None, ['hour-month'],
        dict(name='air density at 950 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 950 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Wind925': (112, '3hr', ['ua925', 'va925', 'ps'],
        None, ['hour-month', 'hour-season', 'hour'],
        dict(name='wind speed at 925 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 925 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Rho925': (113, '3hr', ['ta925', 'hus925', 'ps'],
        None, ['hour-month'],
        dict(name='air density at 925 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 975 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Wind900': (114, '3hr', ['ua900', 'va900', 'ps'],
        None, ['hour-month', 'hour-season', 'hour'],
        dict(name='wind speed at 900 hPa',
             attrU={'CLIMI': 'with grid cells where ps < 900 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Rho900': (115, '3hr', ['ta900', 'hus900', 'ps'],
        None, ['hour-month'],
        dict(name='air density at 900 hPa', units='kg m-3',
             attrU={'CLIMI': 'with grid cells where ps < 900 hPa masked if ps '
                             'is available'}),
        'w'),
    'h3Wind50m': (116, '3hr', ['ua50m', 'va50m'],
        None, ['hour-month', 'hour-season', 'hour'],
        dict(name='wind speed at 50 m height'),
        'w'),
    'h3Rho50m': (117, '3hr', ['ta50m', 'hus50m', 'p50m'],
        None, ['hour-month'],
        dict(name='air density at 50 m height', units='kg m-3'),
        'w'),
    'h3Wind100m': (118, '3hr', ['ua100m', 'va100m'],
        None, ['hour-month', 'hour-season', 'hour'],
        dict(name='wind speed at 100 m height'),
        'w'),
    'h3Rho100m': (119, '3hr', ['ta100m', 'hus100m', 'p100m'],
        None, ['hour-month'],
        dict(name='air density at 100 m height', units='kg m-3'),
        'w'),
    'h3Wind200m': (120, '3hr', ['ua200m', 'va200m'],
        None, ['hour-month', 'hour-season', 'hour'],
        dict(name='wind speed at 200 m height'),
        'w'),
    'h3Rho200m': (121, '3hr', ['ta200m', 'hus200m', 'p200m'],
        None, ['hour-month'],
        dict(name='air density at 200 m height', units='kg m-3'),
        'w')
    }


def _vv2(a, xvn=9):
    a = sorted(a, key=lambda x:(len(x[1]), sorted(x[1][0])), reverse=True)
    while len(a) > 0:
        aa, bb = [a[0][0]], a[0][1]
        del(a[0])
        if len(a) > 0:
            for i in a.copy():
                tmp = ouniqL_(bb + i[1])
                if len(tmp) <= xvn:
                    aa.append(i[0])
                    bb = tmp
                    a.remove(i)
        yield (aa, bb)


def _vv(ii_, tint, subg=None, xvn=9):
    if tint not in ('mon',) and subg is None:
        tmp = [(i, i__[i][2]) for i in ii_ if i__[i][1] == tint]
        return list(_vv2(tmp, xvn))
    else:
        tmp = [i__[i][2] for i in ii_ if i__[i][1] == tint]
        tmp_ = set(flt_l(tmp))
        if tint not in ('mon',) and len(tmp_) > xvn:
            if subg == 'v':
                tmp__ = ss_fr_sl_(list(map(set, tmp)))
                return [([ii for ii in ii_
                          if all([iii in i for iii in i__[ii][2]])], i)
                        for i in tmp__]
            elif subg == 'i':
                ii__ = [ii_[:(len(ii_)//2)], ii_[(len(ii_)//2):]]
                tmp__ = [_vv(i, tint) for i in ii__]
                return nli_(tmp__)
        else:
            return ([i for i in ii_ if i__[i][1] == tint], tmp_)


def _vd_fr_vv(vl):
    if isinstance(vl, tuple):
        vd = dict()
        for i in vl[1]:
            vd.update({'c_' + i: i})
        return (vl[0], vd)
    elif isinstance(vl, list):
        return [_vd_fr_vv(i) for i in vl]
    else:
        raise Exception('check input!')


def _vd(ii_, tint, subg=None, xvn=9):
    return _vd_fr_vv(_vv(ii_, tint, subg, xvn))


def _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, folder='cordex',
         subg=None, xvn=9, user_cfg=None):
    t1 = l__('>>>{}'.format(tint))
    ka0 = dict(dn=dn, gwl=gwl, po_=po_, user_cfg=user_cfg)
    if tint == 'mon':
        f__ = _mclimidx
    elif tint == 'day':
        f__ = _dclimidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    elif tint == '6hr':
        f__ = _h6climidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    elif tint == '3hr':
        f__ = _h3climidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    elif tint == '1hr':
        f__ = _h1climidx
        if y0y1:
            ka0.update(dict(y0y1=y0y1))
    def __xyz(vd):
        ccc = dict()
        if isinstance(vd, tuple):
            t000 = l__("rf__(): {} variables".format(len(vd[1].keys())))
            ll_(', '.join(vd[1].keys()))
            ll_(', '.join(vd[0]))
            tmp, tmp_, tmp__ = [], [], []
            wsyes = False # SURFACE WIND SPEED
            for kk in vd[1].keys():
                if wsyes and kk in ('c_uas', 'c_vas'): # SURFACE WIND SPEED
                    continue # SURFACE WIND SPEED
                if y0y1:
                    cc, ee = rf__(pi_, tint, var=vd[1][kk], period=y0y1,
                                  reg_d=reg_d, folder=folder)
                else:
                    cc, ee = rf__(pi_, tint, var=vd[1][kk], reg_d=reg_d,
                                  folder=folder)
                if cc:
                    tmp.append(kk)
                    if kk == 'c_sfcWind': # SURFACE WIND SPEED
                        wsyes = True # SURFACE WIND SPEED
                if ee == 'cce':
                    tmp_.append(kk)
                elif ee == 'yye':
                    tmp__.append(kk)
                ccc.update({kk: cc})
            if len(tmp__) > 0:
                ll_('YYE: {}'.format(', '.join(tmp__)))
            if len(tmp_) > 0:
                ll_('CCE: {}'.format(', '.join(tmp_)))
            if len(tmp) > 0:
                ll_(', '.join(tmp))
            ll_("rf__()", t000)
            f__(il_=vd[0], **ka0, **ccc)
        elif isinstance(vd, list):
            for i in vd:
                __xyz(i)
        else:
            raise Exception("check out put of '_vd'")
    vD = _vd(il_, tint, subg=subg, xvn=xvn)
    __xyz(vD)
    ll_('<<<{}'.format(tint), t1)


s4 = ('djf', 'mam', 'jja', 'son')


def _get_freq(v_, user_cfg=None):
    vv = v_[0] if isIter_(v_) else v_
    if user_cfg:
        try:
            o = user_cfg['freq_cfg'][vv]
        except:
            o = i__[vv][4]
        return o
    else:
        return i__[vv][4]


def _to1(v_, dgpi, freq=None):
    dn, gwl, po_ = dgpi[:3]

    def _fns(fff):
        return schF_keys_(
                po_,
                '_'.join((i for i in (v_, dn, fff, gwl, '__*') if i)))

    def _fn(fff):
        return _fnfmt(po_, v_, dn, fff, gwl)

    freq = freq if freq else _get_freq(v_, dgpi[4])
    if len(freq) == 1:
        fns = _fns(freq[0])
        if fns:
            o = iris.load(fns)
            cubesv_(concat_cube_(o), _fn(freq[0]))
            for i in fns:
                os.remove(i)
    else:
        for ff in freq:
            fns = _fns(ff)
            if fns:
                o = iris.load(fns)
                cubesv_(concat_cube_(o), _fn(ff))
                for i in fns:
                    os.remove(i)


def _meta(v_, cube):
    pK_ = dict(var_name=re.sub('\W', '_', v_).lower())
    if i__[v_][5]:
        pK_.update(i__[v_][5])
    pst_(cube, **pK_)


def _dn(*terms):
    return '_'.join(i for i in terms if i)


def _fnfmt(idir, *terms):
    nm_ = '{}.nc'.format('_'.join(i for i in terms if i))
    return _djn(idir, nm_)


def _sv(v_, o, dgpi, freq=None, _nm=None):
    dn, gwl, po_ = dgpi[:3]

    def _fn(fff):
        return _fnfmt(po_, v_, dn, fff, gwl, _nm)

    freq = freq if freq else _get_freq(v_, dgpi[4])
    assert isIter_(freq), "type {!r} not acceptable here!".format(type(freq))
    if len(freq) == 1:
        _meta(v_, o)
        cubesv_(o, _fn(freq[0]))
    else:
        for oo, ff in zip(o, freq):
            _meta(v_, oo)
            cubesv_(oo, _fn(ff))


def _dd(v_, cube, dgpi, freq=None, _nm=None):
    if v_ in dgpi[3] and cube:
        freq = freq if freq else _get_freq(v_, dgpi[4])
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                       *freq)
        _sv(v_, o, dgpi, freq=freq, _nm=_nm)
        ll_(v_, t000)


def _d0(v_, cube, dgpi, fA_=(), fK_={}, freq=None):
    if isIter_(v_): # preprocessing regarding required index
        vk_ = v_[0]
        lmsg = '/'.join(('{}',) * len(v_)) + ' {}'
        vids = [i__[i][0] for i in v_]
        lmsg_ = lmsg.format(*vids, v_[0][:8])
    else:
        vk_ = v_
        lmsg_ = '{} {}'.format(i__[v_][0], v_)

    if vk_ not in dgpi[3]:
        return

    freq = freq if freq else _get_freq(vk_, dgpi[4])
    cc = cube if isMyIter_(cube) else (cube,)

    if all(i for i in cc):
        t000 = l__(lmsg_)
        o = i__[vk_][3](*cc, freq, *fA_, **fK_)
        if not isIter_(v_) or len(v_) == 1:
            _sv(vk_, o, dgpi, freq=freq)
        else:
            for i, ii in zip(v_, o):
                _sv(i, ii, dgpi, freq=freq)
        ll_(lmsg_, t000)


def _tt(cube, y0y1=None, mmm=None):
    yr_doy_cube(cube)
    if mmm and isSeason_(mmm):
        seasonyr_cube(cube, mmm)
        yrs = cube.coord('seasonyr').points
        y_y_ = y0y1 if y0y1 else yrs[[0, -1]]
        y_y_ = y_y_ if ismono_(mmmN_(mmm)) else [y_y_[0] + 1, y_y_[-1] - 1]
    else:
        yrs = cube.coord('year').points
        y_y_ = y0y1 if y0y1 else yrs[[0, -1]]
    doy = cube.coord('doy').points
    return (axT_cube(cube), y_y_, yrs, doy)


def _d1(v_, cube, dgpi, y0y1,
        cK_={}, fA_='y', fK_={}, freq=None, out=False):
    if isIter_(v_): # preprocessing regarding required index
        vk_ = v_[0]
        lmsg = '/'.join(('{}',) * len(v_)) + ' {}'
        vids = [i__[i][0] for i in v_]
        lmsg_ = lmsg.format(*vids, v_[0][:8])
    else:
        vk_ = v_
        lmsg_ = '{} {}'.format(i__[v_][0], v_)

    if vk_ not in dgpi[3]:
        return

    freq = freq if freq else _get_freq(vk_, dgpi[4])
    freq = s4 if freq == 'season' else freq
    cc = cube if isMyIter_(cube) else (cube,)

    if any(i is None for i in cc):
        return

    def _inic(y_y_, icK_, mmm=None): # create output cubes
        icK_cp = icK_.copy()
        if mmm:
            icK_cp.update(dict(mmm=mmm))
        return initAnnualCube_(cc[0], y_y_, **icK_cp)

    def _cube_cp(extract_func, mmm): # if extraction regarding 'freq'
        return tuple(extract_func(i, mmm) for i in cc)
            
    def _run_single_freq(x):
        # knowing x
        mmm_ = None
        if isMonth_(x):
            cube_cp = _cube_cp(extract_month_cube, x)
            mmm_ = x
        elif isSeason_(x):
            cube_cp = _cube_cp(extract_season_cube, x)
            mmm_ = x
        elif ff == 'year':
            cube_cp = cc
        else:
            raise Exception("currently only 'year'/month/season acceptable!")
        # axis, y0y1, years of data points, doy of data points, 
        ax_t, y_y_, tyrs, tdoy = _tt(cube_cp[0], y0y1, mmm_)
        if isinstance(cK_, dict):
            o = _inic(y_y_, cK_, mmm_)
        else:
            o = iris.cube.CubeList([_inic(y_y_, i, mmm_) for i in cK_])
        # aditional arguments to index function
        fA = ()
        if 'y' in fA_:
            fA += (tyrs,)
        if 'd' in fA_:
            fA += (tdoy,)
        o_ = _afm_n(cube_cp, ax_t, i__[vk_][3], o, *fA, **fK_)
        o = o_ if o_ else o
        if not isIter_(v_) or len(v_) == 1:
            _sv(vk_, o, dgpi, freq=(x,))
        else:
            for i, ii in zip(v_, o):
                _sv(i, ii, dgpi, freq=(x,))
        return o

    t000 = l__(lmsg_)
    ooo = []
    for ff in freq:
        tmp = _run_single_freq(ff)
        ooo.append(tmp)
        ll_('   >>>>{}'.format(ff), t000)
    ll_(lmsg_, t000)
    if out:
        return ooo[0] if len(ooo) == 1 else ooo


def _mclimidx(dn=None, gwl=None, po_=None, il_=None, user_cfg=None,
              c_pr=None, c_evspsbl=None, c_prsn=None, c_mrro=None,
              c_sund=None, c_rsds=None, c_rlds=None,
              c_tas=None, c_tasmax=None, c_tasmin=None,
              c_tos=None, c_sic=None):

    dgpi = [dn, gwl, po_, il_, user_cfg]
    if any([i is None for i in dgpi[:-1]]):
        raise ValueError("inputs of 'dn', 'gwl', 'po_', 'il_' are mandotory!")

    os.makedirs(po_, exist_ok=True)

    def _mm(v_, cube, freq=None):
        if v_ in il_ and cube:
            freq = freq if freq else _get_freq(v_, dgpi[4])
            o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                           *freq)
            _sv(v_, o, dgpi, freq=freq)

    _mm('PR', c_pr)                                                         #PR
    _mm('ET', c_evspsbl)                                                    #ET
    if 'EffPR' in il_ and c_pr is not None and c_evspsbl is not None:
        o = c_pr.copy(c_pr.data - c_evspsbl.data)
        _mm('EffPR', o)                                                  #EffPR
    _mm('PRSN', c_prsn)                                                   #PRSN
    if 'PRRN' in il_ and c_pr is not None and c_prsn is not None:
        o = c_pr.copy(c_pr.data - c_prsn.data)
        _mm('PRRN', o)                                                    #PRRN
    v_ = 'NetRO'
    if 'NetRO' in il_ and c_mrro is not None:
        o = extract_season_cube(c_mrro, 'amjjas')
        o.attributes.update({'Season': 'amjjas'})
        _mm('NetRO', o)                                                  #NetRO
    _mm('SD', c_sund)                                                       #SD
    _mm('RSDS', c_rsds)                                                   #RSDS
    _mm('RLDS', c_rlds)                                                   #RLDS
    _mm('TAS', c_tas)                                                      #TAS
    _mm('TX', c_tasmax)                                                     #TX
    _mm('TN', c_tasmin)                                                     #TN
    v_ = 'DTR'
    _d0(v_, (c_tasmax, c_tasmin), dgpi)                                    #DTR
    _mm('SST', c_tos)                                                      #SST
    _mm('SIC', c_sic)                                                      #SIC


def _dclimidx(dn=None, gwl=None, po_=None, il_=None, user_cfg=None, y0y1=None,
              c_hurs=None, c_huss=None, c_hus975=None, c_hus950=None,
              c_hus925=None, c_hus900=None, c_hus850=None, c_hus700=None,
              c_hus50m=None, c_hus100m=None, c_hus200m=None,
              c_pr=None, c_prsn=None,
              c_ps=None, c_p50m=None, c_p100m=None, c_p200m=None,
              c_snc=None, c_snd=None, c_snw=None,
              c_tas=None, c_tasmax=None, c_tasmin=None,
              c_ta975=None, c_ta950=None, c_ta925=None, c_ta900=None,
              c_ta850=None, c_ta700=None, c_ta50m=None, c_ta100m=None,
              c_ta200m=None,
              c_ua975=None, c_ua950=None, c_ua925=None, c_ua900=None,
              c_ua50m=None, c_ua100m=None, c_ua200m=None, c_uas=None,
              c_va975=None, c_va950=None, c_va925=None, c_va900=None,
              c_va50m=None, c_va100m=None, c_va200m=None, c_vas=None,
              c_sfcWind=None, c_wsgsmax=None):

    dgpi = [dn, gwl, po_, il_, user_cfg]
    if any([i is None for i in dgpi[:-1]]):
        raise ValueError("inputs of 'dn', 'gwl', 'po_', 'il_' are mandotory!")

    os.makedirs(po_, exist_ok=True)

    _dd('WindGustMax', c_wsgsmax, dgpi)                            #WindGustMax
    v_ = 'WindyDays'
    _d0(v_, c_wsgsmax, dgpi)                                         #WindyDays
    if (c_sfcWind is None and c_uas and c_vas and
        any([i in il_ for i in ['SfcWind', 'CalmDays', 'ConCalmDays',
                                'Wind975toSfc', 'Wind950toSfc',
                                'Wind925toSfc', 'Wind900toSfc']])):
        c_sfcWind = ws_cube(c_uas, c_vas)
        pst_(c_sfcWind, 'surface wind speed', var_name='sfcWind')
    o = None
    if any([i in il_ for i in ['Wind975', 'Wind975toSfc', 'CalmDays975',
                               'ConCalmDays975']]):
        if c_ua975 and c_va975:
            o = ws_cube(c_ua975, c_va975)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 97500.)
    _dd('Wind975', o, dgpi)                                            #Wind975
    _d0('CalmDays975', o, dgpi)                                    #CalmDays975
    _d1('ConCalmDays975', o, dgpi, y0y1)                        #ConCalmDays975
    if 'Wind975toSfc' in  il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
        _dd('Wind975toSfc', o, dgpi)                              #Wind975toSfc
    o = None
    if any([i in il_ for i in ['Wind950', 'Wind950toSfc', 'CalmDays950',
                               'ConCalmDays950']]):
        if c_ua950 and c_va950:
            o = ws_cube(c_ua950, c_va950)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 95000.)
    _dd('Wind950', o, dgpi)                                            #Wind950
    _d0('CalmDays950', o, dgpi)                                    #CalmDays950
    _d1('ConCalmDays950', o, dgpi, y0y1)                        #ConCalmDays950
    if 'Wind950toSfc' in  il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
        _dd('Wind950toSfc', o, dgpi)                              #Wind950toSfc
    o = None
    if any([i in il_ for i in ['Wind925', 'Wind925toSfc', 'CalmDays925',
                               'ConCalmDays925']]):
        if c_ua925 and c_va925:
            o = ws_cube(c_ua925, c_va925)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 92500.)
    _dd('Wind925', o, dgpi)                                            #Wind925
    _d0('CalmDays925', o, dgpi)                                    #CalmDays925
    _d1('ConCalmDays925', o, dgpi, y0y1)                        #ConCalmDays925
    if 'Wind925toSfc' in il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
        _dd('Wind925toSfc', o, dgpi)                              #Wind925toSfc
    o = None
    if any([i in il_ for i in ['Wind900', 'Wind900toSfc', 'CalmDays900',
                               'ConCalmDays900']]):
        if c_ua900 and c_va900:
            o = ws_cube(c_ua900, c_va900)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 90000.)
    _dd('Wind900', o, dgpi)                                            #Wind900
    _d0('CalmDays900', o, dgpi)                                    #CalmDays900
    _d1('ConCalmDays900', o, dgpi, y0y1)                        #ConCalmDays900
    if 'Wind900toSfc' in  il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
    o = None
    if any([i in il_ for i in ['Wind50m', 'CalmDays50m', 'ConCalmDays50m']]):
        if c_ua50m and c_va50m:
            o = ws_cube(c_ua50m, c_va50m)
    _dd('Wind50m', o, dgpi)                                            #Wind50m
    _d0('CalmDays50m', o, dgpi)                                    #CalmDays50m
    _d1('ConCalmDays50m', o, dgpi, y0y1)                        #ConCalmDays50m
    o = None
    if any([i in il_ for i in ['Wind100m', 'CalmDays100m', 'ConCalmDays100m']]):
        if c_ua100m and c_va100m:
            o = ws_cube(c_ua100m, c_va100m)
    _dd('Wind100m', o, dgpi)                                          #Wind100m
    _d0('CalmDays100m', o, dgpi)                                  #CalmDays100m
    _d1('ConCalmDays100m', o, dgpi, y0y1)                      #ConCalmDays100m
    o = None
    if any([i in il_ for i in ['Wind200m', 'CalmDays200m', 'ConCalmDays200m']]):
        if c_ua200m and c_va200m:
            o = ws_cube(c_va200m, c_va200m)
    _dd('Wind200m', o, dgpi)                                          #Wind200m
    _d0('CalmDays200m', o, dgpi)                                  #CalmDays200m
    _d1('ConCalmDays200m', o, dgpi, y0y1)                      #ConCalmDays200m
    _dd('SfcWind', c_sfcWind, dgpi)                                    #SfcWind
    _d0('CalmDays', c_sfcWind, dgpi)                                  #CalmDays
    _d1('ConCalmDays', c_sfcWind, dgpi, y0y1)                      #ConCalmDays
    _d0('WarmDays', c_tasmax, dgpi)                                   #WarmDays
    _d0('ColdDays', c_tasmax, dgpi)                                   #ColdDays
    _d0('FreezingDays', c_tasmax, dgpi)                           #FreezingDays
    _d0('CoolingDegDay', c_tasmax, dgpi, fK_=dict(thr=20))       #CoolingDegDay
    _d1('ConWarmDays', c_tasmax, dgpi, y0y1)                       #ConWarmDays
    _d0('FrostDays', c_tasmin, dgpi)                                 #FrostDays
    _d0('TropicNights', c_tasmin, dgpi)                            #TropicNight
    _d1('SpringFrostDayEnd', c_tasmin, dgpi, y0y1, fA_='yd') #SpringFrostDayEnd
    _d1('FirstDayWithoutFrost', c_tasmin, dgpi, y0y1, fA_='yd')
                                                          #FirstDayWithoutFrost
    _d0('ZeroCrossingDays', (c_tasmax, c_tasmin), dgpi)       #ZeroCrossingDays
    _d0('MinusDays', c_tas, dgpi)                                    #MinusDays
    _d0('DegDay20', c_tas, dgpi, fK_=dict(thr=20))                    #DegDay20
    _d0('DegDay17', c_tas, dgpi, fK_=dict(thr=17, left=True))         #DegDay17
    _d1('DegDay8', c_tas, dgpi, y0y1)                                  #DegDay8
    if any([i in il_ for i in ['VegSeasonDayStart-5', 'VegSeasonDayEnd-5',
                               'VegSeasonLength-5']]):
        o = _d1(['VegSeasonDayStart-5', 'VegSeasonDayEnd-5'],
                c_tas, dgpi, y0y1, #ax_t, y_y_,
                fA_='yd',
                fK_=dict(thr=5),
                out=True)
        v_ = 'VegSeasonLength-5'
        if v_ in il_:
            t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
            o = o[1] - o[0]
            _sv(v_, o, dgpi)
            ll_(v_, t000)                                          #VegSeason-5
    if any([i in il_ for i in ['VegSeasonDayStart-2', 'VegSeasonDayEnd-2',
                               'VegSeasonLength-2']]):
        o = _d1(['VegSeasonDayStart-2', 'VegSeasonDayEnd-2'],
                c_tas, dgpi, y0y1, #ax_t, y_y_,
                fA_='yd',
                fK_=dict(thr=2),
                out=True)
        v_ = 'VegSeasonLength-2'
        if v_ in il_:
            t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
            o = o[1] - o[0]
            _sv(v_, o, dgpi)
            ll_(v_, t000)                                          #VegSeason-2
    _d0('HumiWarmDays', (c_hurs, c_tas), dgpi)                    #HumiWarmDays
    v_ = 'RhoS'
    if v_ in il_ and all([i is not None for i in (c_tas, c_huss, c_ps)]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_tas, c_huss, c_ps), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                     #RhoS
    v_ = 'Rho975'
    if v_ in il_ and all([i is not None for i in [c_ta975, c_hus975]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta975, c_hus975, c_ps) if c_ps is not None else \
             (c_ta975, c_hus975)
        _f_n(_rho_ps_p, cL, 97500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho975
    v_ = 'Rho950'
    if v_ in il_ and all([i is not None for i in [c_ta950, c_hus950]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta950, c_hus950, c_ps) if c_ps is not None else \
             (c_ta950, c_hus950)
        _f_n(_rho_ps_p, cL, 95000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho950
    v_ = 'Rho925'
    if v_ in il_ and all([i is not None for i in [c_ta925, c_hus925]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta925, c_hus925, c_ps) if c_ps is not None else \
             (c_ta925, c_hus925)
        _f_n(_rho_ps_p, cL, 92500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho925
    v_ = 'Rho900'
    if v_ in il_ and all([i is not None for i in [c_ta900, c_hus900]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta900, c_hus900, c_ps) if c_ps is not None else \
             (c_ta900, c_hus900)
        _f_n(_rho_ps_p, cL, 90000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho900
    v_ = 'Rho50m'
    if v_ in il_ and all([i is not None for i in [c_ta50m, c_hus50m, c_p50m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_ta50m, c_hus50m, c_p50m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho50m
    v_ = 'Rho100m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta100m, c_hus100m, c_p100m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_ta100m, c_hus100m, c_p100m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                  #Rho100m
    v_ = 'Rho200m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta200m, c_hus200m, c_p200m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_ta200m, c_hus200m, c_p200m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                  #Rho200m
    _d0('DryDays', c_pr, dgpi)                                         #DryDays
    _dd('PRmax', c_pr, dgpi)                                             #PRmax
    _d0('PRgt10Days', c_pr, dgpi)                                   #PRgt10Days
    _d0('PRgt25Days', c_pr, dgpi, fK_=dict(thr=25))                 #PRgt25Days
    _d1('PR7Dmax', c_pr, dgpi, y0y1)                                   #PR7Dmax
    _d1('LnstDryDays', c_pr, dgpi, y0y1)                           #LnstDryDays
    _d1('SuperCooledPR', (c_pr, c_ps, c_tas, c_ta925, c_ta850, c_ta700,
                          c_hus925, c_hus850, c_hus700), dgpi, y0y1)
                                                                 #SuperCooledPR
    _d0('SncDays', c_snc, dgpi)                                        #SncDays
    _d0('Snc25Days', c_snc, dgpi, fK_=dict(thr=25))                  #Snc25Days
    _d0('Snd10Days', c_snd, dgpi)                                    #Snd10Days
    _d0('Snd20Days', c_snd, dgpi)                                    #Snd20Days
    _dd('SNWmax', c_snw, dgpi)                                          #SNWmax
    _dd('PRSNmax', c_prsn, dgpi)                                       #PRSNmax
    _d0('ColdRainWarmSnowDays', (c_pr, c_tas), dgpi)      #ColdRainWarmSnowDays
    _d0('ColdRainDays', (c_pr, c_tas), dgpi)                      #ColdRainDays
    _d0('ColdRainGT10Days', (c_pr, c_tas), dgpi, fK_=dict(thr_pr=10))
                                                              #ColdRainGT10Days
    _d0('ColdRainGT20Days', (c_pr, c_tas), dgpi, fK_=dict(thr_pr=20))
                                                              #ColdRainGT20Days
    _d0('WarmSnowDays', (c_pr, c_tas), dgpi)                      #WarmSnowDays
    _d0('WarmSnowGT10Days', (c_pr, c_tas), dgpi, fK_=dict(thr_pr=10))
                                                              #WarmSnowGT10Days
    _d0('WarmSnowGT20Days', (c_pr, c_tas), dgpi, fK_=dict(thr_pr=20))
                                                              #WarmSnowGT20Days
    _d0('WarmPRSNdays', (c_prsn, c_tas), dgpi)                    #WarmPRSNdays
    _d0('WarmPRSNgt10Days', (c_prsn, c_tas), dgpi, fK_=dict(thr_pr=10))
                                                              #WarmPRSNgt10Days
    _d0('WarmPRSNgt20Days', (c_prsn, c_tas), dgpi, fK_=dict(thr_pr=20))
                                                              #WarmPRSNgt20Days
    _d0('ColdPRRNdays', (c_pr, c_prsn, c_tas), dgpi)              #ColdPRRNdays
    _d0('ColdPRRNgt10Days', (c_pr, c_prsn, c_tas), dgpi, fK_=dict(thr_pr=10))
                                                              #ColdPRRNgt10Days
    _d0('ColdPRRNgt20Days', (c_pr, c_prsn, c_tas), dgpi, fK_=dict(thr_pr=20))
                                                              #ColdPRRNgt20Days
    if (any([i in il_ for i in ['R5OScw', 'R1OScw', 'R5OSc', 'R1OSc']]) and
        c_pr and c_tas):
        o = dPRRN_fr_PR_T_(c_pr, c_tas)
    else:
        o = None
    if o and c_snc and c_snw:
        attr = None if 'PRRN' not in o.attributes else o.attributes['PRRN']
        fK_ = dict(cSnw=c_snw, attr=attr)
        _d0('R5OScw', (o, c_snc), dgpi, fK_=fK_)                        #R5OScw
        _d0('R1OScw', (o, c_snc), dgpi, fK_=dict(thr_r=1., **fK_))      #R1OScw
    if o and c_snc:
        attr = None if 'PRRN' not in o.attributes else o.attributes['PRRN']
        fK_ = dict(attr=attr)
        _d0('R5OSc', (o, c_snc), dgpi, fK_=fK_)                          #R5OSc
        _d0('R1OSc', (o, c_snc), dgpi, fK_=dict(thr_r=1., **fK_))        #R1OSc


def _h6climidx(**kwArgs):
    _hclimidx(tint='h6', **kwArgs)


def _h3climidx(**kwArgs):
    _hclimidx(tint='h3', **kwArgs)


def _h1climidx(**kwArgs):
    _hclimidx(tint='h1', **kwArgs)


def _hclimidx(tint='h6',
              dn=None, gwl=None, po_=None, il_=None, user_cfg=None, y0y1=None,
              c_huss=None, c_hus975=None, c_hus950=None, c_hus925=None,
              c_hus900=None, c_hus850=None, c_hus700=None, c_hus50m=None,
              c_hus100m=None, c_hus200m=None,
              c_pr=None,
              c_ps=None, c_p50m=None, c_p100m=None, c_p200m=None,
              c_tas=None, c_ta975=None, c_ta950=None, c_ta925=None,
              c_ta900=None, c_ta850=None, c_ta700=None, c_ta50m=None,
              c_ta100m=None, c_ta200m=None,
              c_ua975=None, c_ua950=None, c_ua925=None, c_ua900=None,
              c_ua50m=None, c_ua100m=None, c_ua200m=None, c_uas=None,
              c_va975=None, c_va950=None, c_va925=None, c_va900=None,
              c_va50m=None, c_va100m=None, c_va200m=None, c_vas=None,
              c_sfcWind=None):

    dgpi = [dn, gwl, po_, il_, user_cfg]
    if any([i is None for i in dgpi[:-1]]):
        raise ValueError("inputs of 'dn', 'gwl', 'po_', 'il_' are mandotory!")
    os.makedirs(po_, exist_ok=True)

    v_ = tint+'SfcWind'
    if c_sfcWind is None and c_uas and c_vas and v_ in il_:
        c_sfcWind = ws_cube(c_uas, c_vas)
        pst_(c_sfcWind, 'surface wind speed', var_name='sfcWind')
    _dd(v_, c_sfcWind, dgpi)                                         #hxSfcWind
    v_ = tint+'Wind975'
    o = None
    if v_ in il_ and c_ua975 and c_va975:
        o = ws_cube(c_ua975, c_va975)
        cL = (o, c_ps) if c_ps else (o,)
        _f_n(_wind_msk, cL, 97500., v_, dgpi)
        _to1(v_, dgpi)                                               #hxWind975
    v_ = tint+'Wind950'
    o = None
    if v_ in il_ and c_ua950 and c_va950:
        o = ws_cube(c_ua950, c_va950)
        cL = (o, c_ps) if c_ps else (o,)
        _f_n(_wind_msk, cL, 95000., v_, dgpi)
        _to1(v_, dgpi)                                               #hxWind950
    v_ = tint+'Wind925'
    o = None
    if v_ in il_ and c_ua925 and c_va925:
        o = ws_cube(c_ua925, c_va925)
        cL = (o, c_ps) if c_ps else (o,)
        _f_n(_wind_msk, cL, 92500., v_, dgpi)
        _to1(v_, dgpi)                                               #hxWind925
    v_ = tint+'Wind900'
    o = None
    if v_ in il_ and c_ua900 and c_va900:
        o = ws_cube(c_ua900, c_va900)
        cL = (o, c_ps) if c_ps else (o,)
        _f_n(_wind_msk, cL, 90000., v_, dgpi)
        _to1(v_, dgpi)                                               #hxWind900
    v_ = tint+'Wind50m'
    o = None
    if v_ in il_ and c_ua50m and c_va50m:
        o = ws_cube(c_ua50m, c_va50m)
    _dd(v_, o, dgpi)                                                 #hxWind50m
    v_ = tint+'Wind100m'
    o = None
    if v_ in il_ and c_ua100m and c_va100m:
        o = ws_cube(c_ua100m, c_va100m)
    _dd(v_, o, dgpi)                                                #hxWind100m
    v_ = tint+'Wind200m'
    o = None
    if v_ in il_ and c_ua200m and c_va200m:
        o = ws_cube(c_ua200m, c_va200m)
    _dd(v_, o, dgpi)                                                #hxWind200m
    v_ = tint+'RhoS'
    if v_ in il_ and all([i is not None for i in (c_tas, c_huss, c_ps)]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_tas, c_huss, c_ps), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #hxRhoS
    v_ = tint+'Rho975'
    if v_ in il_ and all([i is not None for i in [c_ta975, c_hus975]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta975, c_hus975, c_ps) if c_ps is not None else \
             (c_ta975, c_hus975)
        _f_n(_rho_ps_p, cL, 97500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho975
    v_ = tint+'Rho950'
    if v_ in il_ and all([i is not None for i in [c_ta950, c_hus950]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta950, c_hus950, c_ps) if c_ps is not None else \
             (c_ta950, c_hus950)
        _f_n(_rho_ps_p, cL, 95000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho950
    v_ = tint+'Rho925'
    if v_ in il_ and all([i is not None for i in [c_ta925, c_hus925]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta925, c_hus925, c_ps) if c_ps is not None else \
             (c_ta925, c_hus925)
        _f_n(_rho_ps_p, cL, 92500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho925
    v_ = tint+'Rho900'
    if v_ in il_ and all([i is not None for i in [c_ta900, c_hus900]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        cL = (c_ta900, c_hus900, c_ps) if c_ps is not None else \
             (c_ta900, c_hus900)
        _f_n(_rho_ps_p, cL, 90000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho900
    v_ = tint+'Rho50m'
    if v_ in il_ and all([i is not None for i in [c_ta50m, c_hus50m, c_p50m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_ta50m, c_hus50m, c_p50m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho50m
    v_ = tint+'Rho100m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta100m, c_hus100m, c_p100m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_ta100m, c_hus100m, c_p100m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                #hxRho100m
    v_ = tint+'Rho200m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta200m, c_hus200m, c_p200m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        _f_n(_rho_ps, (c_ta200m, c_hus200m, c_p200m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                #hxRho200m
    v_ = tint+'SuperCooledPR'
    _d1(v_, (c_pr, c_ps, c_tas, c_ta925, c_ta850, c_ta700,
             c_hus925, c_hus850, c_hus700), dgpi,
        y0y1)                                                  #hxSuperCooledPR


def _wind_msk(cL, p, v_, dgpi, _nm=None):
    if len(cL) > 1:
        iris.util.mask_cube(cL[0], cL[1].data < p) 
    _dd(v_, cL[0], dgpi, _nm=_nm)


def _rho_ps_p(cL, p, v_, dgpi, _nm=None):
    t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
    o = cL[0].copy(rho_fr_t_q_p_(cL[0].data, cL[1].data, p))
    if len(cL) > 2:
        o = iris.util.mask_cube(o, cL[2].data < p)
    pst_(o, 'air density', 'kg m-3', 'rho')
    ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
    _dd(v_, o, dgpi, _nm=_nm)


def _rho_ps(cL, v_, dgpi, _nm=None):
    t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
    o = cL[0].copy(rho_fr_t_q_p_(cL[0].data, cL[1].data, cL[2].data))
    pst_(o, 'air density', 'kg m-3', 'rho')
    ll_('{} {} ... predata'.format(i__[v_][0], v_), t000)
    _dd(v_, o, dgpi, _nm=_nm)


def _szG(cL):
    if not isMyIter_(cL):
        return np.prod(cL.shape) * 8 * 1.e-9
    else:
        return np.prod(cL[0].shape) * len(cL) * 8 * 1.e-9


def _f_n(func, cL, *args, xm=80, **kwargs):
    if _szG(cL) < xm:
        func(cL, *args, **kwargs)
    else:
        n = int(np.ceil(_szG(cL) / xm))
        cLL = [nTslice_cube(i, n) for i in cL]
        nn = len(cLL[0])
        ll_('n_slice = {}'.format(nn))
        t000 = l__('loop nTslice')
        for i in range(nn):
            ###H
            #if i < 5:
            #    continue
            ###I
            func([ii[i].copy() for ii in cLL], *args,
                 _nm='__{}'.format(i), **kwargs)
            ll_(prg_(i, nn), t000)


def _afm_n(cL, ax, func, out, *args, npr=32, xm=80, **kwargs):
    if _szG(cL) < xm:
        ax_fn_mp_([i.data for i in cL] if isMyIter_(cL) else cL.data,
                  ax, func, out, *args, npr=npr, **kwargs)
    else:
        n = int(np.ceil(_szG(cL) / xm))
        cLL = [nTslice_cube(i, n) for i in cL] if isMyIter_(cL) else \
              nTslice_cube(cL, n)
        outL = [nTslice_cube(i, n) for i in out] if isMyIter_(out) else \
               nTslice_cube(out, n)
        nn = len(cLL[0]) if isMyIter_(cL) else len(cLL)
        t000 = l__('loop nTslice')
        for i in range(nn):
            ax_fn_mp_([ii[i].copy().data for ii in cLL] if isMyIter_(cL) else
                      cLL[i].data,
                      ax, func,
                      [ii[i] for ii in outL] if isMyIter_(out) else outL[i],
                      *args, npr=npr, **kwargs)
            ll_(prg_(i, nn), t000)
        out_ = [concat_cube_(iris.cube.CubeList(i)) for i in outL]\
               if isMyIter_(out) else concat_cube_(iris.cube.CubeList(outL))
        return out_


def _xx(pi_, freq):
    if isinstance(pi_, str):
        return _djn(pi_, '*' + freq)
    elif isinstance(pi_, (tuple, list, set, np.ndarray)):
        return [_xx(i, freq) for i in pi_]
    else:
        raise ValueError("'pi_' must be str type or array-like of str")


def _to_xhr(cube, x=6, valid=True):
    nh = 24 / x
    ica.add_categorised_coord(cube, 'xxx', 'time',
                              lambda coord, v: np.ceil(v * nh) / nh)
    o = cube.aggregated_by('xxx',iris.analysis.MEAN)
    if valid:
        dpo = np.diff(o.coord('xxx').points)
        ddpo = np.diff(np.where(dpo != 0)[0])[1]
        sss = None if dpo[ddpo-1] != 0 else 1
        eee = None if dpo[-ddpo] != 0 else -1
    else:
        sss, eee = None, None
    cube.remove_coord('xxx')
    return extract_byAxes_(o, 'time', np.s_[sss:eee])


def _xxx(cube, freq0, freq1):
    freqs = ['mon', 'day', '6hr', '3hr', '1hr']
    if any([i not in freqs for i in [freq0, freq1]]):
        raise ValueError("unknown frequency names!")
    if freqs.index(freq1) > freqs.index(freq0):
        raise ValueError("cannot convert from low to high frequency!")
    if freq1 == 'mon':
        return pSTAT_cube(cube, 'MEAN', 'month')
    elif freq1 == 'day':
        return pSTAT_cube(cube, 'MEAN', 'day')
    elif freq1 == '6hr':
        if cube.cell_methods and cube.cell_methods[0].method.upper() == 'MEAN':
            return _to_xhr(cube)
        else:
            nn = 2 if freq0 == '3hr' else 6
            return extract_byAxes_(cube, 'time', np.s_[::nn])
    else:
        if cube.cell_methods and cube.cell_methods[0].method.upper() == 'MEAN':
            return _to_xhr(cube, x=3)
        else:
            return extract_byAxes_(cube, 'time', np.s_[::3])


def rf__(pi_, freq, folder='cordex', reg_d=None, **kwargs):
    freqs = ['mon', 'day', '6hr', '3hr', '1hr']
    _rf = eval('{}_dir_cubeL'.format(folder))
    o = None
    e = None
    for f_ in freqs[freqs.index(freq):]:
        p_ = _xx(pi_, f_)
        o = _rf(p_, ifconcat=True, **kwargs)
        if o:
            o = o['cube']
            if o:
                if reg_d:
                    o = intersection_(o, **reg_d)
                if 'period' in kwargs:
                    o = extract_period_cube(o, *kwargs['period'], yy=True)
                    e = None if o else 'yye'
                break
            else:
                e = 'cce'
    if f_ != freq and isinstance(o, iris.cube.Cube):
        o = _xxx(o, f_, freq)
    return (o, e)


def _gg(folder='cordex'):
    if folder == 'cmip5':
        yf = _djn(_here_, 'gcm_gwls_.yml')
    elif folder == 'cordex':
        yf = _djn(_here_, 'gcm_gwls.yml')
    else:
        raise Exception("unknown folder: {!r}".format(folder))
    with open(yf, 'r') as ymlfile:
        gg = yaml.safe_load(ymlfile)
    return gg


def _pp(yf0):
    with open(yf0, 'r') as ymlfile:
        pp = yaml.safe_load(ymlfile)
    return pp


def _yy_dn(pD, dn, gwl, gg, curr):
    if gwl[:3] == 'gwl':
        try:
            y0 = gg[gwl][pD['rcp']][pD['gcm']][pD['rip']]
            y0y1 = [y0, y0 + 29]
        except KeyError:
            y0y1 = None
    elif gwl == 'current' and pD['rcp'] == 'rcp85':
        y0y1 = curr
        dn = dn.replace('rcp85', 'historical')
    elif '-' in gwl:
        def _int(s):
            tmp = int(s)
            tmp_ = 0 if tmp > 100 else (1900 if tmp > 50 else 2000)
            return tmp + tmp_
        y0y1 = list(map(_int, gwl.split('-')))
    else:
        y0y1 = curr
    return (y0y1, dn)


def cmip6_rcp_(il_, reg_d, reg_n, po_,
               sss=None, eee=None, idx=None, yml=None, tii=None,
               subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'cmip6_smhi_len.yml'))
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip6_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n)
        t0 = l__('>>>>>>>' + dn)
        pi_ = _djn(pp['root'], p_)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='cmip6',
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def cmip5_imp_rcp_(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'cmip5_import_.yml'))
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip5_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n)
        t0 = l__('>>>>>>>' + dn)
        pi_ = _djn(pp['root'], p_)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='cmip5',
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def cmip5_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None, idx=None, yml=None, tii=None,
                  subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'cmip5_import.yml'))
    #pp = _pp(_djn(_here_, 'cmip5_import_cp.yml'))
    gg = _gg('cmip5')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip5_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n)
        y0y1, dn = _yy_dn(tmp, dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        t0 = l__('>>>>>>>' + dn)
        pi1 = pp['root'] + p_
        pi0 = pi1.replace(tmp['rcp'], 'historical')
        pi_ = pi0 if y0y1[1] <= 2005 else (pi1 if y0y1[0] > 2005 else
                                           [pi0, pi1])
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, folder='cmip5',
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def norcp_rcp_(il_, reg_d, reg_n, po_,
               sss=None, eee=None, idx=None, yml=None, tii=None,
               subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'norcp_.yml'))
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2norcp_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                 tmp['ver'], tmp['prd'], reg_n)
        t0 = l__('>>>>>>>' + dn)
        pi_ = _djn(pp['root'], p_)
        tis = ['mon', 'day', '3hr'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='norcp',
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp_(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'eur-11_import__.yml'))
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                 tmp['ver'], reg_n)
        t0 = l__('>>>>>>>' + dn)
        pi_ = _djn(pp['root'], p_)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d,
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None, idx=None, yml=None, tii=None,
                  subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'eur-11_import.yml'))
    gg = _gg()
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                 tmp['ver'], reg_n)
        y0y1, dn = _yy_dn(tmp, dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        t0 = l__('>>>>>>>' + dn)
        pi1 = pp['root'] + p_
        pi0 = pi1.replace(tmp['rcp'], 'historical')
        pi_ = pi0 if y0y1[1] <= 2005 else (pi1 if y0y1[0] > 2005 else
                                           [pi0, pi1])
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d,
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9, user_cfg=None):
    gwl = ''
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'eur-11_import_eval.yml'))
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                 tmp['ver'], reg_n)
        t0 = l__('>>>>>>>' + dn)
        pi_ = _djn(pp['root'], p_)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, None, po_, reg_d,
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval_dmi(il_, reg_d, reg_n, po_,
                       tii=None, subg=None, xvn=9, user_cfg=None):
    gwl = ''
    pp = _pp(_djn(_here_, 'eur-11_import_eval.yml'))
    y0y1 = [1989, 2010]
    for p_ in pp['p__']:
        tmp = path2cordex_info_(p_)
        dn = _dn(tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                 tmp['ver'], reg_n)
        t0 = l__('>>>>>>>' + dn)
        pi_ = _djn(pp['root'], p_)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d,
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_eval(il_, reg_d, reg_n, po_, yml=None, tii=None,
                    subg=None, xvn=9, user_cfg=None):
    gwl = ''
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'eur-11_smhi-rca4.yml'))
    p_ = pp['root'] + str(pp['eval']) + '/netcdf/'
    gcm = pp[pp['eval']]['gcm']
    rcp = pp[pp['eval']]['rcp']
    rip = pp[pp['eval']]['rip']
    rcm = pp[pp['eval']]['rcm']
    ver = pp[pp['eval']]['ver']
    dn = _dn(gcm, rcp, rip, rcm, ver, reg_n)
    t0 = l__('>>>>>>>' + dn)
    tis = ['mon', 'day'] if tii is None else tii.split(',')
    for tint in tis:
        _xyz(il_, tint, p_, dn, gwl, None, po_, reg_d,
             subg=subg, xvn=xvn, user_cfg=user_cfg)
    ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp_(il_, reg_d, reg_n, po_,
                    sss=None, eee=None, idx=None, yml=None, tii=None,
                    subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'eur-11_smhi-rca4.yml'))
    pp_ = l_ind_(pp['h248'], [int(i) for i in idx.split(',')]) if idx else\
          pp['h248'][sss:eee]
    for ppi in pp_:
        pi_ = _djn(pp['root'], ppi, 'netcdf')
        dn = _dn(pp[ppi]['gcm'], pp[ppi]['rcp'], pp[ppi]['rip'],
                 pp[ppi]['rcm'], pp[ppi]['ver'], reg_n)
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d,
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9, user_cfg=None):
    pp = _pp(yml) if yml else _pp(_djn(_here_, 'eur-11_smhi-rca4.yml'))
    gg = _gg()
    pp_ = l_ind_(pp['rcps'], [int(i) for i in idx.split(',')]) if idx else\
          pp['rcps'][sss:eee]
    for p0p1 in pp_:
        pi0, pi1 = p0p1[0], p0p1[1]
        pi0_, pi1_ = (_djn(pp['root'], pi0, 'netcdf'),
                      _djn(pp['root'], pi1, 'netcdf'))
        dn = _dn(pp[pi1]['gcm'], pp[pi1]['rcp'], pp[pi1]['rip'],
                 pp[pi1]['rcm'], pp[pi1]['ver'], reg_n)
        y0y1, dn = _yy_dn(pp[pi1], dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        pi_ = pi0_ if y0y1[1] <= 2005 else (pi1_ if y0y1[0] > 2005 else
                                            [pi0_, pi1_])
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d,
                 subg=subg, xvn=xvn, user_cfg=user_cfg)
        ll_('<<<<<<<' + dn, t0)


def eobs20_(il_, reg_d, reg_n, po_, y0y1=None, user_cfg=None):
    idir = '/nobackup/rossby22/sm_chali/DATA/hw2018/iii/obs/EOBS20/'
    vo = {'c_pr': ('rr', 1. / 3600 / 24, 'kg m-2 s-1'),
          'c_tas': ('tg', None, 'K'),
          'c_tasmax': ('tx', None, 'K'),
          'c_tasmin': ('tn', None, 'K'),
          'c_rsds': ('qq', 1, 'K'),
          }

    def _eobs_load(var):
        o = iris.load_cube(_djn(idir,
            '{}_ens_mean_0.1deg_reg_v20.0e.nc'.format(vo[var][0]))
        if reg_d is not None:
            o = intersection_(o, **reg_d)
        if vo[var][1] is None and vo[var][2]:
            o.convert_units(vo[var][2])
        elif vo[var][1] == 1 and vo[var][2]:
            o.units = vo[var][2]
        elif vo[var][2]:
            o *= vo[var][1]
            o.units = vo[var][2]
        return o if y0y1 is None else extract_period_cube(o, *y0y1)

    def _getccc(tint):
        tmp = _vd(il_, tint, subg=None, xvn=9)
        if tmp:
            vD = tmp[0]
            ccc = dict()
            for i in vD[1].keys():
                if i in vo.keys():
                    o = _eobs_load(i)
                    ccc.update({i: o})
            return ccc

    t0 = l__('>>>monthly')
    t1 = l__(' >>loading data')
    cccc = _getccc('month')
    ll_(' <<loading data', t1)
    if cccc:
        _mclimidx(il_=il_, dn=_dn('EOBS20', reg_n), gwl='', po_=po_,
                  user_cfg=user_cfg, **cccc)
    ll_('<<<monthly', t0)
    t0 = l__('>>>daily')
    t1 = l__(' >>loading data')
    cccc = _getccc('day')
    ll_(' <<loading data', t1)
    if cccc:
        _dclimidx(il_=il_, dn=_dn('EOBS20', reg_n), gwl='', po_=po_, y0y1=y0y1,
                  user_cfg=user_cfg, **cccc)
    ll_('<<<daily', t0)


def erai_(il_, reg_d, reg_n, po_, y0y1=None, tii=None, subg=None, xvn=9,
          user_cfg=None):
    idir = '/home/rossby/imports/obs/ECMWF/ERAINT/input/'
    tis = ['mon', 'day'] if tii is None else tii.split(',')
    for tint in tis:
        _xyz(il_, tint, idir, _dn('ERAI', reg_n), '', y0y1, po_, reg_d,
             folder='cmip5', subg=subg, xvn=xvn, user_cfg=user_cfg)


def main():
    parser = argparse.ArgumentParser('RUN CLIMIDX')
    parser.add_argument("opt",
                        type=str,
                        help="options for dataset on BI: "
                             "0->eobs | "
                             "1->erai | "
                             "2->cdx_eval; 21->cdx_eval_dmi; "
                             "22->cdx_eval_smhi | "
                             "3->cdx; 30->cdx_smhi; 31->cdx_gwl; "
                             "32->cdx_gwl_smhi; 33->cdx_0130; "
                             "34->cdx_0130_smhi | "
                             "4->norcp | "
                             "5->cmp5; 51->cmp5_gwl; 54->cmp5_xxx | "
                             "6->cmp6")
    parser.add_argument("-x", "--indices",
                        type=str,
                        help="indices to be calculated. Formats: "
                             "file name (yaml): read list from yaml file | "
                             "indexA,indexB,indexC (no space after comma) | "
                             "grp_a[bc]: indices belong to a [and b and c] "
                             "currently available group labels: "
                             "w | t | p | r | c")
    parser.add_argument("-X", "--indices_excl",
                        type=str,
                        help="indices to not be calculated. Format: "
                             "indexA,indexB,indexC (no space after comma)")
    parser.add_argument("-w", "--gwl",
                        type=str,
                        help="warming levels: "
                             "current | gwl15 | gwl2 | gwl25 | gwl3 | gwl35 | "
                             "gwl4 | xx-xx | xxxx-xxxx")
    parser.add_argument("-s", "--start",
                        type=int,
                        help="simulation-loop start.")
    parser.add_argument("-e", "--end",
                        type=int,
                        help="simulation-loop end")
    parser.add_argument("-i", "--idx",
                        type=str,
                        help="simulation-loop index. exp: 0,1,3 "
                             "meaning simulation #1,2,4 in the lists "
                             "(specified in a yaml file) to be calculated")
    parser.add_argument("-c", "--cfg",
                        type=str,
                        help="yaml file that stores user configuration")
    parser.add_argument("-y", "--yml",
                        type=str,
                        help="yaml file that stores paths (simulations)")
    parser.add_argument("-t", "--tint",
                        type=str,
                        help="temporal resolution(s) of input data: mon,day")
    parser.add_argument("--lll",
                        type=str,
                        help="longitude/latitude limits: lo0,lo1,la0,la1")
    parser.add_argument("-d", "--domain",
                        type=str,
                        help="name of domain")
    parser.add_argument("--rdir",
                        type=str,
                        help="directory where the results to be stored!")
    parser.add_argument("-g", "--subg",
                        type=str,
                        help="method for grouping indices for a single call: "
                              "None (default) | 'v' | 'i'")
    parser.add_argument("-n", "--xvn",
                        type=int,
                        default=9,
                        help="maximum number of variables: "
                             "maximum input variables for a single call")
    parser.add_argument("-l", "--log",
                        type=str,
                        help="exclusive log identifier")
    args = parser.parse_args()

    xi_ = args.indices
    if xi_:
        if os.path.isfile(xi_):
            with open(xi_, 'r') as yf:
                il_ = yaml.safe_load(yf)
        elif xi_[:4] == 'grp_':
            il_ = [i for i in i__.keys()
                   if all(ii in i__[i][6] for ii in xi_[4:])]
        else:
            il_ = xi_.split(',')
    else:
        il_ = list(i__.keys())
    if args.indices_excl:
        el_ = args.indices_excl.split(',')
        for i in el_:
            il_.remove(i)
    user_cfg=None
    if args.cfg:
        if os.path.isfile(args.cfg):
            with open(args.cfg, 'r') as yf:
                user_cfg = yaml.safe_load(yf)
        elif os.path.isfile(_djn(_here_, args.cfg)):
            with open(_djn(_here_, args.cfg), 'r') as yf:
                user_cfg = yaml.safe_load(yf)
    if args.lll:
        lo0, lo1, la0, la1 = [float(i) for i in args.lll.split(',')]
        reg_d = {'longitude': [lo0, lo1], 'latitude': [la0, la1]}
    else:
        reg_d = None
    #reg_d = {'longitude': [10.0, 23.0], 'latitude': [55.0, 69.0]}
    #reg_n = 'SWE'
    #reg_d = {'longitude': [-25.0, 45.0], 'latitude': [25.0, 75.0]}
    reg_n = args.domain if args.domain else \
            ('LLL' if reg_d is not None else '')
    if args.rdir:
        rdir = args.rdir
    else:
        rxx = os.environ.get('r26')
        rdir = '{}DATA/energi/res/'.format(rxx)
    pf_ = lambda x: os.path.join(*(i for i in (rdir, x, reg_n) if i))
    poe_ = pf_('eval')
    poo_ = pf_('obs')
    pcdx = pf_('h248/cordex/EUR11')
    pcmp5 = pf_('h248/cmip5')
    pcmp6 = pf_('h248/cmip6')
    pnorcp = pf_('h248/norcp')
    pcdx_ = pf_('gwls/cordex/EUR11')
    pcmp5_ = pf_('gwls/cmip5')
    pcmp6_ = pf_('gwls/cmip6')

    warnings.filterwarnings("ignore", category=UserWarning)
    logn = [args.opt]
    if args.gwl:
        logn.append(args.gwl)
    if args.log:
        logn.append(args.log)
    logn = '-'.join(logn)
    nlog = len(find_patt_(r'^{}_*'.format(logn),
                          pure_fn_(schF_keys_('', logn, ext='.log'))))
    logging.basicConfig(filename=logn + '_'*nlog + '.log',
                        filemode='w',
                        level=logging.INFO)
    logging.info(' {:_^42}'.format('start of program'))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))
    logging.info(' ')

    seiyt = dict(sss=args.start, eee=args.end, idx=args.idx,
                 yml=args.yml, tii=args.tint, subg=args.subg, xvn=args.xvn,
                 user_cfg=user_cfg)

    if args.opt in ('0', 'eobs20'):
        eobs20_(il_, reg_d, reg_n, poo_, user_cfg=user_cfg)
    elif args.opt in ('1', 'erai'):
        erai_(il_, reg_d, reg_n, poo_,
              tii=args.tint, subg=args.subg, xvn=args.xvn, user_cfg=user_cfg)
    elif args.opt in ('2', 'cdx_eval'):
        eur11_imp_eval(il_, reg_d, reg_n, poe_, **seiyt)
    elif args.opt in ('21', 'cdx_eval_dmi'):
        eur11_imp_eval_dmi(il_, reg_d, reg_n, poe_,
                           tii=args.tint, subg=args.subg, xvn=args.xvn,
                           user_cfg=user_cfg)
    elif args.opt in ('22', 'cdx_eval_smhi'):
        eur11_smhi_eval(il_, reg_d, reg_n, poe_, yml=args.yml,
                        tii=args.tint, subg=args.subg, xvn=args.xvn,
                        user_cfg=user_cfg)
    elif args.opt in ('3', 'cdx'):
        eur11_imp_rcp_(il_, reg_d, reg_n, pcdx, **seiyt)
    elif args.opt in ('30', 'cdx_smhi'):
        eur11_smhi_rcp_(il_, reg_d, reg_n, pcdx, **seiyt)
    elif args.opt in ('31', 'cdx_gwl'):
        eur11_imp_rcp(il_, reg_d, reg_n, pcdx_, gwl=args.gwl, **seiyt)
    elif args.opt in ('32', 'cdx_gwl_smhi'):
        eur11_smhi_rcp(il_, reg_d, reg_n, pcdx_, gwl=args.gwl, **seiyt)
    elif args.opt in ('33', 'cdx_0130'):
        eur11_imp_rcp(il_, reg_d, reg_n, pcdx_, curr=[2001, 2030],
                      gwl='curr0130', **seiyt)
    elif args.opt in ('34', 'cdx_0130_smhi'):
        eur11_smhi_rcp(il_, reg_d, reg_n, pcdx_, curr=[2001, 2030],
                       gwl='curr0130', **seiyt)
    elif args.opt in ('4', 'norcp'):
        norcp_rcp_(il_, reg_d, reg_n, pnorcp, **seiyt)
    elif args.opt in ('5', 'cmp5'):
        cmip5_imp_rcp_(il_, reg_d, reg_n, pcmp5, **seiyt)
    elif args.opt in ('51', 'cmp5_gwl'):
        cmip5_imp_rcp(il_, reg_d, reg_n, pcmp5_, gwl=args.gwl, **seiyt)
    elif args.opt in ('54', 'cmp5_xxx'):
        cmip5_imp_rcp(il_, reg_d, reg_n, pcmp5_, curr=[1981, 2100],
                      gwl='xxx', **seiyt)
    elif args.opt in ('6', 'cmp6'):
        cmip6_rcp_(il_, reg_d, reg_n, pcmp6, **seiyt)
    else:
        pass


if __name__ == '__main__':
    start_time = time.time()
    main()
    logging.info(' ')
    logging.info(' {:_^42}'.format('end of program'))
    logging.info(' {:_^42}'.format('TOTAL'))
    logging.info(' ' + rTime_(time.time() - start_time))
    logging.info(strftime(" %a, %d %b %Y %H:%M:%S +0000", localtime()))

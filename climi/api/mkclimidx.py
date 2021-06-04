#!/usr/bin/env python3

from climi.uuuu import *
from climi.climidx import *

import numpy as np
import iris
import iris.coord_categorisation as cat
import os
import yaml
import time
import warnings
import logging
import argparse

from time import localtime, strftime


_here_ = get_path_(__file__)


# INDEX DICT: FORMAT:
#     NAME: (#, TR_i, VARIABLES, FUNCTION, TR_o, METADATA, GROUPS)
# where TR_i, TR_o are input and output temporal resolution.
# acceptable values and desciption:
#     TR_i: ('mon', 'day', '6hr', '3hr', '1hr')
#     VARIABLES: input variables (using the same names as CORDEX/CMIPX)
#     FUNCTION (most fr. climidx mode):
#         None or string: using pSTAT_cube() fr. mode uuuu; see _mm() or _dd()
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
        't'), # SVT_ERIK
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
            #vd.update({'c_' + i: var_[i]})
            vd.update({'c_' + i: i})
        return (vl[0], vd)
    elif isinstance(vl, list):
        return [_vd_fr_vv(i) for i in vl]
    else:
        raise Exception('check input!')


def _vd(ii_, tint, subg=None, xvn=9):
    return _vd_fr_vv(_vv(ii_, tint, subg, xvn))


def _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, folder='cordex',
         subg=None, xvn=9):
    t1 = l__('>>>{}'.format(tint))
    ka0 = dict(dn=dn, gwl=gwl, po_=po_)
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


def _to1(v_, dgpi, freq=None):
    fn = '{}{}.nc'
    dn, gwl, po_ = dgpi[:3]

    def _fns(fff):
        return schF_keys_(po_, '_'.join((i for i in (v_, dn, fff, gwl, '__*')
                                         if i)))

    def _fn(fff):
        return fn.format(po_, '_'.join((i for i in (v_, dn, fff, gwl) if i)))

    freq = freq if freq else i__[v_][4]
    if len(freq) == 1:
        fns = _fns(freq[0])
        if fns:
            o = iris.load(fns)
            cubesv_(concat_cube_(o), _fn(freq[0]))
            for i in fns:
                os.remove(i)
    else:
        for oo, ff in zip(o, freq):
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


def _sv(v_, o, dgpi, freq=None, _nm=None):
    fn = '{}{}.nc'
    dn, gwl, po_ = dgpi[:3]

    def _fn(fff):
        return fn.format(po_,
                         '_'.join((i for i in (v_, dn, fff, gwl, _nm) if i)))

    freq = freq if freq else i__[v_][4]
    if len(freq) == 1:
        _meta(v_, o)
        cubesv_(o, _fn(freq[0]))
    else:
        for oo, ff in zip(o, freq):
            _meta(v_, oo)
            cubesv_(oo, _fn(ff))


def _dd(v_, cube, dgpi, freq=None, _nm=None):
    freq = freq if freq else i__[v_][4]
    if v_ in dgpi[-1] and cube:
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                       *freq)
        _sv(v_, o, dgpi, freq=freq, _nm=_nm)
        ll_(v_, t000)


#def _d0(v_, cube, dgpi, fA_=(), fK_={}, pK_=None, freq=None):
def _d0(v_, cube, dgpi, fA_=(), fK_={}, freq=None):
    if isIter_(v_):
        vk_ = v_[0]
        lmsg = '/'.join(('{}',) * len(v_)) + ' {}'
        vids = [i__[i][0] for i in v_]
        lmsg_ = lmsg.format(*vids, v_[0][:8])
    else:
        vk_ = v_
        lmsg_ = '{} {}'.format(i__[v_][0], v_)
    t000 = l__(lmsg_)
    freq = freq if freq else i__[vk_][4]
    cc = cube if isMyIter_(cube) else (cube,)
    o = i__[vk_][3](*cc, freq, *fA_, **fK_)
    #if pK_:
    #    pst_(o, **pK_)
    if not isIter_(v_) or len(v_) == 1:
        _sv(vk_, o, dgpi, freq=freq)
    else:
        for i, ii in zip(v_, o):
            _sv(i, ii, dgpi, freq=freq)
    ll_(lmsg_, t000)


def _tt(cube, y0y1=None):
    rm_t_aux_cube(cube)
    yr_doy_cube(cube)
    cat.add_season(cube, 'time', name='season', seasons=s4)
    tyrs, tdoy = cube.coord('year').points, cube.coord('doy').points
    y_y_ = y0y1 if y0y1 else tyrs[[0, -1]]
    tsss = cube.coord('season').points
    seasonyr_cube(cube, s4)
    tsyr = cube.coord('seasonyr').points
    ax_t = cube.coord_dims('time')[0]
    return (ax_t, y_y_, tyrs, tdoy, tsss, tsyr)


def _d1(v_, cube, dgpi, ax_t, y_y_,
        cK_={}, fA_=(), fK_={}, freq=None, out=False):
    def _inic(icK_):
        return initAnnualCube_(cube[0] if isMyIter_(cube) else cube,
                               y_y_, **icK_)
    if isIter_(v_):
        vk_ = v_[0]
        lmsg = '/'.join(('{}',) * len(v_)) + ' {}'
        vids = [i__[i][0] for i in v_]
        lmsg_ = lmsg.format(*vids, v_[0][:8])
    else:
        vk_ = v_
        lmsg_ = '{} {}'.format(i__[v_][0], v_)
    if ((freq and len(freq) != 1) or
        (freq is None and len(i__[vk_][4]) != 1)):
        raise Exception("exec-freq more than 1 currently not available!")
    t000 = l__(lmsg_)
    if isinstance(cK_, dict):
        o = _inic(cK_)
    else:
        o = iris.cube.CubeList([_inic(i) for i in cK_])
    o_ = _afm_n(cube, ax_t, i__[vk_][3], o, *fA_, **fK_)
    o = o_ if o_ else o
    if not isIter_(v_) or len(v_) == 1:
        _sv(vk_, o, dgpi, freq=freq)
    else:
        for i, ii in zip(v_, o):
            _sv(i, ii, dgpi, freq=freq)
    ll_(lmsg_, t000)
    if out:
        return o


def _mclimidx(dn=None, gwl=None, po_=None, il_=None,
              c_pr=None, c_evspsbl=None, c_prsn=None, c_mrro=None,
              c_sund=None, c_rsds=None, c_rlds=None,
              c_tas=None, c_tasmax=None, c_tasmin=None,
              c_tos=None, c_sic=None):

    dgpi = [dn, gwl, po_, il_]
    if any([i is None for i in dgpi]):
        raise ValueError("inputs of 'dn', 'gwl', 'po_', 'il_' are mandotory!")

    os.makedirs(po_, exist_ok=True)

    def _mm(v_, cube, freq=None):
        if v_ in il_ and cube:
            o = pSTAT_cube(cube, i__[v_][3] if i__[v_][3] else 'MEAN',
                           *i__[v_][4])
            _sv(v_, o, dgpi, freq=freq)

    _mm('PR', c_pr)                                                         #PR
    _mm('ET', c_evspsbl)                                                    #ET
    if 'EffPR' in il_ and c_pr is not None and c_evspsbl is not None:
        o = c_pr.copy(c_pr.data - c_evspsbl.data)
        #pst_(o, 'effective precipitation', var_name='eff_pr')
        _mm('EffPR', o)                                                  #EffPR
    _mm('PRSN', c_prsn)                                                   #PRSN
    if 'PRRN' in il_ and c_pr is not None and c_prsn is not None:
        o = c_pr.copy(c_pr.data - c_prsn.data)
        #pst_(o, 'rainfall_flux', var_name='prrn')
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
    if v_ in il_ and c_tasmax is not None and c_tasmin is not None:
        _d0(v_, (c_tasmax, c_tasmin), dgpi)                                #DTR
    _mm('SST', c_tos)                                                      #SST
    _mm('SIC', c_sic)                                                      #SIC


def _dclimidx(dn=None, gwl=None, po_=None, il_=None, y0y1=None,
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

    dgpi = [dn, gwl, po_, il_]
    if any([i is None for i in dgpi]):
        raise ValueError("inputs of 'dn', 'gwl', 'po_', 'il_' are mandotory!")

    os.makedirs(po_, exist_ok=True)

    if c_wsgsmax is not None:
        _dd('WindGustMax', c_wsgsmax, dgpi)                        #WindGustMax
        v_ = 'WindyDays'
        if v_ in il_:
            _d0(v_, c_wsgsmax, dgpi)                                 #WindyDays
    if (c_sfcWind is None and c_uas and c_vas and
        any([i in il_ for i in ['SfcWind', 'CalmDays', 'ConCalmDays',
                                'Wind975toSfc', 'Wind950toSfc',
                                'Wind925toSfc', 'Wind900toSfc']])):
        #c_sfcWind = c_uas.copy(np.sqrt(c_uas.data**2 + c_vas.data**2))
        c_sfcWind = ws_cube(c_uas, c_vas)
        pst_(c_sfcWind, 'surface wind speed', var_name='sfcWind')
    o = None
    if any([i in il_ for i in ['Wind975', 'Wind975toSfc', 'CalmDays975',
                               'ConCalmDays975']]):
        if c_ua975 and c_va975:
            #o = c_ua975.copy(np.sqrt(c_ua975.data**2 + c_va975.data**2))
            o = ws_cube(c_ua975, c_va975)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 97500.)
            #pst_(o, 'wind speed at 975 mb', var_name='w975')
    _dd('Wind975', o, dgpi)                                            #Wind975
    if o:
        v_ = 'CalmDays975'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays975
        v_ = 'ConCalmDays975'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)     #ConCalmDays975
            _d1(v_, o, dgpi, ax_t, y_y_, fA_=fA_)               #ConCalmDays975
    if 'Wind975toSfc' in  il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
        #pst_(o, 'wsr_975_to_sfc', '1')
        _dd('Wind975toSfc', o, dgpi)                              #Wind975toSfc
    o = None
    if any([i in il_ for i in ['Wind950', 'Wind950toSfc', 'CalmDays950',
                               'ConCalmDays950']]):
        if c_ua950 and c_va950:
            #o = c_ua950.copy(np.sqrt(c_ua950.data**2 + c_va950.data**2))
            o = ws_cube(c_ua950, c_va950)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 95000.)
            #pst_(o, 'wind speed at 950 mb', var_name='w950')
    _dd('Wind950', o, dgpi)                                            #Wind950
    if o:
        v_ = 'CalmDays950'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays950
        v_ = 'ConCalmDays950'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)     #ConCalmDays950
            _d1(v_, o, dgpi, ax_t, y_y_, fA_=fA_)               #ConCalmDays950
    if 'Wind950toSfc' in  il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
        #pst_(o, 'wsr_950_to_sfc', '1')
        _dd('Wind950toSfc', o, dgpi)                              #Wind950toSfc
    o = None
    if any([i in il_ for i in ['Wind925', 'Wind925toSfc', 'CalmDays925',
                               'ConCalmDays925']]):
        if c_ua925 and c_va925:
            #o = c_ua925.copy(np.sqrt(c_ua925.data**2 + c_va925.data**2))
            o = ws_cube(c_ua925, c_va925)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 92500.)
            #pst_(o, 'wind speed at 925 mb', var_name='w925')
    _dd('Wind925', o, dgpi)                                            #Wind925
    if o:
        v_ = 'CalmDays925'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays925
        v_ = 'ConCalmDays925'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)     #ConCalmDays925
            _d1(v_, o, dgpi, ax_t, y_y_, fA_=fA_)               #ConCalmDays925
    if 'Wind925toSfc' in  il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
        #pst_(o, 'wsr_925_to_sfc', '1')
        _dd('Wind925toSfc', o, dgpi)                              #Wind925toSfc
    o = None
    if any([i in il_ for i in ['Wind900', 'Wind900toSfc', 'CalmDays900',
                               'ConCalmDays900']]):
        if c_ua900 and c_va900:
            #o = c_ua900.copy(np.sqrt(c_ua900.data**2 + c_va900.data**2))
            o = ws_cube(c_ua900, c_va900)
            if c_ps:
                o = iris.util.mask_cube(o, c_ps.data < 90000.)
            #pst_(o, 'wind speed at 900 mb', var_name='w900')
    _dd('Wind900', o, dgpi)                                            #Wind900
    if o:
        v_ = 'CalmDays900'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays900
        v_ = 'ConCalmDays900'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)     #ConCalmDays900
            _d1(v_, o, dgpi, ax_t, y_y_, fA_=fA_)               #ConCalmDays900
    if 'Wind900toSfc' in  il_ and o and c_sfcWind:
        o = o.copy(o.data / c_sfcWind.data)
        #pst_(o, 'wsr_900_to_sfc', '1')
    o = None
    if any([i in il_ for i in ['Wind50m', 'CalmDays50m', 'ConCalmDays50m']]):
        if c_ua50m and c_va50m:
            #o = c_ua50m.copy(np.sqrt(c_ua50m.data**2 + c_va50m.data**2))
            o = ws_cube(c_ua50m, c_va50m)
            #pst_(o, 'wind speed at 50m', var_name='w50m')
    _dd('Wind50m', o, dgpi)                                            #Wind50m
    if o:
        v_ = 'CalmDays50m'
        if v_ in il_:
            _d0(v_, o, dgpi)                                       #CalmDays50m
        v_ = 'ConCalmDays50m'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)     #ConCalmDays50m
            _d1(v_, o, dgpi, ax_t, y_y_, fA_=fA_)               #ConCalmDays50m
    o = None
    if any([i in il_ for i in ['Wind100m', 'CalmDays100m', 'ConCalmDays100m']]):
        if c_ua100m and c_va100m:
            #o = c_ua100m.copy(np.sqrt(c_ua100m.data**2 + c_va100m.data**2))
            o = ws_cube(c_ua100m, c_va100m)
            #pst_(o, 'wind speed at 100m', var_name='w100m')
    _dd('Wind100m', o, dgpi)                                          #Wind100m
    if o:
        v_ = 'CalmDays100m'
        if v_ in il_:
            _d0(v_, o, dgpi)                                      #CalmDays100m
        v_ = 'ConCalmDays100m'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)    #ConCalmDays100m
            _d1(v_, o, dgpi, ax_t, y_y_, fA_=fA_)              #ConCalmDays100m
    o = None
    if any([i in il_ for i in ['Wind200m', 'CalmDays200m', 'ConCalmDays200m']]):
        if c_ua200m and c_va200m:
            #o = c_ua200m.copy(np.sqrt(c_ua200m.data**2 + c_va200m.data**2))
            o = ws_cube(c_va200m, c_va200m)
            #pst_(o, 'wind speed at 200m', var_name='w200m')
    _dd('Wind200m', o, dgpi)                                          #Wind200m
    if o:
        v_ = 'CalmDays200m'
        if v_ in il_:
            _d0(v_, o, dgpi)                                      #CalmDays200m
        v_ = 'ConCalmDays200m'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(o, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, o, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)    #ConCalmDays200m
            _d1(v_, o, dgpi, ax_t, y_y_, fA_=fA_)              #ConCalmDays200m
    _dd('SfcWind', c_sfcWind, dgpi)                                    #SfcWind
    if c_sfcWind:
        v_ = 'CalmDays'
        if v_ in il_:
            _d0(v_, c_sfcWind, dgpi)                                  #CalmDays
        v_ = 'ConCalmDays'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_sfcWind, y0y1)
            #cK_ = dict(name='continue calm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, c_sfcWind, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)#ConCalmDays
            _d1(v_, c_sfcWind, dgpi, ax_t, y_y_, fA_=fA_)          #ConCalmDays
    if c_tasmax:
        v_ = 'WarmDays'
        if v_ in il_:
            _d0(v_, c_tasmax, dgpi)                                   #WarmDays
        v_ = 'ColdDays'
        if v_ in il_:
            _d0(v_, c_tasmax, dgpi)                                   #ColdDays
        v_ = 'FreezingDays'
        if v_ in il_:
            _d0(v_, c_tasmax, dgpi)                               #FreezingDays
        v_ = 'CoolingDegDay'
        if v_ in il_:
            _d0(v_, c_tasmax, dgpi, fK_=dict(thr=20))
            #_d0(v_, c_tasmax, dgpi, fK_=dict(thr=20),
            #    pK_=dict(name='degree day cooling', var_name='dd20x_'))
                                                                 #CoolingDegDay
        v_ = 'ConWarmDays'
        if v_ in il_:
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_tasmax, y0y1)
            #cK_ = dict(name='continue warm days', units='days')
            fA_ = (tyrs,)
            #_d1(v_, c_tasmax, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_) #ConWarmDays
            _d1(v_, c_tasmax, dgpi, ax_t, y_y_, fA_=fA_)           #ConWarmDays
    if c_tasmin:
        v_ = 'FrostDays'
        if v_ in il_:
            _d0(v_, c_tasmin, dgpi)                                  #FrostDays
        v_ = 'TropicNights'
        if v_ in il_:
            _d0(v_, c_tasmin, dgpi)                                #TropicNight
        if any([i in il_ for i in ['SpringFrostDayEnd',
                                   'FirstDayWithoutFrost']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_tasmin, y0y1)
        v_ = 'SpringFrostDayEnd'
        if v_ in il_:
            #cK_ = dict(name='spring frost end-day', units=1,
            #           var_name='frost_end')
            fA_ = (tyrs, tdoy)
            #_d1(v_, c_tasmin, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)
            _d1(v_, c_tasmin, dgpi, ax_t, y_y_, fA_=fA_)
                                                             #SpringFrostDayEnd
        v_ = 'FirstDayWithoutFrost'
        if v_ in il_:
            #cK_ = dict(name='first day without frost', units=1,
            #           var_name='day1nofrost')
            fA_ = (tyrs, tdoy)
            #_d1(v_, c_tasmin, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)
            _d1(v_, c_tasmin, dgpi, ax_t, y_y_, fA_=fA_)
                                                          #FirstDayWithoutFrost
    v_ = 'ZeroCrossingDays'
    if v_ in il_ and c_tasmax and c_tasmin:
        _d0(v_, (c_tasmax, c_tasmin), dgpi)                   #ZeroCrossingDays
    if c_tas:
        v_ = 'MinusDays'
        if v_ in il_:
            _d0(v_, c_tas, dgpi)                                     #MinusDays
        v_ = 'DegDay20'
        if v_ in il_:
            _d0(v_, c_tas, dgpi, fK_=dict(thr=20))                    #DegDay20
        v_ = 'DegDay17'
        if v_ in il_:
            _d0(v_, c_tas, dgpi, fK_=dict(thr=17, left=True))         #DegDay17
        if any([i in il_ for i in ['DegDay8',
                                   'VegSeasonDayStart-5',
                                   'VegSeasonDayEnd-5', 'VegSeasonLength-5',
                                   'VegSeasonDayStart-2',
                                   'VegSeasonDayEnd-2', 'VegSeasonLength-2',
                                   'SuperCooledPR']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_tas, y0y1)
        v_ = 'DegDay8'
        if v_ in il_:
            #cK_ = dict(name='degree day g8 vegetation season',
            #           var_name='dd8_')
            fA_ = (tyrs,)
            #_d1(v_, c_tas, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)        #DegDay8
            _d1(v_, c_tas, dgpi, ax_t, y_y_, fA_=fA_)                  #DegDay8
        if any([i in il_ for i in ['VegSeasonDayStart-5', 'VegSeasonDayEnd-5',
                                   'VegSeasonLength-5']]):
            o = _d1(['VegSeasonDayStart-5', 'VegSeasonDayEnd-5'],
                    c_tas, dgpi, ax_t, y_y_,
                    #cK_=(dict(name='vegetation season day-start', units=1,
                    #          var_name='veg_s'),
                    #     dict(name='vegetation season day-end', units=1,
                    #          var_name='veg_e')),
                    fA_=(tyrs, tdoy),
                    fK_=dict(thr=5),
                    out=True)
            v_ = 'VegSeasonLength-5'
            t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
            o = o[1] - o[0]
            #pst_(o, 'vegetation season length', 'days', 'veg_l')
            _sv(v_, o, dgpi)
            ll_(v_, t000)                                          #VegSeason-5
        if any([i in il_ for i in ['VegSeasonDayStart-2', 'VegSeasonDayEnd-2',
                                   'VegSeasonLength-2']]):
            o = _d1(['VegSeasonDayStart-2', 'VegSeasonDayEnd-2'],
                    c_tas, dgpi, ax_t, y_y_,
                    #cK_=(dict(name='vegetation season day-start', units=1,
                    #          var_name='veg_s'),
                    #     dict(name='vegetation season day-end', units=1,
                    #          var_name='veg_e')),
                    fA_=(tyrs, tdoy),
                    fK_=dict(thr=2),
                    out=True)
            v_ = 'VegSeasonLength-2'
            t000 = l__('{} {} ... predata'.format(i__[v_][0], v_))
            o = o[1] - o[0]
            #pst_(o, 'vegetation season length', 'days', 'veg_l')
            _sv(v_, o, dgpi)
            ll_(v_, t000)                                          #VegSeason-2
    v_ = 'HumiWarmDays'
    if v_ in il_ and c_hurs and c_tas:
        _d0(v_, (c_hurs, c_tas), dgpi)                            #HumiWarmDays
    v_ = 'RhoS'
    if v_ in il_ and all([i is not None for i in (c_tas, c_huss, c_ps)]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_tas.copy(rho_fr_t_q_p_(c_tas.data, c_huss.data, c_ps.data))
        #pst_(o, 'surface air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_tas, c_huss, c_ps), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                     #RhoS
    v_ = 'Rho975'
    if v_ in il_ and all([i is not None for i in [c_ta975, c_hus975]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta975.copy(rho_fr_t_q_p_(c_ta975.data, c_hus975.data, 97500.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 97500.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta975, c_hus975, c_ps) if c_ps is not None else \
             (c_ta975, c_hus975)
        _f_n(_rho_ps_p, cL, 97500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho975
    v_ = 'Rho950'
    if v_ in il_ and all([i is not None for i in [c_ta950, c_hus950]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta950.copy(rho_fr_t_q_p_(c_ta950.data, c_hus950.data, 95000.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 95000.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta950, c_hus950, c_ps) if c_ps is not None else \
             (c_ta950, c_hus950)
        _f_n(_rho_ps_p, cL, 95000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho950
    v_ = 'Rho925'
    if v_ in il_ and all([i is not None for i in [c_ta925, c_hus925]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta925.copy(rho_fr_t_q_p_(c_ta925.data, c_hus925.data, 92500.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 92500.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta925, c_hus925, c_ps) if c_ps is not None else \
             (c_ta925, c_hus925)
        _f_n(_rho_ps_p, cL, 92500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho925
    v_ = 'Rho900'
    if v_ in il_ and all([i is not None for i in [c_ta900, c_hus900]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta900.copy(rho_fr_t_q_p_(c_ta900.data, c_hus900.data, 90000.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 90000.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta900, c_hus900, c_ps) if c_ps is not None else \
             (c_ta900, c_hus900)
        _f_n(_rho_ps_p, cL, 90000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho900
    v_ = 'Rho50m'
    if v_ in il_ and all([i is not None for i in [c_ta50m, c_hus50m, c_p50m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta50m.copy(rho_fr_t_q_p_(c_ta50m.data, c_hus50m.data, c_p50m.data))
        #pst_(o, 'air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_ta50m, c_hus50m, c_p50m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #Rho50m
    v_ = 'Rho100m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta100m, c_hus100m, c_p100m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta100m.copy(rho_fr_t_q_p_(c_ta100m.data, c_hus100m.data,
        #                               c_p100m.data))
        #pst_(o, 'air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_ta100m, c_hus100m, c_p100m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                  #Rho100m
    v_ = 'Rho200m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta200m, c_hus200m, c_p200m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta200m.copy(rho_fr_t_q_p_(c_ta200m.data, c_hus200m.data,
        #                               c_p200m.data))
        #pst_(o, 'air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_ta200m, c_hus200m, c_p200m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                  #Rho200m
    if c_pr:
        v_ = 'DryDays'
        if v_ in il_:
            _d0(v_, c_pr, dgpi)                                        #DryDays
        v_ = 'PRmax'
        _dd(v_, c_pr, dgpi)                                              #PRmax
        if any([i in il_ for i in ['PRgt10Days', 'PRgt25Days']]):
            _d0(['PRgt10Days', 'PRgt25Days'], c_pr, dgpi)            #ExtrPrDay
        if any([i in il_ for i in ['LnstDryDays', 'SuperCooledPR',
                                   'PR7Dmax']]):
            ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_pr, y0y1)
        v_ = 'PR7Dmax'
        if v_ in il_:
            #cK_ = dict(name='max 7-day precipitation', var_name='pr7d')
            fA_ = (tyrs,)
            #_d1(v_, c_pr, dgpi, ax_t, y_y_, cK_=cK_, fA_=fA_)         #PR7Dmax
            _d1(v_, c_pr, dgpi, ax_t, y_y_, fA_=fA_)                   #PR7Dmax
        v_ = 'LnstDryDays'
        if v_ in il_:
            t000 = l__('{} {} LL'.format(i__[v_][0], v_))
            #cK_ = dict(name='longest dry days', units='days')
            for ss in s4:
                yy = [y_y_[0] + 1, y_y_[-1]] if ss == s4[0] else y_y_
                ind = np.logical_and(tsss==ss, ind_inRange_(tsyr, *yy))
                _d1(v_, extract_byAxes_(c_pr, ax_t, ind), dgpi,
                    ax_t, yy,
                    #cK_=cK_,
                    fA_=(tsyr[ind],), freq=(ss,))
            ll_(v_, t000)                                          #LnstDryDays
    v_ = 'SuperCooledPR'
    if v_ in il_ and all([i is not None for i in
                          (c_pr, c_ps, c_tas, c_ta925, c_ta850, c_ta700,
                           c_hus925, c_hus850, c_hus700)]):
        #cK_ = dict(name='supercooled precipitation day', units='days')
        fA_ = (tyrs,)
        _d1(v_, (c_pr, c_ps, c_tas, c_ta925, c_ta850, c_ta700,
                 c_hus925, c_hus850, c_hus700), dgpi,
            #ax_t, y_y_, cK_=cK_, fA_=fA_)                       #SuperCooledPR
            ax_t, y_y_, fA_=fA_)                                 #SuperCooledPR
    if c_snc:
        v_ = 'SncDays'
        if v_ in il_:
            _d0(v_, c_snc, dgpi)                                       #SncDays
        v_ = 'Snc25Days'
        if v_ in il_:
            _d0(v_, c_snc, dgpi, fK_=dict(thr=25))                   #Snc25Days
    if c_snd:
        v_ = 'Snd10Days'
        if v_ in il_:
            _d0(v_, c_snd, dgpi)                                     #Snd10Days
        v_ = 'Snd20Days'
        if v_ in il_:
            _d0(v_, c_snd, dgpi)                                     #Snd20Days
    v_ = 'SNWmax'
    if v_ in il_ and c_snw:
        _dd(v_, c_snw, dgpi)                                            #SNWmax
    v_ = 'PRSNmax'
    if v_ in il_ and c_prsn:
        _dd(v_, c_prsn, dgpi)                                          #PRSNmax
    v_ = 'ColdRainWarmSnowDays'
    if v_ in il_ and c_pr and c_tas:
        _d0(v_, (c_pr, c_tas), dgpi)                      #ColdRainWarmSnowDays
    if (any([i in il_ for i in ['ColdRainDays', 'ColdRainGT10Days',
                                'ColdRainGT20Days']]) and c_pr and c_tas):
        _d0(['ColdRainDays', 'ColdRainGT10Days', 'ColdRainGT20Days'],
            (c_pr, c_tas), dgpi)                                  #ColdRainDays
    if (any([i in il_ for i in ['WarmSnowDays', 'WarmSnowGT10Days',
                                'WarmSnowGT20Days']]) and c_pr and c_tas):
        _d0(['WarmSnowDays', 'WarmSnowGT10Days', 'WarmSnowGT20Days'],
            (c_pr, c_tas), dgpi)                                  #WarmSnowDays
    if (any([i in il_ for i in ['WarmPRSNdays', 'WarmPRSNgt10Days',
                                'WarmPRSNgt20Days']]) and c_prsn and c_tas):
        _d0(['WarmPRSNdays', 'WarmPRSNgt10Days', 'WarmPRSNgt20Days'],
            (c_prsn, c_tas), dgpi)                                #WarmPRSNDays
    if (any([i in il_ for i in ['ColdPRRNdays', 'ColdPRRNgt10Days',
                                'ColdPRRNgt20Days']])
        and c_prsn and c_pr and c_tas):
        _d0(['ColdPRRNdays', 'ColdPRRNgt10Days', 'ColdPRRNgt20Days'],
            (c_pr, c_prsn, c_tas), dgpi)                          #ColdPRRNDays
    if (any([i in il_ for i in ['R5OScw', 'R1OScw', 'R5OSc', 'R1OSc']]) and
        c_pr and c_tas):
        #c_prsn = dPRSN_fr_PR_T_(c_pr, c_tas)
        o = dPRRN_fr_PR_T_(c_pr, c_tas)
    else:
        o = None
    if o and c_snc and c_snw:
        attr = None if 'PRRN' not in o.attributes else  o.attributes['PRRN']
        fK_ = dict(cSnw=c_snw, attr=attr)
        v_ = 'R5OScw'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=fK_)                          #R5OScw
        v_ = 'R1OScw'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=dict(thr_r=1., **fK_))        #R1OScw
    if o and c_snc:
        attr = None if 'PRRN' not in o.attributes else o.attributes['PRRN']
        fK_ = dict(attr=attr)
        v_ = 'R5OSc'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=fK_)                           #R5OSc
        v_ = 'R1OSc'
        if v_ in il_:
            _d0(v_, (o, c_snc), dgpi, fK_=dict(thr_r=1., **fK_))         #R1OSc


def _h6climidx(**kwArgs):
    _hclimidx(tint='h6', **kwArgs)


def _h3climidx(**kwArgs):
    _hclimidx(tint='h3', **kwArgs)


def _h1climidx(**kwArgs):
    _hclimidx(tint='h1', **kwArgs)


def _hclimidx(tint='h6', dn=None, gwl=None, po_=None, il_=None, y0y1=None,
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

    dgpi = [dn, gwl, po_, il_]
    if any([i is None for i in dgpi]):
        raise ValueError("inputs of 'dn', 'gwl', 'po_', 'il_' are mandotory!")
    os.makedirs(po_, exist_ok=True)

    v_ = tint+'SfcWind'
    if c_sfcWind is None and c_uas and c_vas and v_ in il_:
        #c_sfcWind = c_uas.copy(np.sqrt(c_uas.data**2 + c_vas.data**2))
        c_sfcWind = ws_cube(c_uas, c_vas)
        pst_(c_sfcWind, 'surface wind speed', var_name='sfcWind')
    _dd(v_, c_sfcWind, dgpi)                                         #hxSfcWind
    v_ = tint+'Wind975'
    o = None
    if v_ in il_ and c_ua975 and c_va975:
        #o = c_ua975.copy(np.sqrt(c_ua975.data**2 + c_va975.data**2))
        o = ws_cube(c_ua975, c_va975)
        if c_ps:
            o = iris.util.mask_cube(o, c_ps.data < 97500.)
        #pst_(o, 'wind speed at 975 mb', var_name='w975')
    _dd(v_, o, dgpi)                                                 #hxWind975
    v_ = tint+'Wind950'
    o = None
    if v_ in il_ and c_ua950 and c_va950:
        #o = c_ua950.copy(np.sqrt(c_ua950.data**2 + c_va950.data**2))
        o = ws_cube(c_ua950, c_va950)
        if c_ps:
            o = iris.util.mask_cube(o, c_ps.data < 95000.)
        #pst_(o, 'wind speed at 950 mb', var_name='w950')
    _dd(v_, o, dgpi)                                                 #hxWind950
    v_ = tint+'Wind925'
    o = None
    if v_ in il_ and c_ua925 and c_va925:
        #o = c_ua925.copy(np.sqrt(c_ua925.data**2 + c_va925.data**2))
        o = ws_cube(c_ua925, c_va925)
        if c_ps:
            o = iris.util.mask_cube(o, c_ps.data < 92500.)
        #pst_(o, 'wind speed at 925 mb', var_name='w925')
    _dd(v_, o, dgpi)                                                 #hxWind925
    v_ = tint+'Wind900'
    o = None
    if v_ in il_ and c_ua900 and c_va900:
        #o = c_ua900.copy(np.sqrt(c_ua900.data**2 + c_va900.data**2))
        o = ws_cube(c_ua900, c_va900)
        if c_ps:
            o = iris.util.mask_cube(o, c_ps.data < 90000.)
        #pst_(o, 'wind speed at 900 mb', var_name='w900')
    _dd(v_, o, dgpi)                                                 #hxWind900
    v_ = tint+'Wind50m'
    o = None
    if v_ in il_ and c_ua50m and c_va50m:
        #o = c_ua50m.copy(np.sqrt(c_ua50m.data**2 + c_va50m.data**2))
        o = ws_cube(c_ua50m, c_va50m)
        #pst_(o, 'wind speed at 50m', var_name='w50m')
    _dd(v_, o, dgpi)                                                 #hxWind50m
    v_ = tint+'Wind100m'
    o = None
    if v_ in il_ and c_ua100m and c_va100m:
        #o = c_ua100m.copy(np.sqrt(c_ua100m.data**2 + c_va100m.data**2))
        o = ws_cube(c_ua100m, c_va100m)
        #pst_(o, 'wind speed at 100m', var_name='w100m')
    _dd(v_, o, dgpi)                                                #hxWind100m
    v_ = tint+'Wind200m'
    o = None
    if v_ in il_ and c_ua200m and c_va200m:
        #o = c_ua200m.copy(np.sqrt(c_ua200m.data**2 + c_va200m.data**2))
        o = ws_cube(c_ua200m, c_va200m)
        #pst_(o, 'wind speed at 200m', var_name='w200m')
    _dd(v_, o, dgpi)                                                #hxWind200m
    v_ = tint+'RhoS'
    if v_ in il_ and all([i is not None for i in (c_tas, c_huss, c_ps)]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_tas.copy(rho_fr_t_q_p_(c_tas.data, c_huss.data, c_ps.data))
        #pst_(o, 'surface air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_tas, c_huss, c_ps), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                   #hxRhoS
    v_ = tint+'Rho975'
    if v_ in il_ and all([i is not None for i in [c_ta975, c_hus975]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta975.copy(rho_fr_t_q_p_(c_ta975.data, c_hus975.data, 97500.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 97500.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta975, c_hus975, c_ps) if c_ps is not None else \
             (c_ta975, c_hus975)
        _f_n(_rho_ps_p, cL, 97500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho975
    v_ = tint+'Rho950'
    if v_ in il_ and all([i is not None for i in [c_ta950, c_hus950]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta950.copy(rho_fr_t_q_p_(c_ta950.data, c_hus950.data, 95000.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 95000.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta950, c_hus950, c_ps) if c_ps is not None else \
             (c_ta950, c_hus950)
        _f_n(_rho_ps_p, cL, 95000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho950
    v_ = tint+'Rho925'
    if v_ in il_ and all([i is not None for i in [c_ta925, c_hus925]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta925.copy(rho_fr_t_q_p_(c_ta925.data, c_hus925.data, 92500.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 92500.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta925, c_hus925, c_ps) if c_ps is not None else \
             (c_ta925, c_hus925)
        _f_n(_rho_ps_p, cL, 92500., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho925
    v_ = tint+'Rho900'
    if v_ in il_ and all([i is not None for i in [c_ta900, c_hus900]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta900.copy(rho_fr_t_q_p_(c_ta900.data, c_hus900.data, 90000.))
        #if c_ps is not None:
        #    o = iris.util.mask_cube(o, c_ps.data < 90000.)
        #pst_(o, 'air density', 'kg m-3', 'rho')
        cL = (c_ta900, c_hus900, c_ps) if c_ps is not None else \
             (c_ta900, c_hus900)
        _f_n(_rho_ps_p, cL, 90000., v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho900
    v_ = tint+'Rho50m'
    if v_ in il_ and all([i is not None for i in [c_ta50m, c_hus50m, c_p50m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta50m.copy(rho_fr_t_q_p_(c_ta50m.data, c_hus50m.data, c_p50m.data))
        #pst_(o, 'air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_ta50m, c_hus50m, c_p50m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                 #hxRho50m
    v_ = tint+'Rho100m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta100m, c_hus100m, c_p100m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta100m.copy(rho_fr_t_q_p_(c_ta100m.data, c_hus100m.data,
        #                               c_p100m.data))
        #pst_(o, 'air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_ta100m, c_hus100m, c_p100m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                #hxRho100m
    v_ = tint+'Rho200m'
    if v_ in il_ and all([i is not None
                          for i in [c_ta200m, c_hus200m, c_p200m]]):
        t000 = l__('{} {}'.format(i__[v_][0], v_))
        #o = c_ta200m.copy(rho_fr_t_q_p_(c_ta200m.data, c_hus200m.data,
        #                               c_p200m.data))
        #pst_(o, 'air density', 'kg m-3', 'rho')
        _f_n(_rho_ps, (c_ta200m, c_hus200m, c_p200m), v_, dgpi)
        _to1(v_, dgpi)
        ll_(v_, t000)                                                #hxRho200m
    v_ = tint+'SuperCooledPR'
    if v_ in il_ and all([i is not None for i in
                          (c_pr, c_ps, c_tas, c_ta925, c_ta850, c_ta700,
                           c_hus925, c_hus850, c_hus700)]):
        ax_t, y_y_, tyrs, tdoy, tsss, tsyr = _tt(c_pr, y0y1)
        #cK_ = dict(name='supercooled precipitation events', units='1',
        #           var_name='prsc')
        fA_ = (tyrs,)
        _d1(v_, (c_pr, c_ps, c_tas, c_ta925, c_ta850, c_ta700,
                 c_hus925, c_hus850, c_hus700), dgpi,
            #ax_t, y_y_, cK_=cK_, fA_=fA_)                     #hxSuperCooledPR
            ax_t, y_y_, fA_=fA_)                               #hxSuperCooledPR


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
        return pi_ + '*' + freq + '/'
    elif isinstance(pi_, (tuple, list, set, np.ndarray)):
        return [_xx(i, freq) for i in pi_]
    else:
        raise ValueError("'pi_' must be str type or array-like of str")


def _to_xhr(cube, x=6, valid=True):
    nh = 24 / x
    cat.add_categorised_coord(cube, 'xxx', 'time',
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
        yf = _here_ + 'gcm_gwls_.yml'
    elif folder == 'cordex':
        yf = _here_ + 'gcm_gwls.yml'
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
    elif gwl == 'curr0130':
        y0y1 = curr
    else:
        y0y1 = curr
    return (y0y1, dn)


def cmip6_rcp_(il_, reg_d, reg_n, po_,
               sss=None, eee=None, idx=None, yml=None, tii=None,
               subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'cmip6_smhi_len.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip6_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='cmip6',
                 subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def cmip5_imp_rcp_(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'cmip5_import_.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip5_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='cmip5',
                 subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def cmip5_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None, idx=None, yml=None, tii=None,
                  subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'cmip5_import.yml')
    #pp = _pp(_here_ + 'cmip5_import_cp.yml')
    gg = _gg('cmip5')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cmip5_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], reg_n))
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
                 subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def norcp_rcp_(il_, reg_d, reg_n, po_,
               sss=None, eee=None, idx=None, yml=None, tii=None,
               subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'norcp_.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2norcp_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], tmp['prd'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day', '3hr'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, folder='norcp',
                 subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp_(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_import__.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                  sss=None, eee=None, idx=None, yml=None, tii=None,
                  subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_import.yml')
    gg = _gg()
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
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
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval(il_, reg_d, reg_n, po_,
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9):
    gwl = ''
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_import_eval.yml')
    pp_ = l_ind_(pp['p_'], [int(i) for i in idx.split(',')]) if idx else\
          pp['p_'][sss:eee]
    for p_ in pp_:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, None, po_, reg_d, subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def eur11_imp_eval_dmi(il_, reg_d, reg_n, po_, tii=None, subg=None, xvn=9):
    gwl = ''
    pp = _pp(_here_ + 'eur-11_import_eval.yml')
    y0y1 = [1989, 2010]
    for p_ in pp['p__']:
        tmp = path2cordex_info_(p_)
        dn = '_'.join((tmp['gcm'], tmp['rcp'], tmp['rip'], tmp['rcm'],
                       tmp['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        pi_ = pp['root'] + p_
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_eval(il_, reg_d, reg_n, po_, yml=None, tii=None,
                    subg=None, xvn=9):
    gwl = ''
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_smhi-rca4.yml')
    p_ = pp['root'] + str(pp['eval']) + '/netcdf/'
    gcm = pp[pp['eval']]['gcm']
    rcp = pp[pp['eval']]['rcp']
    rip = pp[pp['eval']]['rip']
    rcm = pp[pp['eval']]['rcm']
    ver = pp[pp['eval']]['ver']
    dn = '_'.join((gcm, rcp, rip, rcm, ver, reg_n))
    t0 = l__('>>>>>>>' + dn)
    tis = ['mon', 'day'] if tii is None else tii.split(',')
    for tint in tis:
        _xyz(il_, tint, p_, dn, gwl, None, po_, reg_d, subg=subg, xvn=xvn)
    ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp_(il_, reg_d, reg_n, po_,
                    sss=None, eee=None, idx=None, yml=None, tii=None,
                    subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_smhi-rca4.yml')
    pp_ = l_ind_(pp['h248'], [int(i) for i in idx.split(',')]) if idx else\
          pp['h248'][sss:eee]
    for ppi in pp_:
        pi_ = '{}{}/netcdf/'.format(pp['root'], ppi)
        dn = '_'.join((pp[ppi]['gcm'], pp[ppi]['rcp'], pp[ppi]['rip'],
                       pp[ppi]['rcm'], pp[ppi]['ver'], reg_n))
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, '', None, po_, reg_d, subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def eur11_smhi_rcp(il_, reg_d, reg_n, po_, gwl='gwl15', curr=[1971, 2000],
                   sss=None, eee=None, idx=None, yml=None, tii=None,
                   subg=None, xvn=9):
    pp = _pp(yml) if yml else _pp(_here_ + 'eur-11_smhi-rca4.yml')
    gg = _gg()
    pp_ = l_ind_(pp['rcps'], [int(i) for i in idx.split(',')]) if idx else\
          pp['rcps'][sss:eee]
    for p0p1 in pp_:
        pi0, pi1 = p0p1[0], p0p1[1]
        pi0_, pi1_ = ('{}{}/netcdf/'.format(pp['root'], pi0),
                      '{}{}/netcdf/'.format(pp['root'], pi1))
        dn = '_'.join((pp[pi1]['gcm'], pp[pi1]['rcp'], pp[pi1]['rip'],
                       pp[pi1]['rcm'], pp[pi1]['ver'], reg_n))
        y0y1, dn = _yy_dn(pp[pi1], dn, gwl, gg, curr)
        if y0y1 is None:
            continue
        pi_ = pi0_ if y0y1[1] <= 2005 else (pi1_ if y0y1[0] > 2005 else
                                            [pi0_, pi1_])
        t0 = l__('>>>>>>>' + dn)
        tis = ['mon', 'day'] if tii is None else tii.split(',')
        for tint in tis:
            _xyz(il_, tint, pi_, dn, gwl, y0y1, po_, reg_d, subg=subg, xvn=xvn)
        ll_('<<<<<<<' + dn, t0)


def eobs20_(il_, reg_d, reg_n, po_, y0y1=None):
    idir = '/nobackup/rossby22/sm_chali/DATA/hw2018/iii/obs/EOBS20/'
    vo = {'c_pr': ('rr', 1. / 3600 / 24, 'kg m-2 s-1'),
          'c_tas': ('tg', None, 'K'),
          'c_tasmax': ('tx', None, 'K'),
          'c_tasmin': ('tn', None, 'K'),
          'c_rsds': ('qq', 1, 'K'),
          }

    def _eobs_load(var):
        o = iris.load_cube('{}{}_ens_mean_0.1deg_reg_v20.0e.nc'
                           .format(idir, vo[var][0]))
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
        vD = _vd(il_, tint, subg=None, xvn=999)[0]
        ccc = dict()
        for i in vD[1].keys():
            if i in vo.keys():
                o = _eobs_load(i)
            ccc.update(i, o)
        return ccc

    t0 = l__('>>>monthly')
    t1 = l__(' >>loading data')
    cccc = _getccc('month')
    ll_(' <<loading data', t1)
    _mclimidx(il_=il_, dn='EOBS20_' + reg_n, gwl='', po_=po_, **cccc)
    ll_('<<<monthly', t0)
    t0 = l__('>>>daily')
    t1 = l__(' >>loading data')
    cccc = _getccc('day')
    ll_(' <<loading data', t1)
    _dclimidx(il_=il_, dn='EOBS20_' + reg_n, gwl='', po_=po_, y0y1=y0y1,
              **cccc)
    ll_('<<<daily', t0)


def erai_(il_, reg_d, reg_n, po_, y0y1=None, tii=None, subg=None, xvn=9):
    #from uuuu.cccc import _unify_xycoord_points
    #idir = '/nobackup/rossby22/sm_chali/DATA/hw2018/iii/obs/ERAI/'
    #def _erai_load(idir, var):
    #    o = iris.load(idir + var + '_day_ERA*.nc')
    #    o = concat_cube_(o)
    #    if reg_d is not None:
    #        o = intersection_(o, **reg_d)
    #    return extract_period_cube(o, *y0y1)
    #t0 = l__('>>>loading data')
    #c_pr = _erai_load(idir, 'pr')
    #c_tas = _erai_load(idir, 'tas')
    #c_tasmax = _erai_load(idir, 'tasmax')
    #c_tasmin = _erai_load(idir, 'tasmin')
    #c_tx_m = pSTAT_cube(c_tasmax, 'MEAN', 'month')
    #c_tn_m = pSTAT_cube(c_tasmin, 'MEAN', 'month')
    #_unify_xycoord_points((c_tx_m, c_tn_m))
    #m__ = dict(c_pr=c_pr, c_tas=c_tas, c_tasmax=c_tx_m, c_tasmin=c_tn_m)
    #d__ = dict(c_pr=c_pr, c_tas=c_tas, c_tasmax=c_tasmax, c_tasmin=c_tasmin)
    #ll_('<<<loading data', t0)
    #t0 = l__('>>>monthly')
    #_mclimidx(il_=il_, dn='ERAI_' + reg_n, gwl='', po_=po_, **m__)
    #ll_('<<<monthly', t0)
    #t0 = l__('>>>daily')
    #_dclimidx(il_=il_, dn='ERAI_' + reg_n, gwl='', po_=po_, y0y1=y0y1, **d__)
    #ll_('<<<daily', t0)
    idir = '/home/rossby/imports/obs/ECMWF/ERAINT/input/'
    tis = ['mon', 'day'] if tii is None else tii.split(',')
    for tint in tis:
        _xyz(il_, tint, idir, 'ERAI_{}'.format(reg_n), '', y0y1, po_, reg_d,
             folder='cmip5', subg=subg, xvn=xvn)


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
                             "gwl4")
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
    if args.lll:
        lo0, lo1, la0, la1 = [float(i) for i in args.lll.split(',')]
        reg_d = {'longitude': [lo0, lo1], 'latitude': [la0, la1]}
    else:
        reg_d = None
    #reg_d = {'longitude': [10.0, 23.0], 'latitude': [55.0, 69.0]}
    #reg_n = 'SWE'
    #reg_d = {'longitude': [-25.0, 45.0], 'latitude': [25.0, 75.0]}
    reg_n = args.domain if args.domain else \
            ('LLL' if reg_d is not None else 'ALL')
    if args.rdir:
        rdir = args.rdir
    else:
        rxx = os.environ.get('r26')
        rdir = '{}DATA/energi/res/'.format(rxx)
        #rdir = '/nobackup/rossby22/sm_chali/DATA/energi/res/'
    pf_ = lambda x: '{}{}/{}/'.format(rdir, x, reg_n)
    poe_ = pf_('eval')
    poo_ = pf_('obs')
    pcdx = pf_('h248/cordex/EUR11') #rdir + 'h248/cordex/EUR11/' + reg_n + '/'
    pcmp5 = pf_('h248/cmip5') #rdir + 'h248/cmip5/' + reg_n + '/'
    pcmp6 = pf_('h248/cmip6') #rdir + 'h248/cmip6/' + reg_n + '/'
    pnorcp = pf_('h248/norcp') #rdir + 'h248/norcp/' + reg_n + '/'
    pcdx_ = pf_('gwls/cordex/EUR11') #rdir + 'gwls/' + reg_n + '/'
    pcmp5_ = pf_('gwls/cmip5') #rdir + 'obs/' + reg_n + '/'
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
                 yml=args.yml, tii=args.tint, subg=args.subg, xvn=args.xvn)

    if args.opt in ('0', 'eobs20'):
        eobs20_(il_, reg_d, reg_n, poo_)
    elif args.opt in ('1', 'erai'):
        erai_(il_, reg_d, reg_n, poo_,
              tii=args.tint, subg=args.subg, xvn=args.xvn)
    elif args.opt in ('2', 'cdx_eval'):
        eur11_imp_eval(il_, reg_d, reg_n, poe_, **seiyt)
    elif args.opt in ('21', 'cdx_eval_dmi'):
        eur11_imp_eval_dmi(il_, reg_d, reg_n, poe_,
                           tii=args.tint, subg=args.subg, xvn=args.xvn)
    elif args.opt in ('22', 'cdx_eval_smhi'):
        eur11_smhi_eval(il_, reg_d, reg_n, poe_, yml=args.yml,
                        tii=args.tint, subg=args.subg, xvn=args.xvn)
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

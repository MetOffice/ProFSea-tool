"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

from math import cos, asin, sqrt
import pandas as pd
import os

from profsea.config import settings


def closest(data, v):
    """
    :param data: all tide gauges latitude and longitude
    :param v: site latitude and longitude
    :return: DataFrame row of the closest TG to site lat and lon
    """
    return min(data, key=lambda p: distance(v['lat'], v['lon'],
                                            p['lat'], p['lon']))


def distance(lat1, lon1, lat2, lon2):
    """
    The haversine formula determines the great-circle distance between two
    points on a sphere given their longitudes and latitudes.
    Ref: https://en.wikipedia.org/wiki/Haversine_formula

    :param lat1: site latitude
    :param lon1: site longitude
    :param lat2: TG latitude
    :param lon2: TG longitude
    :return: distance between points on a sphere
    """
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * \
        (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def extract_site_info(data_source, data_type, region, site_name, latlon):
    """
    Extract all available tide gauges from station list
    Check if site_name is in this list, if so return metadata
    If not construct a dataframe in the same format that contains user defined
    metadata.
    :param data_source: source of tide gauge information
    :param data_type: temporal scale of tide gauge data
    :param region: region of the world to assess sea level rise
    :param site_name: site specific location
    :param latlon: site location's latitude and longitude ([] if using tide
                   gauge)
    :return: Data frame of all metadata (tide gauge or site specific) for each
    location requested
    """
    print('running function extract_site_info')

    # Get a list of all available tide gauges from data_source
    df = tide_gauge_locations(region=region, source=data_source,
                              type=data_type)

    # Create an empty pandas DataFrame to hold site information
    dfObj = pd.DataFrame()
    for i, location in enumerate(site_name):
        # Re-format user defined site name
        if data_source == 'PSMSL':
            if location == 'Aberdeen':
                print(f'CAUTION - There are two gauges available for ' +
                      'Aberdeen, using ABERDEEN I (1931-2019)')
                site_to_check = 'ABERDEEN I'
            else:
                site_to_check = location.upper()
        else:
            raise ValueError("data source not recognised")

        # --------------------------------------------------------------------

        if site_to_check in df.index:
            print(f'{site_to_check} - Site metadata taken from {data_source}')
            site_data = df.loc[site_to_check, :]

            if data_source == 'PSMSL':
                if 'NEWPORT' in site_to_check:
                    print(f'CAUTION - There are two Newport gauges listed in' +
                          'the file list from PSMSL')
                    user_input = input('Location of Newport required, UK or ' +
                                       'US?')
                    if user_input == 'UK':
                        print('Newport in the UK selected')
                        site_data = site_data[~site_data.index.duplicated(
                            keep='last')].transpose()
                    else:
                        print('Newport in the US selected')
                        site_data = site_data[~site_data.index.duplicated(
                            keep='first')].transpose()

            df_temp = pd.DataFrame(site_data).transpose()
            dfObj = dfObj.append(df_temp)

        elif latlon != [[]]:
            print(f'{site_to_check} - Site metadata taken from user input')
            data_type = 'user defined'
            station_id = 'NA'
            latitude = latlon[i][0]
            longitude = latlon[i][1]
            site_data = [[site_to_check, data_type, station_id, latitude,
                        longitude]]

            df_temp = pd.DataFrame(site_data, columns=[
                'Location', 'Dataset type', 'Station ID', 'Latitude',
                'Longitude'])
            df_temp = df_temp.set_index('Location')
            dfObj = dfObj.append(df_temp)

        else:
            raise IndexError(f'No site metadata for this site: {site_to_check}, have you spelled it correctly?')            

    return dfObj


def find_nearest_station_id(root_dir, data_source, data_type, data_region,
                            site_lat, site_lon):
    """
    Extracts all available tide gauges from data sources, then calculates
    the closest tide gauge to the site based on latitude and longitude using
    the Haversine formula
      ** N.B. Testing only completed on PSMSL station list **

    :param root_dir: base directory for all input and output data
    :param data_source: source of tide gauge information
    :param data_type: temporal scale of tide gauge data
    :param data_region: region of the world to assess sea level rise
    :param site_lat: site latitude
    :param site_lon: site longitude
    :return: Nearest tide gauge name and station ID
    """
    # Get a list of all available tide gauges from data_source, extract
    # latitude and longitude
    df = tide_gauge_locations(region=data_region, source=data_source,
                              type=data_type, in_dir=root_dir)
    latitude = df.loc[:, 'Latitude'].tolist()
    longitude = df.loc[:, 'Longitude'].tolist()

    # Re-format latitude and longitude
    temp_data_list = []
    for i in range(len(latitude)):
        temp_data_list.append({'lat': latitude[i], 'lon': longitude[i]})

    # Find the closest latitude and longitude in list to site location
    site_location = {'lat': site_lat, 'lon': site_lon}
    closest_latlon = closest(temp_data_list, site_location)

    # Extract a dataframe based on the closest latitude and longitude
    df_nearest_tg = df.loc[(df['Latitude'] == closest_latlon['lat']) & (
            df['Longitude'] == closest_latlon['lon'])]
    tg_station_name = df_nearest_tg.index.values.astype(str)[0]
    tg_station_id = df_nearest_tg.loc[tg_station_name, 'Station ID']

    return tg_station_id, tg_station_name


def get_psmsl_gauges(data_type):
    """
    Search for available tide gauges from PSMSL, extract metadata using
    function read_psmsl_list_of_gauges, re-order into correct format for
    future code.
    :param data_type: temporal scale of tide gauge data
    :return: Re-formatted dataframe of available tide gauges from PSMSL
    """
    print('running function get_psmsl_gauges')

    columns_needed = ['Location', 'ID', 'Lat', 'Lon']
    ids = []
    tide_gauges = []

    for d in data_type:
        # Read in the PSMSL station list of all gauges worldwide
        psmsl_gauges = read_psmsl_list_of_gauges(d)

        psmsl_basedir = settings["tidegaugeinfo"]["psmsldir"]

        for row in psmsl_gauges.index.values:
            idn = int(psmsl_gauges.at[row, 'ID'])
            if any(j == idn for j in ids):
                continue
            ids.append(idn)
            tide_gauge_file = f'{idn}.rlrdata'
            tide_gauge_file_full = os.path.join(
                f'{psmsl_basedir}rlr_{d}', 'data', f'{tide_gauge_file}')

            if os.path.isfile(tide_gauge_file_full):
                l = psmsl_gauges.loc[row, columns_needed].tolist()
                l.insert(1, f'PSMSL - {data_type[0]}')
                tide_gauges.append(l)

    return tide_gauges


def read_psmsl_list_of_gauges(data_type):
    """
    Read in a list of all gauges available from PSMSL.
    :param data_type: temporal scale of tide gauge data, for PSMSL can be
                      annual, monthly or hourly
    :return: Pandas data frame of all available tide gauges from PSMSL
    """
    print('running function read_psmsl_list_of_gauges')

    psmsl_basedir = settings["tidegaugeinfo"]["psmsldir"]

    if data_type == 'annual':
        station_list_file = 'filelist.txt'
        filename = f'{psmsl_basedir}rlr_annual/{station_list_file}'
        df = pd.read_csv(filename, sep="; ", header=None, names=[
            'ID', 'Lat', 'Lon', 'Location', 'num', 'num2', 'flag'],
            engine='python')
        df.drop(columns=['num', 'num2', 'flag'], inplace=True)
        df['Location'] = df['Location'].str.strip()
        df = df[['Location', 'ID', 'Lat', 'Lon']]
    elif data_type == 'monthly' or 'hourly':
        raise ValueError("Monthly or Hourly tide gauge data not available " +
                         "for PSMSL gauges")

    return df


def tide_gauge_locations(**args):
    """
    Wrapper to read in a list of all available tide gauges found on PSMSL.
    :param args: Argument list
        region --> region of the world to assess sea level rise
        source --> source of tide gauge information
        type --> temporal scale of tide gauge data
    :return: Data frame of all available tide gauges from relevant source,
        indexed by Location
    """
    print('running function tide_gauge_location')

    # Name of columns in dataframe.
    columns = ['Location', 'Dataset type', 'Station ID', 'Latitude',
               'Longitude']

    # Update parameters passed to this function
    data_region = None
    data_source = None
    data_type = None

    if 'region' in args:
        data_region = args['region']
    if 'source' in args:
        data_source = args['source']
    if 'type' in args:
        data_type = args['type']

    if data_region is None or data_source is None:
        raise ValueError("tide_gauge_locations: must specify names of " +
                         "region and source")

    if data_source == 'PSMSL':
        if 'type' not in args:
            raise ValueError(
                "tide_gauge_locations: must specify data_type when " +
                "specifying PSMSL as source (hourly or monthly)")

    if data_source == 'PSMSL':
        tide_gauges = get_psmsl_gauges(data_type)
    else:
        raise ValueError(f'Source {data_source} not known')

    df = pd.DataFrame(tide_gauges, columns=columns)

    return df.set_index('Location')

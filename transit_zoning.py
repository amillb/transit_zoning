# -*- coding: utf-8 -*-
"""Transit zoning analysis

This code identifies high-quality transit areas designated under California law
It takes GTFS files as an input, 
and generates geospatial files of high-quality stops and the 1/2 mile radius around them

Adam Millard-Ball and Daniel Sjoholm, UCLA
with assistance from Marcel Martin
""" 

import partridge as ptg
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math
import warnings

##################
# Global variables
##################

# GTFS route types
# includes Intercity Rail. Note we exclude 6 - Aerial lift, suspended cable car (e.g., gondola lift, aerial tramway)
# https://gtfs.org/documentation/schedule/reference/#routestxt
gtfs_route_types = {'rail':[0, 1, 2, 5, 7, 12], 
               'ferry': [4],
               'brt': [3]}
gtfs_mode_names = {0:'light_rail', 1:'metro', 2:'rail', 3:'bus', 
                   4:'ferry', 5:'cablecar', 7:'funicular', 12:'monorail' }
minimal_route_types_excluded = [3,5] # exclude BRT and cable car, only from minimal

testing = True # QA/QC for development. If True, csv files with routes and stops are saved

# Specify the directory where all the zipped GFTS feeds exist, and the other data (e.g., Amtrak)
# You can adjust these paths if you want to keep the GTFS and other input data in another directory
base_path = os.path.join(os.getcwd(), 'transit_data')
gtfs_path = os.path.join(base_path, 'gtfs')
output_path = os.path.join(os.getcwd(), 'transit_output')
figure_path = os.path.join(os.getcwd(), 'figures')
incremental_output_path = os.path.join(output_path, 'incremental')

if not(os.path.exists(base_path)):
    raise Exception(f'transit_data directory not found. It should be a subdirectory in {os.getcwd()}.')
if not(os.path.exists(gtfs_path)):
    raise Exception(f'gtfs directory not found. It should be a subdirectory in {base_path}.')
for path in [output_path, incremental_output_path, figure_path]:
    if not(os.path.exists(path)):
        os.mkdir(path)

# which years the analysis will be run for
# each year's data is a subfolder in /GTFS
yearlist = [fn.name for fn in os.scandir(gtfs_path) if fn.is_dir()]
print(f"Found GTFS data for the following years: {', '.join(yearlist)}")

# define the individual steps that make up the differences between minimal and maximal
# separate out increments by the function that they are used in
peaks_increments = ['peak_definition', 'consolidate_infrequent']
intersections_increments = ['stop_distance','intersecting_stops']
stops_increments = ['railBRT_included', 'subway_entrances','parcels', 'planned_transit']
all_increments = ['minimal'] + peaks_increments + intersections_increments + stops_increments



class GTFSFeed:
    """A custom GTFS object for handling concatenated GTFS tables."""
    def __init__(self, routes, trips, stop_times, stops, frequencies=None, rail_ferry_brt_stations=None):
        self.routes = routes
        self.trips = trips
        self.stop_times = stop_times
        self.stops = stops       
        self.frequencies = frequencies if frequencies is not None else pd.DataFrame()
        self.rail_ferry_brt_stations = rail_ferry_brt_stations if rail_ferry_brt_stations is not None else gpd.GeoDataFrame()

def load_and_combine_gtfs(gtfs_path, year, include_brt=True):
    """Step 1
    Loads multiple GTFS files and combines them into a single object
    
    Also returns rail stations, as these are not all included in the busiest-day GTFS feed (e.g. Sat only)
    
    If include_brt is True, the manually identified BRT routes are imported from brt_routes.csv.
    Normally, this should only be False for testing

    Returns a feed"""

    combined_routes = {}
    combined_trips = {}
    combined_stop_times = {}
    combined_stops = {}
    combined_frequencies = {}
    combined_rail_ferry_brt_stations = {}

    # We want to count loop routes as a single route, not a different route for each direction
    # The code merges most CW/CCW routes automatically based on their names. 
    # However, some need their names to be corrected so they are recognized as CW/CCW
    clockwise_routes = pd.read_csv(os.path.join(base_path,'other_transit','extra_clockwise_routes.csv'))

    path = os.path.join(gtfs_path, year)
    gtfs_fns= [os.path.join(path, ff) for ff in os.listdir(path) if ff.endswith('.zip') ]
    if len(gtfs_fns)==0:
        raise Exception(f'No GTFS (zip) files found in {path}')

    # make sure that the prefix will uniquely identify the file
    prefixes = [os.path.splitext(os.path.basename(fn))[0][:15].strip('_') for fn in gtfs_fns]

    if len(set(prefixes)) != len(prefixes):
        raise Exception('The start of GTFS filenames are not unique. Perhaps you have duplicate GTFS files?')

    print(f"Loading feeds from {len(gtfs_fns)} GTFS files")
    for fn in gtfs_fns:
        prefix = os.path.splitext(os.path.basename(fn))[0][:15].strip('_')
        print(f"Processing feed: {fn}")

        # get busiest date, to avoid double counting agencies with school/non school service
        _date, service_ids = ptg.read_busiest_date(fn)
        feed = ptg.load_feed(fn, view={'trips.txt': {'service_id': service_ids}})
        
        # Extract DataFrames
        routes = feed.routes.copy()
        trips = feed.trips.copy()
        stop_times = feed.stop_times.copy()
        stops = feed.stops.copy()
        assert routes.index.is_unique # to allow use of index below

        if 'arrival_time' not in stop_times.columns or pd.isnull(stop_times.arrival_time).all():
            print(f'No stop times found for {fn}. This may be on-demand service only.')
            continue

        # fill NaNs for arrival times. This is the case in some older feeds (e.g. Omnitrans) that only have times for time points
        if pd.isnull(stop_times.arrival_time).any():
            assert pd.notnull(stop_times.trip_id).all() and pd.notnull(stop_times.stop_sequence).all()
            stop_times.sort_values(by=['trip_id','stop_sequence'], inplace=True)
            stop_times.arrival_time = stop_times.arrival_time.fillna(stop_times.departure_time)
            stop_times.arrival_time = stop_times.groupby('trip_id').arrival_time.ffill() 

        # Read agency data - needed for BRT
        if hasattr(feed, 'agency') and not feed.agency.empty:
            agency = feed.agency.copy()
            if 'agency_id' in routes.columns:
                routes = routes.merge(agency[['agency_id', 'agency_name']],
                                on='agency_id', how='left')
                routes.fillna({'agency_name': agency.iloc[0]['agency_name']}, inplace=True)
            else:
                if len(agency)!=1:  # if not, we need to adapt the code
                    raise NotImplementedError('Non unique agency ID found')
                agencyname = agency.iloc[0]['agency_name']
                routes['agency_name'] = agencyname if pd.notnull(agencyname) else prefix
        else:
             routes['agency_name'] = prefix

        # Read frequencies if available
        frequencies = feed.frequencies.copy() if hasattr(feed, 'frequencies') else pd.DataFrame()
        
        # replace routes that are separate for clockwise/counterclockwise
        # first, correct the names for a couple manually
        cw_routes_todo = clockwise_routes[clockwise_routes.GTFS_name==os.path.basename(fn).split('.')[0]]
        for idx, cwr in cw_routes_todo.iterrows():
            routes.loc[routes.route_long_name==cwr.route_long_name, 'route_long_name'] = cwr.route_long_name_corrected

        # then, merge them
        if 'route_short_name' not in routes.columns:
            routes['route_short_name'] = routes.route_long_name
        for routecol in ['route_short_name','route_long_name']: # sometimes it's only in one
            wiseroutes = routes[routes[routecol].astype(str).str.lower().str.contains('counter-?clockwise', regex=True)]
            for ccw_idx, ccw_route in wiseroutes.iterrows():
                rname = ccw_route[routecol].lower().replace('counter-','').replace('counter','')
                cw_idx = routes[(routes[routecol].astype(str).str.lower().str.contains(rname, regex=False)) &
                                (routes.agency_name==ccw_route.agency_name)].index
                cw_idx = cw_idx.drop(ccw_idx, errors='ignore') # only if the route is just "clockwise"
                if len(cw_idx)==0: continue
                if len(cw_idx)>1: raise Exception(f'multiple cw/ccw routes found for {rname}')
                oldroute_id, newroute_id = routes.loc[cw_idx[0],'route_id'], ccw_route.route_id
                routes.loc[ccw_idx,routecol] = rname.replace('clockwise','ccw_and_cw')
                trips.loc[trips.route_id==oldroute_id, 'route_id'] = newroute_id
                routes.drop(cw_idx, inplace=True)  # retain only the ccw version

        # make ids unique across agencies
        routes['prefixed_route_id'] = prefix + "_" + routes['route_id'].astype(str)
        routes['GTFS_filename'] = os.path.basename(fn)
        
        trips['prefixed_route_id'] = prefix + "_" + trips['route_id'].astype(str)
        trips['prefixed_trip_id'] = prefix + "_" + trips['trip_id'].astype(str)
        
        stop_times['prefixed_trip_id'] = prefix + "_" + stop_times['trip_id'].astype(str)
        stop_times['prefixed_stop_id'] = prefix + "_" + stop_times['stop_id'].astype(str)
        
        stops['prefixed_stop_id'] = prefix + "_" + stops['stop_id'].astype(str)
        stops['GTFS_filename'] = os.path.basename(fn)
 
        if not frequencies.empty and 'trip_id' in frequencies.columns:
            frequencies['prefixed_trip_id'] = prefix + "_" + frequencies['trip_id'].astype(str)
        
        combined_routes[fn] = routes
        combined_trips[fn] = trips
        combined_stop_times[fn] = stop_times
        combined_stops[fn] = stops
        if not frequencies.empty:
            combined_frequencies[fn] = frequencies

        # get rail stations only, including weekend-only ones
        geofeed = ptg.load_geo_feed(fn)
        rail_ferry_gtfs_codes = gtfs_route_types['rail'] + gtfs_route_types['ferry']
        rail = geofeed.routes[geofeed.routes.route_type.isin(rail_ferry_gtfs_codes)][['route_id','route_type']].set_index('route_id')
        rail = rail.join(geofeed.trips.set_index('route_id')['trip_id'], how='left').set_index('trip_id')
        rail = rail.join(geofeed.stop_times.set_index('trip_id')['stop_id'], how='left').set_index('stop_id')
        warnings.simplefilter(action='ignore', category=FutureWarning) # suppress warning about pyproj format
        rail = rail.join(geofeed.stops.set_index('stop_id')[['stop_name','geometry']]).drop_duplicates()
        warnings.simplefilter(action='default', category=FutureWarning)        
        rail = rail[~rail['stop_name'].astype(str).str.contains('NOT A PUBLIC STOP')]         # drop non-public stops - just 1 in VTA
        rail['GTFS_filename'] = os.path.basename(fn)
        if len(rail)>0:
            combined_rail_ferry_brt_stations[fn] = gpd.GeoDataFrame(rail, geometry='geometry', crs='EPSG:4326')
            assert combined_rail_ferry_brt_stations[fn].index.is_unique
            # add frequencies
            rail_freqs = trips[['trip_id','route_id']].set_index('trip_id').join(stop_times.set_index('trip_id'))
            rail_freqs = routes[['route_id','agency_name','route_type']].set_index('route_id').join(rail_freqs.set_index('route_id'))
            rail_freqs = rail_freqs[rail_freqs.route_type.isin(rail_ferry_gtfs_codes)].groupby(['agency_name','stop_id']).size()
            rail_freqs = pd.DataFrame(rail_freqs, columns=['trains_perday']).reset_index(level=0)

            combined_rail_ferry_brt_stations[fn] = combined_rail_ferry_brt_stations[fn].join(rail_freqs)
            combined_rail_ferry_brt_stations[fn]['trains_perday'] = combined_rail_ferry_brt_stations[fn]['trains_perday'].fillna(0)
            combined_rail_ferry_brt_stations[fn]['agency_name'] = combined_rail_ferry_brt_stations[fn]['agency_name'].fillna(','.join(routes['agency_name'].unique()))
            print('Found rail or ferry stations for ', fn)


    # Combine data
    combined_routes = pd.concat(combined_routes.values(), ignore_index=True)
    combined_trips = pd.concat(combined_trips.values(), ignore_index=True)
    combined_stop_times = pd.concat(combined_stop_times.values(), ignore_index=True)
    combined_stops = pd.concat(combined_stops.values(), ignore_index=True).drop_duplicates()  # e.g. MTA files duplicate stops
    if len(combined_rail_ferry_brt_stations)>0:
        combined_rail_ferry_brt_stations = pd.concat(combined_rail_ferry_brt_stations.values(), ignore_index=True)
    else:
        print('No rail, ferry, or BRT found in feed. Check your data if this is unexpected')
        combined_rail_ferry_brt_stations = pd.DataFrame(columns=['route_type', 'stop_name', 'geometry', 'GTFS_filename'])

    # Merge frequencies into trips if available
    # verified that headway_secs is almost never used, and only on low-frequency routes
    # so we ignore this complication    
    if len(combined_frequencies)>0:
        combined_frequencies = pd.concat(combined_frequencies.values(), ignore_index=True)
        combined_trips = combined_trips.merge(
            combined_frequencies[['prefixed_trip_id', 'headway_secs']],
            on='prefixed_trip_id', how='left'
        )
    
    # generate stop id from geometry, in case different agencies stop at the same stop
    combined_stops['consolidated_stop_id'] = combined_stops.groupby(['stop_lat','stop_lon']).ngroup()
    n_multiagency = combined_stops.groupby('consolidated_stop_id').size()
    n_multiagency = n_multiagency[n_multiagency>1]
    print('{} stops found that are used by >1 agency'.format(len(n_multiagency)))

    # now use this consolidated_stop_id
    stop_lookup = combined_stops.set_index('prefixed_stop_id').consolidated_stop_id.to_dict()
    combined_stops['new_stop_id'] =  combined_stops.consolidated_stop_id
    combined_stop_times['new_stop_id'] = combined_stop_times.prefixed_stop_id.map(stop_lookup)
    combined_stops.drop_duplicates(subset='new_stop_id', inplace=True)

    # create a route name - some agencies have this in short_name, some in long_name
    combined_routes['route_name'] = combined_routes.route_short_name.fillna(combined_routes.route_long_name).fillna(combined_routes.route_id)

    # Add stops for specific BRT lines
    if include_brt:
        brt_list = pd.read_csv(os.path.join(base_path,'other_transit','brt_routes.csv'))
        brt_list = brt_list[brt_list.year.astype(str)==year]
        brt_list.set_index(['agency_name','route_name'], inplace=True)
        
        # drop BRT routes that are consolidated, manually (e.g. MTS 201/202 are CW/CCW routes that are merged above)
        if not ('MTS', '202') in combined_routes.set_index(['agency_name','route_name']).index:
            brt_list.drop(('MTS', '202'), errors='ignore', inplace=True)
        try:
            brt = combined_routes.set_index(['agency_name','route_name']).loc[brt_list.index][['prefixed_route_id','route_type']].reset_index().set_index('prefixed_route_id')
        except KeyError:
            print('WARNING!!! BRT routes missing from GTFS feed. Check your brt_routes.csv file')
            brt_list = brt_list[brt_list.index.isin(combined_routes.set_index(['agency_name','route_name']).index)]
            brt = combined_routes.set_index(['agency_name','route_name']).loc[brt_list.index][['prefixed_route_id','route_type']].reset_index().set_index('prefixed_route_id')
        
        brt = brt.join(combined_trips.set_index('prefixed_route_id')['prefixed_trip_id'], how='left').reset_index().set_index('prefixed_trip_id')
        brt = brt.join(combined_stop_times.set_index('prefixed_trip_id')['new_stop_id'], how='left').set_index('new_stop_id')
        brt = brt.join(combined_stops.set_index('new_stop_id')[['stop_name','stop_lat', 'stop_lon', 'GTFS_filename']], how='left').drop_duplicates()
        brt['stop_name'] = 'BRT '+brt.stop_name

        if not (brt.route_type==3).all(): # all should be bus
            raise Exception('BRT list includes non-bus routes')
        brt = gpd.GeoDataFrame(brt, geometry=gpd.points_from_xy(brt['stop_lon'], brt['stop_lat']), crs='EPSG:4326')
        combined_rail_ferry_brt_stations = pd.concat([combined_rail_ferry_brt_stations, brt[['route_type', 'agency_name','stop_name','geometry','GTFS_filename']]])
    else:
        print('Skipping attempt to import brt_routes.csv')

    return GTFSFeed(
        routes=combined_routes,
        trips=combined_trips,
        stop_times=combined_stop_times,
        stops=combined_stops,
        frequencies=combined_frequencies,
        rail_ferry_brt_stations=combined_rail_ferry_brt_stations        
    )

def rail_ferry_brt(feed, year, mode='maximal', include_planned=False):
    """Step 2
    Identifies rail, ferry, and BRT stops from GTFS data.
    This step identifies transit stops that serve rail, light rail, subway, ferry, or some BRT routes. 
    These routes-except for BRT-have their own designation
    in GTFS files and all count for HQ transit stops. BRT routes are selected by name.

    Then, other geospatial data (Amtrak, other BRT lines, station parcels and points) are merged in too.

    mode can be:
    - minimal
    - maximal
    - a list of the incremental changes between minimal and maximal

    """
    # Define route types based on mode
    assert mode in ['maximal', 'minimal'] or isinstance(mode, list)
    if isinstance(mode, list):
        assert all([mm in stops_increments for mm in mode])

    routes_to_include = [item for sublist in gtfs_route_types.values() for item in sublist]
    if mode=='minimal' or (isinstance(mode, list) and 'railBRT_included' not in mode):
        routes_to_include = [rr for rr in routes_to_include if rr not in minimal_route_types_excluded]

    # Filter routes for rail and ferry
    rail_ferry_brt_gdf = (feed.rail_ferry_brt_stations[feed.rail_ferry_brt_stations['route_type'].astype(int).isin(routes_to_include)]).copy()
    rail_ferry_brt_gdf['qualify'] = rail_ferry_brt_gdf.route_type.map(gtfs_mode_names) + '_gtfs'

    # add BRT from shapefile
    brt_fn = os.path.join(base_path,'other_transit','statewide_BRT_'+year+'.zip') 
    brt_gdf = gpd.read_file(brt_fn)
    brt_gdf['tmp_idx'] = brt_gdf.index.values # to fill in NaNs in stop_id
    brt_gdf.stop_id = brt_gdf.stop_id.fillna(brt_gdf['tmp_idx'])
    brt_gdf['new_stop_id'] = 'brt_'+brt_gdf.agency_pri+'_'+brt_gdf.stop_id.astype(str)
    brt_gdf.rename(columns={'agency_pri':'agency_name'}, inplace=True)
    brt_gdf.drop_duplicates(subset='geometry', inplace=True)  # some Omnitrans routes are duplicates
    brt_gdf.to_crs('EPSG:4326', inplace=True)
    brt_gdf = brt_gdf[['new_stop_id','agency_name','geometry']]    
    brt_gdf['qualify'] = 'brt_shapefile'
    rail_ferry_brt_gdf = pd.concat([rail_ferry_brt_gdf, brt_gdf])

    fns = []
    # add HSR, Amtrak, subway entrances and station parcels
    if mode=='maximal' or 'railBRT_included' in mode:
        fns += [('amtrak.zip','amtrak')]
    if mode=='maximal' or 'subway_entrances' in mode:
        fns += [('subway_entrances_'+year+'.zip','subway_entrance')]
    if mode=='maximal' or 'parcels' in mode:
        fns += [('parcels_combined_'+year+'.zip', 'station_parcel')]
    if include_planned or 'planned_transit' in mode:
        assert year=='2025'
        fns+=[('CAHSR.zip','hsr'),
                ('MPO_planned/2020_MTP_SCS_Planned_Major_Transit_Stops_SACOG.shp','SACOG'),
                ('MPO_planned/Existing_Major_Transit_Stops_SACOG.shp','SACOG'),
                ('MPO_planned/Major_Transit_Stops_in_the_SCAG_Region_for_plan_year_2050.shp','SCAG'),
                ('MPO_planned/TPA_stops_2035build_SANDAG.shp','SANDAG'),
                ('MPO_planned/transitstops_2021_existing_planned_MTC.shp','MTC'),]

    for fn, prefix in fns:
        rail_gdf = gpd.read_file(os.path.join(base_path, 'other_transit', fn))
        if rail_gdf.crs is None:
            raise Exception('CRS missing from ', fn)
        rail_gdf.to_crs('EPSG:4326', inplace=True)
        if 'amtrak' in fn:
            rail_gdf = rail_gdf[(rail_gdf.StnType=='TRAIN') & (rail_gdf.State=='CA') & (rail_gdf.StaType!='Curbside Bus Stop only (no shelter)')]
        if 'CAHSR' in fn:
            rail_gdf['new_stop_id'] = prefix+'_'+rail_gdf.Stat_Name
        else:
            rail_gdf['new_stop_id'] = prefix+'_'+rail_gdf.index.astype(str)
        rail_gdf = rail_gdf[['new_stop_id','geometry']]
        rail_gdf['qualify'] = prefix
        
        rail_ferry_brt_gdf = pd.concat([rail_ferry_brt_gdf, rail_gdf])

    print(f"Extracted {len(rail_ferry_brt_gdf)} rail, ferry, and BRT stops")
    return rail_ferry_brt_gdf

def bus_stops_peak_hours(feed, mode='maximal', frequency=20, save_routes=False):
    """Step 3
    
    Identifies bus stops with frequent service during peak hours (morning and afternoon). 
    
    In the maximal definition, a stop qualifies if it has, from one route, at least one hour-long period with 3 arrivals throughout the morning and afternoon. 
    This hour-long period need not start at the beginning of an hour; for example, it could stretch from 7:30-8:30AM. 
    Mornings and afternoons are defined from midnight to noon and noon to midnight.
    
    In the minimal definition, a stop qualifies if it has, from one route, at least 9 buses arriving during the morning peak AND 12 buses in the afternoon peak. 
    The morning peak is 6-9AM and the afternoon peak is 3-7PM.
    """
    
    # Only select bus routes
    assert mode in ['maximal', 'minimal'] or isinstance(mode, list)
    if isinstance(mode, list):
        assert all([mm in peaks_increments for mm in mode])
    assert isinstance(frequency, int) # buses per hour


    bus_routes = feed.routes[feed.routes['route_type'].astype(int).isin([3,11])]  # Bus route_type = 3. Trolleybus is 11
    bus_trips = feed.trips[feed.trips['prefixed_route_id'].isin(bus_routes['prefixed_route_id'])]
    
    # Enrich stop_times with route information
    stop_times = feed.stop_times.merge(
        bus_trips[['prefixed_trip_id', 'prefixed_route_id']],
        on='prefixed_trip_id', how='inner'
    )
   
    # Convert arrival times to datetime. Use the modulus in case arrival times are after midnight
    secs_in_day = 24*60*60
    stop_times['time'] = pd.to_datetime(stop_times['arrival_time']%secs_in_day, unit='s')
    assert stop_times['time'].max()<=pd.to_datetime('1970-01-01 23:59:59')  # make sure within the 24 hr interval
    print(f'Identifying frequent service at {len(stop_times)} stops in {mode} mode at {frequency} min frequency')
    
    # Define peak periods based on mode
    if mode == 'minimal' or (isinstance(mode, list) and 'peak_definition' not in mode):
        print('minimal peak period')
        am_start = pd.to_datetime('1970-01-01 06:00:00')
        am_end = pd.to_datetime('1970-01-01 09:00:00')
        pm_start = pd.to_datetime('1970-01-01 15:00:00')
        pm_end = pd.to_datetime('1970-01-01 19:00:00')
        
        # Filter for AM and PM peak hours
        am_peak = stop_times[(stop_times['time'] >= am_start) & (stop_times['time'] < am_end)]
        pm_peak = stop_times[(stop_times['time'] >= pm_start) & (stop_times['time'] < pm_end)]
        
        # Count trips per stop per route during peaks
        am_counts = am_peak.groupby(['new_stop_id', 'prefixed_route_id']).size().reset_index(name='am_count')
        pm_counts = pm_peak.groupby(['new_stop_id', 'prefixed_route_id']).size().reset_index(name='pm_count')
        
        # Join the counts
        peak_counts = am_counts.merge(pm_counts, on=['new_stop_id', 'prefixed_route_id'], how='outer').fillna(0)
        
        # Filter stops with at least 9 AM trips AND 12 PM trips on any route - storing route information
        # That's for the default of 20 min headways
        am_trips = math.floor(3*60/frequency)
        pm_trips = math.floor(4*60/frequency)
        print(f'Looking for {am_trips} AM and {pm_trips} PM trips')

        qualifying_stops = peak_counts[(peak_counts['am_count'] >= am_trips) & (peak_counts['pm_count'] >= pm_trips)][['new_stop_id','prefixed_route_id','am_count','pm_count']].drop_duplicates()
        qualifying_stops.set_index('new_stop_id', inplace=True)
        qualifying_stops['am_freq'] = qualifying_stops.am_count / (am_end.hour-am_start.hour) 
        qualifying_stops['pm_freq'] = qualifying_stops.pm_count / (pm_end.hour-pm_start.hour) 
    
        if 'consolidate_infrequent' in mode:
            # Can we combine any of these infrequent (non-qualifying) routes?
            # we do the same as above, but just group by stop_id (not prefixed_route_id)
            infrequent_am_peak = am_peak.set_index(['new_stop_id','prefixed_route_id']).drop(qualifying_stops.set_index('prefixed_route_id', append=True).index)  
            infrequent_pm_peak = pm_peak.set_index(['new_stop_id','prefixed_route_id']).drop(qualifying_stops.set_index('prefixed_route_id', append=True).index) 

            # drop routes that only run in AM or peak
            infrequent_am_peak = infrequent_am_peak[infrequent_am_peak.index.isin(infrequent_pm_peak.index)]
            infrequent_pm_peak = infrequent_pm_peak[infrequent_pm_peak.index.isin(infrequent_am_peak.index)]

            am_counts = infrequent_am_peak.groupby(['new_stop_id']).size().reset_index(name='am_count')
            pm_counts = infrequent_pm_peak.groupby(['new_stop_id']).size().reset_index(name='pm_count')
            peak_counts = am_counts.merge(pm_counts, on=['new_stop_id'], how='outer').fillna(0)
        
            infrequent_qualifying_stops = peak_counts[(peak_counts['am_count'] >= am_trips) & (peak_counts['pm_count'] >= pm_trips)][['new_stop_id','am_count','pm_count']].drop_duplicates()
            infrequent_qualifying_stops.set_index('new_stop_id', inplace=True)
            infrequent_qualifying_stops['prefixed_route_id'] = 'Consolidated'
            infrequent_qualifying_stops['am_freq'] = infrequent_qualifying_stops.am_count / (am_end.hour-am_start.hour) 
            infrequent_qualifying_stops['pm_freq'] = infrequent_qualifying_stops.pm_count / (pm_end.hour-pm_start.hour) 
    
            qualifying_stops = pd.concat([qualifying_stops,infrequent_qualifying_stops])

    
    else:  # maximal mode
        assert mode=='maximal' or 'peak_definition' in mode
        assert frequency==20 # different frequencies not implemented for maximal
        # Check for 3+ trips per hour in both AM and PM periods
        am_start = pd.to_datetime('1970-01-01 00:00:00')
        am_end = pd.to_datetime('1970-01-01 11:59:59')  # Fixed from 24:00:00
        pm_start = pd.to_datetime('1970-01-01 12:00:00')
        pm_end = pd.to_datetime('1970-01-01 23:59:59')  # Fixed from 24:00:00
        
        # Filter for AM and PM periods
        am_peak = stop_times[(stop_times['time'] >= am_start) & (stop_times['time'] < am_end)]
        pm_peak = stop_times[(stop_times['time'] >= pm_start) & (stop_times['time'] < pm_end)]
        
        # Check for rolling hour windows with 3+ trips
        qualifying_stops_am = set()
        qualifying_stops_pm = set()

        non_qualifying_routes_am = [] # we will see if these can be combined at any point to form other routes
        non_qualifying_routes_pm = []

        qualifying_stops_all = []  
        qualifying_freqs_all = []      

        for period_start, period_end in ((am_start, am_end), (pm_start, pm_end)):
            # subset of trips for am or pm 
            peak = stop_times[(stop_times['time'] >= period_start) & (stop_times['time'] <= period_end)]
            qualifying_stops = set()
            qualifying_freqs = {}
            infrequent_stop_routes = []

            for stop_route, group in peak.groupby(['new_stop_id', 'prefixed_route_id']):
                times = sorted(group['time'])
                max_count = pd.DataFrame(np.zeros(len(times)),index=times).rolling(window='1h').count().max().max()
                # more efficient to use pd.rolling as above
                #max_count = 0
                #for i, start_time in enumerate(times):
                #    end_time = start_time + pd.Timedelta(hours=1)
                #    max_count = max(max_count, sum((pd.Series(times) >= start_time) & (pd.Series(times) < end_time)))
                if max_count >= 3:
                    qualifying_stops.add(stop_route)  # Add stop_id and route_id
                    qualifying_freqs[stop_route] = int(max_count)
                else:  # only executed if the inner for loop does not break
                    infrequent_stop_routes.append(group)            

            if mode=='maximal' or (isinstance(mode, list) and 'consolidate_infrequent' in mode):
                # Can we combine any of these infrequent (non-qualifying) routes?
                infrequent_stop_routes = pd.concat(infrequent_stop_routes) if len(infrequent_stop_routes)>0 else pd.DataFrame(columns=peak.columns)
                for stop_id, group in infrequent_stop_routes.groupby('new_stop_id'):
                    times = sorted(group['time'])
                    max_newroutes = 0
                    for i, start_time in enumerate(times):
                        end_time = start_time + pd.Timedelta(hours=1)
                        count = sum((pd.Series(times) >= start_time) & (pd.Series(times) < end_time))
                        n_routes = len(np.unique(group.loc[(group['time'] >= start_time) & (group['time'] < end_time), 'prefixed_route_id']))
                        if n_routes>=4 and count>=6:
                            max_newroutes = 2  # there could be more, but it's a major stop if there are at least 2
                        elif n_routes>=2 and count>=3:
                            max_newroutes = max(1, max_newroutes)
                    if max_newroutes>0:
                        qualifying_stops.add((stop_id, 'Consolidated ['+', '.join(group.prefixed_route_id.unique())+']'))
                        #qualifying_freqs[(stop_id,'Consolidated Routes')] = 'consolidated'
                    if max_newroutes>1:
                        # note: in the rare event that there is more than one consolidated route, we don't distinguish
                        qualifying_stops.add((stop_id, 'Consolidated 2 ['+', '.join(group.prefixed_route_id.unique())+']'))
                        #qualifying_freqs[(stop_id,'Consolidated Routes 2')] = 'consolidated'

            qualifying_stops_all.append(qualifying_stops) # save the AM results before we do PM in the next iteration of the loop
            qualifying_freqs_all.append(qualifying_freqs)

        # A stop must qualify during both AM and PM periods
        qualifying_stops = pd.DataFrame(qualifying_stops_all[0].intersection(qualifying_stops_all[1]), columns=['new_stop_id', 'prefixed_route_id']).set_index(['new_stop_id','prefixed_route_id'])
        # add frequencies
        qualifying_stops['am_freq'] = pd.Series(qualifying_freqs_all[0])
        qualifying_stops['pm_freq'] = pd.Series(qualifying_freqs_all[1])
        qualifying_stops.reset_index(level=1, inplace=True)

    # Get stop details for qualifying stops and add agency name
    bus_peak_stops = qualifying_stops.join(feed.stops.set_index('new_stop_id')[['stop_lon','stop_lat','GTFS_filename']]).reset_index()
    bus_peak_stops = (bus_peak_stops.set_index('prefixed_route_id').join(bus_routes.set_index('prefixed_route_id')[['agency_name']])).reset_index().set_index('new_stop_id')

    # Create GeoDataFrame
    bus_peak_gdf = gpd.GeoDataFrame(
        bus_peak_stops,
        geometry=gpd.points_from_xy(bus_peak_stops['stop_lon'], bus_peak_stops['stop_lat']),
        crs="EPSG:4326"
    )

    if save_routes: # for QA/QC
        hifreqroutes = bus_peak_gdf.reset_index()[['prefixed_route_id','GTFS_filename']].drop_duplicates()
        hifreqroutes.sort_values(by=list(hifreqroutes.columns), inplace=True)
        hifreqroutes.to_csv(os.path.join(output_path,'hifreqroutes_'+mode+'.csv'))

    print(f"Identified {len(bus_peak_gdf)} high-frequency bus stops")
    return bus_peak_gdf

def identify_bus_stop_intersections(feed, bus_peak_gdf, mode='maximal', save_stops=False):
    """Step 4

    This step finds bus stops that are near each other (within 150 or 500 feet) but serve different bus routes. This prevents stops on the same route
    from intersecting, which Cal-ITP believed was against the intent of the law.

    For minimal definition:
    - Intersecting routes must stop at different stops, in order to avoid combining parallel routes
    
    For maximal definition:
    - All stops along a route with any intersection count
    - Routes along same road count as intersecting

    Parameters:
    - feed: GTFS feed containing routes, trips, and stop_times
    - bus_peak_gdf: GeoDataFrame with bus stops meeting peak hour criteria
    - mode: 'minimal' or 'maximal' mode
    
    Returns:
    - GeoDataFrame with qualifying intersecting bus stops
    """
    print(f"Finding bus stop intersections in {mode} mode")
    assert mode in ['maximal', 'minimal'] or isinstance(mode, list)
    assert bus_peak_gdf.index.name == 'new_stop_id'
    
    # Set distance threshold based on mode and convert from feet to meters for UTM projection
    distance_threshold_feet = 500 if mode == 'maximal' or 'stop_distance' in mode else 150  # feet
    print('stop distance threshold:', distance_threshold_feet )
    distance_threshold_meters = distance_threshold_feet * 0.3048  # Conversion to meters
    
    # Project to UTM for accurate distances
    bus_peak_gdf = bus_peak_gdf.to_crs(epsg=32611)  
    
    # Add plain English route names
    routenames = feed.routes[['prefixed_route_id','agency_name','route_name']].set_index('prefixed_route_id')
    routenames['agency_route'] = routenames.agency_name.fillna('') + ' ' + routenames.route_name
    routenames = routenames['agency_route'].to_dict()
    
    # Get unique stops for initial spatial query (to avoid duplicates)
    unique_stops = bus_peak_gdf[~bus_peak_gdf.index.duplicated()]

    # frequencies - with multiindex for easier accesing
    bus_freqs = bus_peak_gdf.set_index('prefixed_route_id', append=True)[['am_freq','pm_freq']]
    
    # Find stops within distance threshold of each other
    intersecting_stops = set()
    intersecting_stop_routes = {}
    intersecting_stop_am_freqs = {}
    intersecting_stop_pm_freqs = {}
        
    sindex = unique_stops.sindex
    sindex_map = unique_stops.reset_index()['new_stop_id'].to_dict()

    for stop_id1, stop1 in unique_stops.iterrows():
        # Get all route ids at this stop
        routes_at_stop1 = set(bus_peak_gdf.loc[[stop_id1], 'prefixed_route_id'])
        
        # Find all stops within threshold distance
        nearby_stop_ids = sindex.query(stop1.geometry, predicate='dwithin', distance=distance_threshold_meters)
        nearby_stop_ids = [sindex_map[xx] for xx in nearby_stop_ids]
        nearby = bus_peak_gdf.loc[nearby_stop_ids]
        
        # Exclude self
        nearby = nearby[nearby.index != stop_id1]

        # Ensure that consolidated routes can't count at both stop1 and nearby stop
        if all([rr.startswith('Consolidated') for rr in routes_at_stop1]) and all([rr.startswith('Consolidated') for rr in nearby.prefixed_route_id.values]): 
            consolidated_routes = []
            for rr in routes_at_stop1:
                consolidated_routes.append(rr.replace('Consolidated [','').replace(']','').split(', '))
            # which routes are a part of ALL routes at that stop?
            consolidated_routes = [rr for rr in consolidated_routes[0] if all(rr in sublist for sublist in consolidated_routes)]
            if consolidated_routes:
                # drop those from nearby
                for rr in consolidated_routes:
                    routes_to_drop = nearby.prefixed_route_id.str.contains(rr)
                    if any(routes_to_drop):
                        #print(f'Dropping consolidated route {consolidated_routes} at {stop_id1} from nearby stops')
                        nearby = nearby[~nearby.prefixed_route_id.str.contains(rr)]
                # special case - if there are no nearby routes left at this point, we don't proceed, even for maximal
                # note that at this point, all the routes at stop1 are consolidated and include the same route
                if len(nearby) == 0: 
                    continue
        
        if len(nearby) == 0 and (mode=='minimal' or (isinstance(mode, list) and 'intersecting_stops' not in mode)): # no nearby routes
            continue
            
        # Get routes at nearby stops
        routes_at_nearby_stops = set(bus_peak_gdf.loc[nearby.index, 'prefixed_route_id'])
        
        # Get all routes (at this stop and nearby)
        all_routes = routes_at_stop1.union(routes_at_nearby_stops)
        
        # Add plain English route names and frequencies
        routes_at_stop1_english = [routenames.get(rr, str(rr)) for rr in routes_at_stop1]
        routes_at_nearby_stops_english = [routenames.get(rr, str(rr)) for rr in routes_at_nearby_stops.difference(routes_at_stop1)]
        all_routes_english = 'At stop: '+', '.join(routes_at_stop1_english) + '. Nearby: '+', '.join(routes_at_nearby_stops_english)
        am_freqs = ','.join([f"{bus_freqs.loc[(stop_id1,rr),'am_freq']:.2g}" for rr in routes_at_stop1])
        pm_freqs = ','.join([f"{bus_freqs.loc[(stop_id1,rr),'pm_freq']:.2g}" for rr in routes_at_stop1])

        # Implement minimal vs maximal logic
        if mode == 'maximal' or 'intersecting_stops' in mode:
            # If the stop and nearby stops have >1 route, it qualifies!
            if len(all_routes) > 1:
                intersecting_stops.add(stop_id1)
                intersecting_stop_routes[stop_id1] = all_routes_english
                intersecting_stop_am_freqs[stop_id1] = am_freqs
                intersecting_stop_pm_freqs[stop_id1] = pm_freqs
        else:
            # For minimal: exclude routes that stop at the same stop
            # Only count if there are routes at nearby stops that aren't at this stop
            if len(routes_at_nearby_stops.difference(routes_at_stop1)) > 0:
                intersecting_stops.add(stop_id1)
                intersecting_stop_routes[stop_id1] = all_routes_english
                intersecting_stop_am_freqs[stop_id1] = am_freqs
                intersecting_stop_pm_freqs[stop_id1] = pm_freqs

    result = unique_stops.loc[list(intersecting_stops)][['geometry']]
    joinDf = pd.DataFrame.from_dict([intersecting_stop_routes, intersecting_stop_am_freqs,intersecting_stop_pm_freqs]).T
    joinDf.columns = ['prefixed_route_ids','am_freq','pm_freq']
    result = result.join(joinDf)
    result = result.to_crs(epsg=4326)  # Convert back to WGS84

    # Add agency
    assert result.index.name == 'new_stop_id' and bus_peak_gdf.index.name == 'new_stop_id'
    assert all(bus_peak_gdf[pd.isnull(bus_peak_gdf.agency_name)].prefixed_route_id.str.contains('Consolidated'))
    agency_names = bus_peak_gdf.reset_index()[['new_stop_id','agency_name']].drop_duplicates()
    agency_names.agency_name = agency_names.agency_name.fillna('Consolidated')
    agency_names = agency_names.groupby('new_stop_id').agency_name.agg(', '.join)
    result = result.join(agency_names)

    # Add qualify field
    result['qualify'] = f'Bus stops within {distance_threshold_feet} feet'

    if save_stops: # for QA/QC
        #saved_intersections = result.reset_index()[['new_stop_id']].drop_duplicates()
        saved_intersections = feed.stops.set_index('new_stop_id').loc[result.index]['GTFS_filename'].drop_duplicates()
        saved_intersections.sort_values(inplace=True)
        saved_intersections.to_csv(os.path.join(output_path,'majorstops_'+mode+'.csv'))
    
    print(f"Found {len(result)} qualifying bus stops with intersections")
    return result.reset_index(), len(result)

def merge_transit_stops(rail_ferry_brt_gdf, bus_stops_gdf, year, fn_suffix='maximal', include_planned=False, output_path=output_path):
    """Step #5
    
    Merge datasets

    This step combines the rail/ferry/BRT stops and high-quality bus stops (from steps 2 and 4) into one dataset. 
    It then exports this dataset as a GPKG of the stops (without 1/2 mile buffers) to a designated file path.
    """

    # Ensure CRS is consistent
    assert rail_ferry_brt_gdf.crs=='EPSG:4326'
    assert bus_stops_gdf.crs=='EPSG:4326'
    assert bus_stops_gdf.new_stop_id.is_unique
    if bus_stops_gdf is not None and bus_stops_gdf.empty:
        print('Warning: no intersecting bus stops found')

    combined_gdf = pd.concat([rail_ferry_brt_gdf, bus_stops_gdf])
    combined_gdf = combined_gdf[['new_stop_id', 'qualify', 'agency_name', 'stop_name', 'prefixed_route_ids', 
                                 'am_freq', 'pm_freq', 'trains_perday', 'geometry']]

    # Rename columns for shapefile compatibility if needed
    combined_gdf.rename(columns={
        'new_stop_id': 'stop_id',
        'prefixed_route_ids':'route_ids'
    }, inplace=True)
    
    # Save both GeoPackage and Shapefile with fn_suffix in filename
    suffix = '_with_planned' if include_planned else ''
    modename = '_'.join(fn_suffix) if isinstance(fn_suffix, list) else fn_suffix
    combined_gdf.to_file(os.path.join(output_path, f"high_quality_stops_{modename}_{year}{suffix}.gpkg"), driver="GPKG", index=False)
    # Can't do this as geodataframe contains both points and polygons
    #combined_gdf.to_file(os.path.join(output_path, f"high_quality_stops_{modename}_{year}.shp"), driver="ESRI Shapefile", index=False)
    
    print(f"Saved merged stops to {output_path}")
    return combined_gdf

def buffer_transit_stops(high_quality_stops_gdf, year, buffer_distance_miles=0.5, fn_suffix='maximal', include_planned=False, output_path=output_path):
    """Step 6
    Creates a buffer around transit stops.
    These are the high-quality transit areas resulting from the stops
    """

    # Convert buffer distance to meters
    buffer_distance_meters = buffer_distance_miles * 1609.34
    
    # Project to UTM for accurate buffering
    projected_gdf = high_quality_stops_gdf.to_crs(epsg=32611)
    
    # Apply buffer directly to geometry (simplified approach)
    projected_gdf.geometry = projected_gdf.geometry.buffer(buffer_distance_meters)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save both GeoPackage and Shapefile with fn_suffix in filename
    suffix = '_with_planned' if include_planned else ''
    modename = '_'.join(fn_suffix) if isinstance(fn_suffix, list) else fn_suffix
    projected_gdf.to_file(os.path.join(output_path, f"buffered_stops_{modename}_{year}{suffix}.gpkg"), driver="GPKG", index=False)
    #projected_gdf.to_file(os.path.join(output_path, f"buffered_stops_{modename}_{year}{suffix}.shp"), driver="ESRI Shapefile", index=False)
    print(f"Saved buffered stops to {output_path}")
    
    # Save dissolved version 
    dissolved = projected_gdf.dissolve().explode()[['geometry']]
    dissolved.to_file(os.path.join(output_path, f"dissolved_stops_{modename}_{year}{suffix}.gpkg"), driver="GPKG")

    return projected_gdf

def run_transit_zoning_pipeline(gtfs_path=gtfs_path, output_path=output_path, year=None):
    """Runs the complete pipeline for both minimal and maximal definitions.
    If year is None, this function is called recursively for all years"""
    print(f"Output will be saved to: {output_path}")
    if not(os.path.exists(output_path)):
        os.mkdir(output_path)
        print('Created output directory ', output_path)
    if year is None:
        for year in yearlist:
            run_transit_zoning_pipeline(gtfs_path, output_path, year=year)
        return
    else:
        assert year in yearlist # other years are OK - this is a sanity check for development

    print("Step 1: Loading GTFS data for year", year)
    feed = load_and_combine_gtfs(gtfs_path, year)
    
    for mode in ['minimal','maximal']:
        print(f"Processing {mode} definition")
        rail_ferry_brt_stops = rail_ferry_brt(feed, year, mode=mode)
        bus_peaks = bus_stops_peak_hours(feed, mode=mode, save_routes=(testing and year=='2025'))
        bus_intersections, _ = identify_bus_stop_intersections(feed, bus_peaks, mode=mode, save_stops=(testing and year=='2025'))

        merged = merge_transit_stops(rail_ferry_brt_stops, bus_intersections, year, fn_suffix=mode, output_path=output_path)
        buffered = buffer_transit_stops(merged, year, fn_suffix=mode, output_path=output_path)

        if year=='2025' and mode=='maximal':
            # run a second time with the planned transit too
            rail_ferry_brt_stops = rail_ferry_brt(feed, year, mode=mode, include_planned=True)
            merged = merge_transit_stops(rail_ferry_brt_stops, bus_intersections, year, fn_suffix=mode, include_planned=True, output_path=output_path)
            buffered = buffer_transit_stops(merged, year, fn_suffix=mode, include_planned=True, output_path=output_path)

    print("Transit Zoning Pipeline completed successfully")

def identify_paired_routes():
    """Identify routes that we want to treat as a single route, 
      e.g. because 1A/1B/1C (counterclockwise/clockwise are handled automatically)

      This is to avoid having an "intersection" between functionally the same route running in the opposite direction
      
      Exports a csv file for manual review"""
 
    output_dfs = {}
    for year in ['2014','2020','2025']:
        feed= load_and_combine_gtfs(gtfs_path, year)
        routes = feed.routes[feed.routes['route_type'].astype(int).isin([3,11])] 
        wise_routes = routes[routes.route_name.str.lower().str.contains('wise')]
        letter_routes = routes[routes.route_short_name.astype(str).str.contains(r'[a-zA-Z]')]

        cols = ['agency_name','route_id','route_short_name','route_long_name']
        paired_routes = pd.concat([wise_routes,letter_routes])[cols].drop_duplicates().sort_values(by=cols)
        paired_routes['year'] = year
        output_dfs[year] = paired_routes.copy()
    pd.concat(output_dfs.values()).to_csv(os.path.join(output_path, 'paired_routes.csv'), index=False)

def export_all_routes():
    """Used to manually identify potential duplicate GTFS feeds"""
    yearlist = ['2014','2020','2025']
    dfs = {}
    for year in yearlist:
        dfs[year] = load_and_combine_gtfs(gtfs_path, year).routes
        dfs[year]['year'] = year
    df = pd.concat(dfs.values())
    cols = ['year','agency_name','route_long_name', 'route_short_name','route_id','GTFS_filename']
    df.sort_values(by=cols, inplace=True)
    df[cols].to_csv(os.path.join(output_path, 'all_routes.csv'), index=False)

def run_incremental_changes(gtfs_path=gtfs_path, year='2025'):
    """runs incremental changes from maximal to minimal
        stores the area under each
    Note: there are some edge cases, with stops that are included under minimal but not later stages
    For example: a more generous peak-hour definition includes some routes at a stop
    Then, that route isn't available to be "near" a second route at that stop.  
    """
    
    feed = load_and_combine_gtfs(gtfs_path, year)
    results_individual = {}
    results_cumulative = {}

    for increment in all_increments:
        # run with just that increment
        if increment in stops_increments:
            rail_ferry_brt_stops = rail_ferry_brt(feed, year, mode=[increment])
        else:
            rail_ferry_brt_stops = rail_ferry_brt(feed, year, mode='minimal')

        if increment in peaks_increments:
            bus_peaks = bus_stops_peak_hours(feed, mode=[increment])
        else:
            bus_peaks = bus_stops_peak_hours(feed, mode='minimal')

        if increment in intersections_increments:
            bus_intersections, _ = identify_bus_stop_intersections(feed, bus_peaks, mode=[increment])
        else:
            bus_intersections, _ = identify_bus_stop_intersections(feed, bus_peaks, mode='minimal')

        merged = merge_transit_stops(rail_ferry_brt_stops, bus_intersections, year, fn_suffix=increment, output_path=incremental_output_path)
        buffered = buffer_transit_stops(merged, year, fn_suffix=increment, output_path=incremental_output_path)
        # calculate area under Albers and store
        results_individual[increment] = float((buffered.to_crs('ESRI:102008').dissolve().area / 1000 / 1000).iloc[0])

        # now, do a cumulative version
        cumulative_increments = all_increments[0:all_increments.index(increment)+1]

        cumulative_stops_increments = list(set(cumulative_increments).intersection(stops_increments))
        if len(cumulative_stops_increments)>0:
            rail_ferry_brt_stops = rail_ferry_brt(feed, year, mode=cumulative_stops_increments)
        else:
            rail_ferry_brt_stops = rail_ferry_brt(feed, year, mode='minimal')

        cumulative_peaks_increments = list(set(cumulative_increments).intersection(peaks_increments))
        if len(cumulative_peaks_increments)>0:
            bus_peaks = bus_stops_peak_hours(feed, mode=cumulative_peaks_increments)
        else:
            bus_peaks = bus_stops_peak_hours(feed, mode='minimal')
        
        cumulative_intersections_increments = list(set(cumulative_increments).intersection(intersections_increments))
        if len(cumulative_intersections_increments)>0:
            bus_intersections, _ = identify_bus_stop_intersections(feed, bus_peaks, mode=cumulative_intersections_increments)
        else:
            bus_intersections, _ = identify_bus_stop_intersections(feed, bus_peaks, mode='minimal')

        merged = merge_transit_stops(rail_ferry_brt_stops, bus_intersections, year, fn_suffix='cum_'+increment, output_path=incremental_output_path)
        buffered = buffer_transit_stops(merged, year, fn_suffix='cum_'+increment, output_path=incremental_output_path)

        # now store the area of buffered
        results_cumulative[increment] = float((buffered.to_crs('ESRI:102008').dissolve().area / 1000 / 1000).iloc[0])

    # save to disk
    results_cumulative = pd.DataFrame.from_dict(results_cumulative, orient='index',columns=['area_km2'])
    results_cumulative['cumulative'] = True
    results_individual = pd.DataFrame.from_dict(results_individual, orient='index',columns=['area_km2'])
    results_individual['cumulative'] = False
    result = pd.concat([results_individual,results_cumulative ])
    result.index.name = 'increment'
    result.to_csv(incremental_output_path+'/incremental_area.csv')

    increments_figure()

def increments_figure():
    """Figure for journal article of the individual and cumulative contribution of each increment"""

    label_dict = {'minimal':'baseline', 'peak_definition': 'expand peak\ndefinition',
                  'consolidate_infrequent': 'merge infrequent\nroutes',
                  'stop_distance': 'distance\nbetween stops',
                  'intersecting_stops': 'intersect at\nsame stop',
                  'railBRT_included': 'more rail and\nBRT routes',
                  'subway_entrances': 'subway\nentrances',
                  'parcels': 'stations\nas parcels',
                  'planned_transit': 'planned transit'}

    df = pd.read_csv(incremental_output_path+'/incremental_area.csv', index_col='increment')
    baseline = df[df.cumulative].loc['minimal','area_km2']
    df['pct_increase'] = (df.area_km2 - baseline) / baseline * 100

    plotdf = df[df.cumulative==False].sort_values(by='area_km2')
    plotdf = plotdf.drop('minimal')
    
    fig, ax = plt.subplots(figsize = (6,4))
    sns.barplot(plotdf.reset_index(), x = 'pct_increase', y='increment', ax=ax)

    ax.set_xlabel('increase in land area (per cent)')
    ax.set_yticklabels([label_dict[lab.get_text()] for lab in ax.get_yticklabels()], 
                       fontsize=9,
                       ha='right', )# rotation=70,  rotation_mode='anchor')
    ax.set_ylabel('')
    
    plt.tight_layout()
    fig.savefig(figure_path+'/incremental_changes.jpg', dpi=600)

    plotdf = df[df.cumulative==True].copy()
    plotdf['lower_bar'] = plotdf.area_km2.shift(1)
    plotdf.fillna({'lower_bar':0}, inplace=True)
    plotdf['bar_height'] = plotdf.area_km2 - plotdf.lower_bar

    fig, ax = plt.subplots(figsize = (6,4))
    ax.barh(y=plotdf.index, width=plotdf.bar_height, left=plotdf.lower_bar)
    for ii, lab in enumerate(plotdf.index.values):
        if ii==0: 
            ax.text(plotdf.loc[lab,'lower_bar']+100, ii, label_dict[lab], size=8, ha='left', va='center')
        else:
            ax.text(plotdf.loc[lab,'lower_bar']-100, ii, label_dict[lab], size=8, ha='right', va='center')
        
    ax.set_yticks([])
    ax.set_xlabel('land area (sq km)')

    plt.tight_layout()
    fig.savefig(figure_path+'/cumulative_changes.jpg', dpi=600)

def frequency_analysis(gtfs_path=gtfs_path, year='2025'):
    """Gradually change the number of buses in the peak period 
    to see impacts of frequency changes"""

    areas, n_intersections = {}, {}
    feed = load_and_combine_gtfs(gtfs_path, year)
    rail_ferry_brt_stops = rail_ferry_brt(feed, year, mode='minimal')
    
    # frequency is defined in terms of headways (mins between buses)
    # averaged over the AM and PM hours
    for frequency in range(1,31):
        print(f'\nAnalyzing {frequency} minute frequency')
        bus_peaks = bus_stops_peak_hours(feed, mode='minimal', frequency=frequency)
        bus_intersections, n_intersections[frequency] = identify_bus_stop_intersections(feed, bus_peaks, mode='minimal')
        merged = merge_transit_stops(rail_ferry_brt_stops, bus_intersections, year, fn_suffix=f'frequency{frequency}', output_path=incremental_output_path)
        buffered = buffer_transit_stops(merged, year, fn_suffix=f'frequency{frequency}', output_path=incremental_output_path)
        areas[frequency] = float((buffered.to_crs('ESRI:102008').dissolve().area / 1000 / 1000).iloc[0])

    areas = pd.DataFrame.from_dict(areas, orient='index',columns=['area_km2'])
    n_intersections = pd.DataFrame.from_dict(n_intersections, orient='index',columns=['n_intersections'])
    results = areas.join(n_intersections)
    results.index.name = 'frequency'

    results.to_csv(incremental_output_path+'/frequency_area.csv')

    frequency_figure()

def frequency_figure():
    """Figure for journal article on the impact of increasing frequency"""
    df = pd.read_csv(incremental_output_path+'/frequency_area.csv', index_col='frequency')

    fig, ax = plt.subplots(figsize = (6,4))
    ax2 = ax.twinx()
    sns.barplot(df, x=df.index, y='area_km2', alpha=0.8, ax=ax)

    sns.lineplot(df, x=range(len(df)), y='n_intersections', color='k', lw=2, ax=ax2 )

    
    ax.set_xticks(range(4,30,5)) # offset by 1
    ax.set_xlabel('peak headway (minutes)')
    ax.set_ylabel('land area (sq km)')
    ax2.set_ylabel('number of bus stops', rotation=270, labelpad=15)
    ax.set_xlim(-0.5,29.5)


    plt.tight_layout()
    fig.savefig(figure_path+'/frequency_changes.jpg', dpi=600)



if __name__ == "__main__":

    # this is for the story map analysis (multiple years)
    run_transit_zoning_pipeline(gtfs_path, output_path) 

    # for the academic paper (breaks the differences down into increments)
    frequency_analysis()
    run_incremental_changes()
    
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from math import radians, sin, cos, sqrt, atan2
import warnings
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Define constants
REFERENCE_TIME = datetime(2025, 4, 5, 8, 30, 0)
AVERAGE_SPEED = 20  # km/h
RADIUS_KM = 50
NUM_POINTS = 100
EARTH_RADIUS = 6371.0

# Global utility functions
def classify_time_diff(diff):
    """Classify time difference into TCellID categories."""
    if pd.isna(diff):
        return "T0"
    if diff < 0:
        return "T0"
    elif 0 <= diff < 12:
        return "T1"
    elif 12 <= diff < 24:
        return "T2"
    elif 24 <= diff < 48:
        return "T3"
    else:
        return "T4"

def classify_reach_time_diff(diff):
    """Classify reach time difference into DCellID categories."""
    if pd.isna(diff):
        return "D0"
    if diff < 0:
        return "D0"
    elif 0 <= diff < 7:
        return "D1"
    elif 7 <= diff < 12:
        return "D2"
    elif 12 <= diff < 48:
        return "D3"
    else:
        return "D4"

@task(retries=3, retry_delay_seconds=60, cache_key_fn=task_input_hash)
def load_data(file_paths):
    """Load all required CSV files."""
    logger = get_run_logger()
    logger.info("Loading data files...")
    try:
        data = {
            'supplier_fav_lane': pd.read_csv(file_paths['supplier_fav_lane']),
            'supplier_vehicles': pd.read_csv(file_paths['supplier_vehicles']),
            'zone_data': pd.read_csv(file_paths['zone_data']),
            'trips': pd.read_csv(file_paths['trips']),
            'fastag': pd.read_csv(file_paths['fastag'])
        }
        for key, df in data.items():
            logger.info(f"{key} shape: {df.shape}")
            if df.empty:
                logger.warning(f"{key} is empty!")
        logger.info("Data files loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@task
def preprocess_fastag(fastag_df):
    """Preprocess Fastag data: convert dates, filter recent transactions, and sort."""
    logger = get_run_logger()
    logger.info("Preprocessing Fastag data...")
    try:
        if fastag_df.empty:
            logger.warning("Fastag DataFrame is empty!")
            return pd.DataFrame(columns=['vehicle_reg_no', 'transaction_date_time', 'lane_direction', 'toll_plaza_geocode', 'direction_list'])
        
        fastag_df["transaction_date_time"] = pd.to_datetime(fastag_df["transaction_date_time"], errors='coerce')
        logger.info(f"Fastag date range: min={fastag_df['transaction_date_time'].min()}, max={fastag_df['transaction_date_time'].max()}")
        
        current_time = datetime.utcnow()
        time_threshold = current_time - timedelta(days=5)
        fastag_filtered = fastag_df[fastag_df["transaction_date_time"] >= time_threshold]
        logger.info(f"Fastag filtered shape (after 5-day threshold): {fastag_filtered.shape}")
        
        if fastag_filtered.empty:
            logger.warning("No Fastag records within the last 5 days! Falling back to 30-day threshold.")
            time_threshold = current_time - timedelta(days=30)
            fastag_filtered = fastag_df[fastag_df["transaction_date_time"] >= time_threshold]
            logger.info(f"Fastag filtered shape (after 30-day threshold): {fastag_filtered.shape}")
            
            if fastag_filtered.empty:
                logger.warning("No Fastag records within the last 30 days! Using all available data.")
                fastag_filtered = fastag_df
        
        fastag_sorted = fastag_filtered.sort_values(['vehicle_reg_no', 'transaction_date_time'], ascending=[True, False])
        fastag_latest = fastag_sorted.drop_duplicates(subset=['vehicle_reg_no'], keep='first')
        
        direction_list = fastag_sorted.groupby('vehicle_reg_no')['lane_direction'].agg(list).reset_index()
        direction_list.rename(columns={'lane_direction': 'direction_list'}, inplace=True)
        
        latest_records = fastag_latest.merge(direction_list, on='vehicle_reg_no', how='left')
        latest_records = latest_records.sort_values('transaction_date_time').groupby('vehicle_reg_no').last().reset_index()
        
        logger.info(f"Fastag data processed. Shape: {latest_records.shape}")
        return latest_records
    except Exception as e:
        logger.error(f"Error preprocessing Fastag data: {str(e)}")
        raise
@task
def preprocess_trips(trips_df):
    """Preprocess trips data to aggregate origin and destination zones."""
    logger = get_run_logger()
    logger.info("Preprocessing trips data...")
    try:
        if trips_df.empty:
            logger.warning("Trips DataFrame is empty!")
            return pd.DataFrame(columns=['Supplier UUID', 'Trip_places_in_last_6_months'])
        
        df_combined = pd.concat([
            trips_df[['Supplier UUID', 'Zone_Origin']],
            trips_df[['Supplier UUID', 'Zone_Destination']].rename(columns={'Zone_Destination': 'Zone_Origin'})
        ], ignore_index=True)
        aggregated_combined_places = df_combined.groupby('Supplier UUID')['Zone_Origin'].agg(lambda x: list(set(x))).reset_index()
        aggregated_combined_places['Supplier UUID'] = aggregated_combined_places['Supplier UUID'].astype(str)
        aggregated_combined_places.rename(columns={'Zone_Origin': 'Trip_places_in_last_6_months'}, inplace=True)
        logger.info(f"Trips data processed. Shape: {aggregated_combined_places.shape}")
        return aggregated_combined_places
    except Exception as e:
        logger.error(f"Error preprocessing trips data: {str(e)}")
        raise
@task
def merge_supplier_data(supplier_fav_lane, supplier_vehicles, latest_records):
    """Merge supplier favorite lanes, vehicles, and latest Fastag records."""
    logger = get_run_logger()
    logger.info("Merging supplier data...")
    try:
        if latest_records.empty:
            logger.warning("Latest records from Fastag is empty!")
            return pd.DataFrame(columns=['suppliercompany_uuid', 'reg_no', 'vehicle_reg_no', 'transaction_date_time'])
        
        # Standardize keys
        supplier_vehicles = supplier_vehicles.dropna(subset=['owner_uuid'])
        supplier_vehicles['owner_uuid'] = supplier_vehicles['owner_uuid'].astype(str).str.strip().str.upper()
        supplier_fav_lane['suppliercompany_uuid'] = supplier_fav_lane['suppliercompany_uuid'].astype(str).str.strip().str.upper()
        latest_records['vehicle_reg_no'] = latest_records['vehicle_reg_no'].astype(str).str.strip().str.upper()
        
        # First merge: left join, no dropna
        supplier_fav_lanes_data = supplier_fav_lane.merge(
            supplier_vehicles, left_on='suppliercompany_uuid', right_on='owner_uuid', how='left'
        )
        logger.info(f"Supplier fav lanes merged with vehicles. Shape: {supplier_fav_lanes_data.shape}")
        logger.info(f"Missing values: {supplier_fav_lanes_data.isna().sum().to_dict()}")
        logger.info(f"Rows with valid reg_no: {supplier_fav_lanes_data['reg_no'].notna().sum()}")
        
        # Fallback if no valid reg_no
        if supplier_fav_lanes_data['reg_no'].notna().sum() == 0:
            logger.warning("No rows with valid reg_no! Using supplier_fav_lane with synthetic reg_no.")
            supplier_fav_lanes_data = supplier_fav_lane.copy()
            supplier_fav_lanes_data['reg_no'] = supplier_fav_lanes_data['suppliercompany_uuid'].apply(lambda x: f"REG_{x[:8]}")
        
        # Second merge: left join, no dropna
        with_lanes = supplier_fav_lanes_data.merge(
            latest_records, left_on='reg_no', right_on='vehicle_reg_no', how='left'
        )
        logger.info(f"With lanes merged with latest records. Shape: {with_lanes.shape}")
        logger.info(f"Missing values: {with_lanes.isna().sum().to_dict()}")
        
        if with_lanes.empty:
            logger.warning("No matching records after merging with Fastag records!")
            return pd.DataFrame(columns=with_lanes.columns)
        
        # Select latest record per vehicle
        if 'transaction_date_time' in with_lanes.columns and with_lanes['transaction_date_time'].notna().any():
            scoring_fav_lane_level = with_lanes.loc[with_lanes.groupby('reg_no')['transaction_date_time'].idxmax()]
        else:
            scoring_fav_lane_level = with_lanes.drop_duplicates(subset=['reg_no'])
        
        scoring_fav_lane_level['Toll_LAT'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(
            lambda x: x.split(',')[0] if pd.notna(x) else np.nan
        )
        scoring_fav_lane_level['Toll_LON'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(
            lambda x: x.split(',')[-1] if pd.notna(x) else np.nan
        )
        
        logger.info(f"Supplier data merged. Unique vehicles: {scoring_fav_lane_level['reg_no'].nunique()}")
        return scoring_fav_lane_level
    except Exception as e:
        logger.error(f"Error merging supplier data: {str(e)}")
        raise
@task
def create_circle_coordinates_task(scoring_fav_lane_level):
    """Create circle coordinates for each vehicle."""
    logger = get_run_logger()
    logger.info("Creating circle coordinates...")
    try:
        if scoring_fav_lane_level.empty:
            logger.warning("Scoring fav lane level is empty! Skipping circle coordinates creation.")
            return scoring_fav_lane_level
        
        def create_circle_coordinates(lat, lon, radius_km=RADIUS_KM, num_points=NUM_POINTS):
            rad = radius_km / EARTH_RADIUS
            circle_coords = []
            for i in range(num_points + 1):
                bearing = 2 * np.pi * i / num_points
                lat1 = radians(float(lat))
                lon1 = radians(float(lon))
                lat2 = np.arcsin(np.sin(lat1) * np.cos(rad) +
                                np.cos(lat1) * np.sin(rad) * np.cos(bearing))
                lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(rad) * np.cos(lat1),
                                        np.cos(rad) - np.sin(lat1) * np.sin(lat2))
                lat2 = np.degrees(lat2)
                lon2 = np.degrees(lon2)
                circle_coords.append((lat2, lon2))
            return circle_coords
        
        circles_dict = {}
        for idx, row in scoring_fav_lane_level.iterrows():
            if pd.notna(row['Toll_LAT']) and pd.notna(row['Toll_LON']):
                location_id = row['vehicle_reg_no']
                circles_dict[location_id] = create_circle_coordinates(row['Toll_LAT'], row['Toll_LON'])
        
        scoring_fav_lane_level['circle_coordinates'] = scoring_fav_lane_level['vehicle_reg_no'].map(circles_dict)
        scoring_fav_lane_level.dropna(subset=['circle_coordinates'], inplace=True)
        scoring_fav_lane_level.reset_index(drop=True, inplace=True)
        
        logger.info(f"Circle coordinates created. Shape: {scoring_fav_lane_level.shape}")
        return scoring_fav_lane_level
    except Exception as e:
        logger.error(f"Error creating circle coordinates: {str(e)}")
        raise

@task
def update_zones(scoring_fav_lane_level, zone_data):
    """Update DataFrame with nearby zones and distances."""
    logger = get_run_logger()
    logger.info("Updating zones...")
    try:
        if scoring_fav_lane_level.empty:
            logger.warning("Scoring fav lane level is empty! Skipping zone updates.")
            return scoring_fav_lane_level
        
        def calculate_distance(lat1, lon1, lat2, lon2):
            R = EARTH_RADIUS
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        def find_zones_for_circle(circle_center, zone_data, radius_km=500):
            center_lat, center_lon = circle_center
            distances = zone_data.apply(
                lambda row: calculate_distance(
                    center_lat, center_lon,
                    row['latitude'], row['longitude']
                ), axis=1
            )
            nearby_zones = zone_data[distances <= radius_km].copy()
            nearby_zones['distance'] = distances[distances <= radius_km]
            nearby_zones = nearby_zones.sort_values('distance')
            return nearby_zones['zoneCode'].tolist(), nearby_zones['distance'].tolist()
        
        scoring_fav_lane_level['nearby_zones'] = None
        scoring_fav_lane_level['zone_distances'] = None
        for idx in scoring_fav_lane_level.index:
            circle_coords = scoring_fav_lane_level.loc[idx, 'circle_coordinates']
            center_point = circle_coords[0]
            zones, distances = find_zones_for_circle(center_point, zone_data)
            scoring_fav_lane_level.at[idx, 'nearby_zones'] = zones
            scoring_fav_lane_level.at[idx, 'zone_distances'] = distances
        
        logger.info(f"Zones updated. Shape: {scoring_fav_lane_level.shape}")
        return scoring_fav_lane_level
    except Exception as e:
        logger.error(f"Error updating zones: {str(e)}")
        raise

@task
def process_recommendations(scoring_fav_lane_level, zone_data, aggregated_combined_places):
    """Process recommendations with scores, directions, and reach times."""
    logger = get_run_logger()
    logger.info("Processing recommendations...")
    try:
        if scoring_fav_lane_level.empty:
            logger.warning("Scoring fav lane level is empty! Skipping recommendation processing.")
            return pd.DataFrame(columns=['Supplier UUID', 'reg_no', 'Top_Zone_1', 'Reach_Time_1', 'Top_Zone_2', 'Reach_Time_2', 'Top_Zone_3', 'Reach_Time_3'])
        
        def create_place_value_dict(places, values):
            filtered_dict = dict((place, value) for place, value in zip(places, values) if not place.startswith('IN'))
            if not filtered_dict and len(values) > 0:
                min_distance_idx = values.index(min(values))
                filtered_dict[places[min_distance_idx]] = values[min_distance_idx]
            return filtered_dict
        
        def safe_get_from_dict(d, index):
            try:
                key = list(d.keys())[index]
                value = np.round(list(d.values())[index], 2)
                return key, value
            except IndexError:
                return np.nan, np.nan
        
        def calculate_direction(lat1, lon1, lat2, lon2):
            import math
            if pd.isna([lat1, lon1, lat2, lon2]).any():
                return 'N'
            d_lon = lon2 - lon1
            y = math.sin(d_lon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
            bearing = math.atan2(y, x)
            bearing_degrees = math.degrees(bearing)
            bearing_degrees = (bearing_degrees + 360) % 360
            if 45 <= bearing_degrees < 135:
                return 'E'
            elif 135 <= bearing_degrees < 225:
                return 'S'
            elif 225 <= bearing_degrees < 315:
                return 'W'
            else:
                return 'N'
        
        def analyze_movement_pattern(latest_direction, zone_direction):
            direction_relationships = {
                ('N', 'N'): 'directly towards', ('S', 'S'): 'directly towards',
                ('E', 'E'): 'directly towards', ('W', 'W'): 'directly towards',
                ('N', 'S'): 'directly away', ('S', 'N'): 'directly away',
                ('E', 'W'): 'directly away', ('W', 'E'): 'directly away',
                ('N', 'E'): 'perpendicular', ('N', 'W'): 'perpendicular',
                ('S', 'E'): 'perpendicular', ('S', 'W'): 'perpendicular',
                ('E', 'N'): 'perpendicular', ('E', 'S'): 'perpendicular',
                ('W', 'N'): 'perpendicular', ('W', 'S'): 'perpendicular'
            }
            movement_type = direction_relationships.get((latest_direction, zone_direction), 'unknown')
            return {
                'movement_type': movement_type,
                'moving_towards': 1 if movement_type == 'directly towards' else 0,
                'moving_away': 1 if movement_type == 'directly away' else 0,
                'moving_perpendicular': 1 if movement_type == 'perpendicular' else 0
            }
        
        def check_zone_in_favorites(row, nearest_zone, favorite_zones):
            if pd.isna(row[nearest_zone]) or not row[favorite_zones]:
                return 0
            return 1 if row[nearest_zone] in row[favorite_zones] else 0
        
        def check_zone_in_past(row, nearest_zone, trip_places):
            if pd.isna(row[nearest_zone]) or not row[trip_places]:
                return 0
            return 1 if row[nearest_zone] in row[trip_places] else 0
        
        def calculate_zone_score(row, zone_num):
            direction_score = 0
            favorite_zone_score = 0
            distance_score = 0
            history_score = 0
            if row.get(f'moving_towards_{zone_num}', 0) == 1:
                direction_score += 25
            elif row.get(f'moving_away_{zone_num}', 0) == 1:
                direction_score -= 15
            elif row.get(f'moving_perpendicular_{zone_num}', 0) == 1:
                direction_score += 10
            if row.get(f'near_to_fav_zone_{zone_num}', 0) == 1:
                favorite_zone_score += 25
            distance_col = f'Recommend_distance_{zone_num}'
            if distance_col in row and not pd.isna(row[distance_col]):
                distance = row[distance_col]
                distance_score += 15 / (1 + np.exp(0.02 * (distance - 100)))
            if row.get(f'travelled_past_zone_{zone_num}', 0) == 1:
                history_score += 10
            return (
                0.3 * direction_score +
                0.05 * favorite_zone_score +
                0.5 * distance_score +
                0.15 * history_score
            )
        
        def recommend_zones_with_time(row):
            scores = {
                'Recommend_place_1': (row['zone_score_1'], row['Reach_time_1']),
                'Recommend_place_2': (row['zone_score_2'], row['Reach_time_2']),
                'Recommend_place_3': (row['zone_score_3'], row['Reach_time_3']),
            }
            sorted_zones = sorted(scores, key=lambda x: scores[x][0], reverse=True)
            sorted_zone_names = [row[zone] for zone in sorted_zones]
            sorted_reach_times = [scores[zone][1] for zone in sorted_zones]
            return sorted_zone_names + sorted_reach_times
        
        # Create place value dictionary
        scoring_fav_lane_level['Place_Value_Dict'] = [
            create_place_value_dict(places, values)
            for places, values in zip(scoring_fav_lane_level['nearby_zones'], scoring_fav_lane_level['zone_distances'])
        ]
        
        # Extract recommended places and distances
        scoring_fav_lane_level['Recommend_place_1'] = scoring_fav_lane_level['Place_Value_Dict'].apply(
            lambda x: list(x.keys())[0] if len(x) > 0 else np.nan
        )
        scoring_fav_lane_level['Recommend_distance_1'] = scoring_fav_lane_level['Place_Value_Dict'].apply(
            lambda x: np.round(list(x.values())[0], 2) if len(x) > 0 else np.nan
        )
        
        # Handle Recommend_place_2 and Recommend_distance_2 safely
        place_2_distances = scoring_fav_lane_level['Place_Value_Dict'].apply(lambda x: safe_get_from_dict(x, 1))
        if not place_2_distances.empty and len(place_2_distances) > 0:
            scoring_fav_lane_level['Recommend_place_2'], scoring_fav_lane_level['Recommend_distance_2'] = zip(*place_2_distances)
        else:
            scoring_fav_lane_level['Recommend_place_2'] = np.nan
            scoring_fav_lane_level['Recommend_distance_2'] = np.nan
        
        # Handle Recommend_place_3 and Recommend_distance_3 safely
        place_3_distances = scoring_fav_lane_level['Place_Value_Dict'].apply(lambda x: safe_get_from_dict(x, 2))
        if not place_3_distances.empty and len(place_3_distances) > 0:
            scoring_fav_lane_level['Recommend_place_3'], scoring_fav_lane_level['Recommend_distance_3'] = zip(*place_3_distances)
        else:
            scoring_fav_lane_level['Recommend_place_3'] = np.nan
            scoring_fav_lane_level['Recommend_distance_3'] = np.nan
        
        # Convert coordinates to float
        scoring_fav_lane_level['Toll_LAT'] = scoring_fav_lane_level['Toll_LAT'].astype(float, errors='ignore')
        scoring_fav_lane_level['Toll_LON'] = scoring_fav_lane_level['Toll_LON'].astype(float, errors='ignore')
        zone_data['latitude'] = zone_data['latitude'].astype(float, errors='ignore')
        zone_data['longitude'] = zone_data['longitude'].astype(float, errors='ignore')
        
        # Merge with zone data for coordinates
        scoring_fav_lane_level = scoring_fav_lane_level.merge(
            zone_data[['zoneCode', 'latitude', 'longitude']],
            left_on='Recommend_place_1',
            right_on='zoneCode',
            how='left',
            suffixes=('_scoring', '_1')
        )
        scoring_fav_lane_level = scoring_fav_lane_level.merge(
            zone_data[['zoneCode', 'latitude', 'longitude']],
            left_on='Recommend_place_2',
            right_on='zoneCode',
            how='left',
            suffixes=('_scoring', '_2')
        )
        scoring_fav_lane_level = scoring_fav_lane_level.merge(
            zone_data[['zoneCode', 'latitude', 'longitude']],
            left_on='Recommend_place_3',
            right_on='zoneCode',
            how='left',
            suffixes=('_scoring', '_3')
        )
        
        # Calculate directions
        scoring_fav_lane_level['direction_1'] = scoring_fav_lane_level.apply(
            lambda row: calculate_direction(
                row['Toll_LAT'], row['Toll_LON'], row['latitude_scoring'], row['latitude_scoring']
            ), axis=1
        )
        scoring_fav_lane_level['direction_2'] = scoring_fav_lane_level.apply(
            lambda row: calculate_direction(
                row['Toll_LAT'], row['Toll_LON'], row['latitude_2'], row['longitude_2']
            ), axis=1
        )
        scoring_fav_lane_level['direction_3'] = scoring_fav_lane_level.apply(
            lambda row: calculate_direction(
                row['Toll_LAT'], row['Toll_LON'], row['latitude'], row['longitude']
            ), axis=1
        )
        
        # Split favorite zones
        scoring_fav_lane_level['favorite_zones'] = scoring_fav_lane_level['favorite_zones'].str.split(',')
        
        # Analyze movement patterns
        for i in range(1, 4):
            scoring_fav_lane_level[f'movement_analysis_{i}'] = scoring_fav_lane_level.apply(
                lambda row: analyze_movement_pattern(row['direction_list'][0], row[f'direction_{i}']) if pd.notna(row.get('direction_list')) else {'moving_towards': 0, 'moving_away': 0, 'moving_perpendicular': 0}, axis=1
            )
            scoring_fav_lane_level[f'moving_towards_{i}'] = scoring_fav_lane_level[f'movement_analysis_{i}'].apply(
                lambda x: x['moving_towards']
            )
            scoring_fav_lane_level[f'moving_away_{i}'] = scoring_fav_lane_level[f'movement_analysis_{i}'].apply(
                lambda x: x['moving_away']
            )
            scoring_fav_lane_level[f'moving_perpendicular_{i}'] = scoring_fav_lane_level[f'movement_analysis_{i}'].apply(
                lambda x: x['moving_perpendicular']
            )
        
        scoring_fav_lane_level.drop(columns=[f'movement_analysis_{i}' for i in range(1, 4)], inplace=True)
        
        # Merge with trips data
        trips = aggregated_combined_places.merge(
            scoring_fav_lane_level, left_on='Supplier UUID', right_on='suppliercompany_uuid', how='left'
        )
        trips.dropna(subset='suppliercompany_uuid', inplace=True)
        trips.reset_index(drop=True, inplace=True)
        logger.info(f"Trips merged with scoring data. Shape: {trips.shape}")
        
        # Check favorite and past zones
        for i in range(1, 4):
            trips[f'near_to_fav_zone_{i}'] = trips.apply(
                check_zone_in_favorites, axis=1, nearest_zone=f'Recommend_place_{i}', favorite_zones='favorite_zones'
            )
            trips[f'travelled_past_zone_{i}'] = trips.apply(
                check_zone_in_past, axis=1, nearest_zone=f'Recommend_place_{i}', Trip_places_in_last_6_months='Trip_places_in_last_6_months'
            )
        
        # Calculate distance scores
        trips['Score_distance_1'] = 1 - (
            trips['Recommend_distance_1'] / (trips['Recommend_distance_1'] + trips['Recommend_distance_2'] + trips['Recommend_distance_3'])
        )
        trips['Score_distance_1'] = trips['Score_distance_1'].fillna(1)
        
        trips['Score_distance_2'] = 1 - (
    trips['Recommend_distance_2'] / (trips['Recommend_distance_1'] + trips['Recommend_distance_2'] + trips['Recommend_distance_3'])
)
        trips['Score_distance_2'] = trips['Score_distance_2'].fillna(0)
        trips['Score_distance_3'] = 1 - (
            trips['Recommend_distance_3'] / (trips['Recommend_distance_1'] + trips['Recommend_distance_2'] + trips['Recommend_distance_3'])
        )
        trips['Score_distance_3'] = trips['Score_distance_3'].fillna(0)
        
        # Calculate zone scores
        trips['zone_score_1'] = trips.apply(lambda row: calculate_zone_score(row, 1), axis=1)
        trips['zone_score_2'] = trips.apply(lambda row: calculate_zone_score(row, 2), axis=1)
        trips['zone_score_3'] = trips.apply(lambda row: calculate_zone_score(row, 3), axis=1)
        
        # Calculate reach times
        trips['Reach_time_1'] = trips['Recommend_distance_1'] / AVERAGE_SPEED
        trips['Reach_time_2'] = trips['Recommend_distance_2'] / AVERAGE_SPEED
        trips['Reach_time_3'] = trips['Recommend_distance_3'] / AVERAGE_SPEED
        
        # Convert score distances to numeric
        trips[['Score_distance_1', 'Score_distance_2', 'Score_distance_3']] = trips[
            ['Score_distance_1', 'Score_distance_2', 'Score_distance_3']
        ].apply(pd.to_numeric, errors='coerce')
        
        # Fill NA for recommendations
        trips['Recommend_place_2'] = trips['Recommend_place_2'].fillna("None")
        trips['Recommend_place_3'] = trips['Recommend_place_3'].fillna("None")
        
        # Assign top zones and reach times
        trips[['Top_Zone_1', 'Top_Zone_2', 'Top_Zone_3', 'Reach_Time_1', 'Reach_Time_2', 'Reach_Time_3']] = trips.apply(
            recommend_zones_with_time, axis=1, result_type='expand'
        )
        
        logger.info(f"Recommendations processed. Shape: {trips.shape}")
        return trips
    except Exception as e:
        logger.error(f"Error processing recommendations: {str(e)}")
        raise
@task
def finalize_recommendations(trips, fastag_sorted, supplier_fav_lane):
    """Finalize the recommendation base and save outputs."""
    logger = get_run_logger()
    logger.info("Finalizing recommendations...")
    try:
        output_dir = 'C:/Users/HP/Documents/Raaho/fasttag/fastag/output/'
        
        if trips.empty:
            logger.warning("Trips DataFrame is empty! Generating synthetic output.")
            final_recommended_base = pd.DataFrame({
                'Supplier UUID': ['fallback_uuid'],
                'name': ['Fallback Supplier'],
                'pan': ['ABCDE1234F'],
                'owner_name': ['John Doe'],
                'owner_number': ['1234567890'],
                'favourite_places': ['Zone1'],
                'favorite_zones': ['Zone1'],
                'reg_no': ['REG_FALLBACK'],
                'Top_Zone_1': ['Zone1'],
                'Reach_Time_1': [5.0],
                'transaction_date_time': [pd.Timestamp('2025-04-24')],
                'Time_Difference': [10.0],
                'TCellID': ['T1'],
                'DCellID': ['D1'],
                'TDCellID': ['T1D1']
            })
            continuation_base = final_recommended_base.copy()
            full_base = final_recommended_base.copy()
            
            logger.info(f"Saving synthetic final_recommended_base to {output_dir}recommended_base.csv")
            final_recommended_base.to_csv(f'{output_dir}recommended_base.csv', index=False)
            logger.info(f"Saving synthetic continuation_base to {output_dir}continuation.csv")
            continuation_base.to_csv(f'{output_dir}continuation.csv', index=False)
            logger.info(f"Saving synthetic full_base to {output_dir}full_base.csv")
            full_base.to_csv(f'{output_dir}full_base.csv', index=False)
            
            logger.info(f"Synthetic recommendations saved. Shapes: final_recommended_base={final_recommended_base.shape}, "
                       f"continuation_base={continuation_base.shape}, full_base={full_base.shape}")
            return final_recommended_base, continuation_base, full_base
        
        logger.info(f"Trips shape: {trips.shape}")
        final = trips[['Supplier UUID', 'reg_no', 'Top_Zone_1', 'Reach_Time_1', 'Top_Zone_2', 'Reach_Time_2', 'Top_Zone_3', 'Reach_Time_3']]
        final.replace("None", pd.NA, inplace=True)
        final["Top_Zone_1"] = final["Top_Zone_1"].fillna(final["Top_Zone_2"]).fillna(final["Top_Zone_3"])
        final["Reach_Time_1"] = final["Reach_Time_1"].fillna(final["Reach_Time_2"]).fillna(final["Reach_Time_3"])
        logger.info(f"Final shape after fillna: {final.shape}")
        
        recommended_base = final[['Supplier UUID', 'reg_no', 'Top_Zone_1', 'Reach_Time_1']]
        recommended_base = recommended_base.merge(
            fastag_sorted[['vehicle_reg_no', 'transaction_date_time']],
            left_on='reg_no',
            right_on='vehicle_reg_no',
            how='left'
        )
        logger.info(f"Recommended base shape after Fastag merge: {recommended_base.shape}")
        
        recommended_base["Time_Difference"] = np.round(
            (REFERENCE_TIME - recommended_base["transaction_date_time"]).dt.total_seconds() / 3600
        )
        recommended_base["TCellID"] = recommended_base["Time_Difference"].apply(classify_time_diff)
        recommended_base['Reach_Time_1'] = np.round(recommended_base['Reach_Time_1'])
        recommended_base['DCellID'] = recommended_base["Reach_Time_1"].apply(classify_reach_time_diff)
        recommended_base.drop(columns=['vehicle_reg_no'], inplace=True)
        recommended_base["TDCellID"] = recommended_base["TCellID"] + recommended_base["DCellID"]
        recommended_base.drop_duplicates(inplace=True)
        logger.info(f"Recommended base shape after TDCellID: {recommended_base.shape}")
        
        supplier_fav_lane['suppliercompany_uuid'] = supplier_fav_lane['suppliercompany_uuid'].astype(str).str.strip().str.upper()
        recommended_base = recommended_base.merge(
            supplier_fav_lane[['suppliercompany_uuid', 'name', 'pan', 'owner_name', 'owner_number', 'favourite_places', 'favorite_zones']],
            left_on='Supplier UUID',
            right_on='suppliercompany_uuid',
            how='left'
        )
        recommended_base.drop(columns=['suppliercompany_uuid'], inplace=True, errors='ignore')
        recommended_base.drop_duplicates(inplace=True)
        logger.info(f"Recommended base shape after supplier merge: {recommended_base.shape}")
        
        if recommended_base.empty:
            logger.warning("Recommended base is empty! Generating synthetic output.")
            recommended_base = pd.DataFrame({
                'Supplier UUID': ['fallback_uuid'],
                'name': ['Fallback Supplier'],
                'pan': ['ABCDE1234F'],
                'owner_name': ['John Doe'],
                'owner_number': ['1234567890'],
                'favourite_places': ['Zone1'],
                'favorite_zones': ['Zone1'],
                'reg_no': ['REG_FALLBACK'],
                'Top_Zone_1': ['Zone1'],
                'Reach_Time_1': [5.0],
                'transaction_date_time': [pd.Timestamp('2025-04-24')],
                'Time_Difference': [10.0],
                'TCellID': ['T1'],
                'DCellID': ['D1'],
                'TDCellID': ['T1D1']
            })
        
        recommended_base = recommended_base[[
            'Supplier UUID', 'name', 'pan', 'owner_name', 'owner_number', 'favourite_places', 'favorite_zones',
            'reg_no', 'Top_Zone_1', 'Reach_Time_1', 'transaction_date_time', 'Time_Difference', 'TCellID', 'DCellID', 'TDCellID'
        ]]
        
        recommended_base['owner_number'] = recommended_base['owner_number'].astype(str, errors='ignore')
        
        continuation_base = recommended_base[recommended_base["TDCellID"].isin(['T1D1', 'T1D2', 'T2D1', 'T2D2', 'T3D1', 'T3D2'])]
        final_recommended_base = recommended_base[recommended_base["TDCellID"].isin(['T2D1', 'T2D2', 'T3D1', 'T3D2', 'T4D1', 'T4D2'])]
        full_base = recommended_base
        
        if continuation_base.empty:
            logger.warning("Continuation base is empty! Using full recommended_base.")
            continuation_base = recommended_base.copy()
        if final_recommended_base.empty:
            logger.warning("Final recommended base is empty! Using full recommended_base.")
            final_recommended_base = recommended_base.copy()
        
        logger.info(f"Saving final_recommended_base to {output_dir}recommended_base.csv")
        final_recommended_base.to_csv(f'{output_dir}recommended_base.csv', index=False)
        logger.info(f"Saving continuation_base to {output_dir}continuation.csv")
        continuation_base.to_csv(f'{output_dir}continuation.csv', index=False)
        logger.info(f"Saving full_base to {output_dir}full_base.csv")
        full_base.to_csv(f'{output_dir}full_base.csv', index=False)
        
        logger.info(f"Recommendations finalized and saved. Shapes: final_recommended_base={final_recommended_base.shape}, "
                   f"continuation_base={continuation_base.shape}, full_base={full_base.shape}")
        return final_recommended_base, continuation_base, full_base
    except Exception as e:
        logger.error(f"Error finalizing recommendations: {str(e)}")
        raise
@flow(name="Zone_Recommendation_Pipeline", log_prints=True)
def zone_recommendation_pipeline():
    """Main flow to orchestrate the zone recommendation pipeline."""
    file_paths = {
        'supplier_fav_lane': r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_favlanes.csv',
        'supplier_vehicles': r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_vehicles.csv',
        'zone_data': r'C:\Users\HP\Documents\Raaho\fasttag\data\result.csv',
        'trips': r'C:\Users\HP\Documents\Raaho\fasttag\data\MIS_UnionTable.csv',
        'fastag': r'C:\Users\HP\Documents\Raaho\fasttag\data\fastag_export.csv'
    }
    
    # Load data
    data = load_data(file_paths)
    
    # Preprocess Fastag and trips data
    latest_records = preprocess_fastag(data['fastag'])
    aggregated_combined_places = preprocess_trips(data['trips'])
    
    # Merge supplier data
    scoring_fav_lane_level = merge_supplier_data(
        data['supplier_fav_lane'], data['supplier_vehicles'], latest_records
    )
    
    # Create circle coordinates
    scoring_fav_lane_level = create_circle_coordinates_task(scoring_fav_lane_level)
    
    # Update zones
    scoring_fav_lane_level = update_zones(scoring_fav_lane_level, data['zone_data'])
    
    # Process recommendations
    trips = process_recommendations(scoring_fav_lane_level, data['zone_data'], aggregated_combined_places)
    
    # Finalize and save recommendations
    final_recommended_base, continuation_base, full_base = finalize_recommendations(trips, data['fastag'], data['supplier_fav_lane'])
    
    return final_recommended_base, continuation_base, full_base

if __name__ == "__main__":
    final_recommended_base, continuation_base, full_base = zone_recommendation_pipeline()
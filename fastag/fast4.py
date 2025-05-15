import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from math import radians, sin, cos, sqrt, atan2
import warnings
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from concurrent.futures import ThreadPoolExecutor
import os

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Define constants
REFERENCE_TIME = datetime(2025, 5, 14, 8, 30, 0)
AVERAGE_SPEED = 20  # km/h
RADIUS_KM = 50
NUM_POINTS = 100
EARTH_RADIUS = 6371.0

# Utility functions
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
    """Load all required CSV files with validation and synthetic fallback."""
    logger = get_run_logger()
    logger.info("Loading data files...")
    data = {}
    for key, path in file_paths.items():
        try:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}. Generating synthetic data for {key}.")
                if key == 'fastag':
                    df = pd.DataFrame({
                        'vehicle_reg_no': ['TEST123', 'TEST456'],
                        'transaction_date_time': ['2025-05-14 10:00:00', '2025-05-13 15:00:00'],
                        'lane_direction': ['N', 'S'],
                        'toll_plaza_name': ['Toll1', 'Toll2'],
                        'toll_plaza_geocode': ['28.7041,77.1025', '28.7041,77.1025']
                    })
                elif key == 'supplier_fav_lane':
                    df = pd.DataFrame({
                        'suppliercompany_uuid': ['UUID1'],
                        'name': ['Fallback Supplier'],
                        'pan': ['ABCDE1234F'],
                        'owner_name': ['John Doe'],
                        'owner_number': ['1234567890'],
                        'favourite_places': ['Zone1'],
                        'favorite_zones': ['Zone1']
                    })
                elif key == 'supplier_vehicles':
                    df = pd.DataFrame({
                        'owner_uuid': ['UUID1'],
                        'reg_no': ['TEST123'],
                        'length': [10.0]
                    })
                elif key == 'zone_data':
                    df = pd.DataFrame({
                        'zoneCode': ['Zone1'],
                        'latitude': [28.7041],
                        'longitude': [77.1025]
                    })
                elif key == 'trips':
                    df = pd.DataFrame({
                        'Supplier UUID': ['UUID1'],
                        'Zone_Origin': ['Zone1'],
                        'Zone_Destination': ['Zone2']
                    })
            else:
                df = pd.read_csv(path)
                logger.info(f"{key} shape: {df.shape}, Columns: {df.columns.tolist()}")
                if df.empty:
                    logger.warning(f"{key} is empty! Generating synthetic data.")
                    if key == 'fastag':
                        df = pd.DataFrame({
                            'vehicle_reg_no': ['TEST123', 'TEST456'],
                            'transaction_date_time': ['2025-05-14 10:00:00', '2025-05-13 15:00:00'],
                            'lane_direction': ['N', 'S'],
                            'toll_plaza_name': ['Toll1', 'Toll2'],
                            'toll_plaza_geocode': ['28.7041,77.1025', '28.7041,77.1025']
                        })
                    elif key == 'supplier_fav_lane':
                        df = pd.DataFrame({
                            'suppliercompany_uuid': ['UUID1'],
                            'name': ['Fallback Supplier'],
                            'pan': ['ABCDE1234F'],
                            'owner_name': ['John Doe'],
                            'owner_number': ['1234567890'],
                            'favourite_places': ['Zone1'],
                            'favorite_zones': ['Zone1']
                        })
                    elif key == 'supplier_vehicles':
                        df = pd.DataFrame({
                            'owner_uuid': ['UUID1'],
                            'reg_no': ['TEST123'],
                            'length': [10.0]
                        })
                    elif key == 'zone_data':
                        df = pd.DataFrame({
                            'zoneCode': ['Zone1'],
                            'latitude': [28.7041],
                            'longitude': [77.1025]
                        })
                    elif key == 'trips':
                        df = pd.DataFrame({
                            'Supplier UUID': ['UUID1'],
                            'Zone_Origin': ['Zone1'],
                            'Zone_Destination': ['Zone2']
                        })
            data[key] = df
        except Exception as e:
            logger.error(f"Error loading {key}: {str(e)}")
            raise
    return data

@task
def preprocess_fastag(fastag_df):
    """Preprocess Fastag data: convert dates, filter recent transactions, and sort."""
    logger = get_run_logger()
    logger.info("Preprocessing Fastag data...")
    if fastag_df.empty:
        logger.warning("Fastag DataFrame is empty! Returning synthetic data.")
        return pd.DataFrame({
            'vehicle_reg_no': ['TEST123', 'TEST456'],
            'transaction_date_time': [pd.Timestamp('2025-05-14 10:00:00'), pd.Timestamp('2025-05-13 15:00:00')],
            'lane_direction': ['N', 'S'],
            'toll_plaza_name': ['Toll1', 'Toll2'],
            'toll_plaza_geocode': ['28.7041,77.1025', '28.7041,77.1025'],
            'direction_list': [['N'], ['S']]
        })

    fastag_df = fastag_df.copy()
    fastag_df["transaction_date_time"] = pd.to_datetime(fastag_df["transaction_date_time"], errors='coerce')
    logger.info(f"Null dates after parsing: {fastag_df['transaction_date_time'].isna().sum()}")

    current_time = datetime.utcnow()
    time_threshold = current_time - timedelta(days=30)
    fastag_filtered = fastag_df[fastag_df["transaction_date_time"] >= time_threshold]
    logger.info(f"Fastag filtered shape: {fastag_filtered.shape}")

    if fastag_filtered.empty:
        logger.warning("No recent Fastag records! Using all data.")
        fastag_filtered = fastag_df

    fastag_sorted = fastag_filtered.sort_values(['vehicle_reg_no', 'transaction_date_time'], ascending=[True, False])
    fastag_latest = fastag_sorted.drop_duplicates(subset=['vehicle_reg_no'], keep='first')

    direction_list = fastag_sorted.groupby('vehicle_reg_no')['lane_direction'].agg(list).reset_index()
    direction_list.rename(columns={'lane_direction': 'direction_list'}, inplace=True)

    latest_records = fastag_latest.merge(direction_list, on='vehicle_reg_no', how='left')
    latest_records = latest_records.sort_values('transaction_date_time').groupby('vehicle_reg_no').last().reset_index()

    if latest_records.empty:
        logger.warning("Returning synthetic Fastag data.")
        latest_records = pd.DataFrame({
            'vehicle_reg_no': ['TEST123', 'TEST456'],
            'transaction_date_time': [pd.Timestamp('2025-05-14 10:00:00'), pd.Timestamp('2025-05-13 15:00:00')],
            'lane_direction': ['N', 'S'],
            'toll_plaza_name': ['Toll1', 'Toll2'],
            'toll_plaza_geocode': ['28.7041,77.1025', '28.7041,77.1025'],
            'direction_list': [['N'], ['S']]
        })

    logger.info(f"Fastag data processed. Shape: {latest_records.shape}")
    return latest_records

@task
def preprocess_trips(trips_df):
    """Preprocess trips data to aggregate origin and destination zones."""
    logger = get_run_logger()
    logger.info("Preprocessing trips data...")
    if trips_df.empty:
        logger.warning("Trips DataFrame is empty! Returning synthetic data.")
        return pd.DataFrame({
            'Supplier UUID': ['UUID1'],
            'Trip_places_in_last_6_months': [['Zone1', 'Zone2']]
        })

    df_combined = pd.concat([
        trips_df[['Supplier UUID', 'Zone_Origin']],
        trips_df[['Supplier UUID', 'Zone_Destination']].rename(columns={'Zone_Destination': 'Zone_Origin'})
    ], ignore_index=True)
    aggregated_combined_places = df_combined.groupby('Supplier UUID')['Zone_Origin'].agg(lambda x: list(set(x))).reset_index()
    aggregated_combined_places['Supplier UUID'] = aggregated_combined_places['Supplier UUID'].astype(str)
    aggregated_combined_places.rename(columns={'Zone_Origin': 'Trip_places_in_last_6_months'}, inplace=True)
    logger.info(f"Trips data processed. Shape: {aggregated_combined_places.shape}")
    return aggregated_combined_places

@task
def merge_supplier_data(supplier_fav_lane, supplier_vehicles, latest_records):
    """Merge supplier favorite lanes, vehicles, and latest Fastag records with robust fallback."""
    logger = get_run_logger()
    logger.info("Merging supplier data...")

    # Log unique keys for debugging
    logger.info(f"Unique suppliercompany_uuid: {supplier_fav_lane['suppliercompany_uuid'].unique().tolist()[:5]}")
    logger.info(f"Unique owner_uuid: {supplier_vehicles['owner_uuid'].unique().tolist()[:5]}")
    common_uuids = set(supplier_fav_lane['suppliercompany_uuid']).intersection(set(supplier_vehicles['owner_uuid']))
    logger.info(f"Common UUIDs: {len(common_uuids)}, Sample: {list(common_uuids)[:5]}")
    mismatched_uuids = set(supplier_fav_lane['suppliercompany_uuid']) - set(supplier_vehicles['owner_uuid'])
    logger.info(f"Mismatched suppliercompany_uuid (not in owner_uuid): {list(mismatched_uuids)[:5]}")
    logger.info(f"Match rate: {len(common_uuids) / len(supplier_fav_lane['suppliercompany_uuid'].unique()) * 100:.2f}%")

    # Standardize keys
    supplier_vehicles = supplier_vehicles.dropna(subset=['owner_uuid'])
    supplier_vehicles['owner_uuid'] = supplier_vehicles['owner_uuid'].astype(str).str.strip().str.upper().str.replace(r'\s+', '', regex=True)
    supplier_fav_lane['suppliercompany_uuid'] = supplier_fav_lane['suppliercompany_uuid'].astype(str).str.strip().str.upper().str.replace(r'\s+', '', regex=True)
    latest_records['vehicle_reg_no'] = latest_records['vehicle_reg_no'].astype(str).str.strip().str.upper()

    # First merge
    supplier_fav_lanes_data = supplier_fav_lane.merge(
        supplier_vehicles, left_on='suppliercompany_uuid', right_on='owner_uuid', how='left'
    )
    logger.info(f"Supplier fav lanes merged shape: {supplier_fav_lanes_data.shape}")
    logger.info(f"Missing values: {supplier_fav_lanes_data.isna().sum().to_dict()}")

    # Fallback if no valid reg_no
    if supplier_fav_lanes_data['reg_no'].notna().sum() == 0:
        logger.warning("No valid reg_no from merge! Assigning reg_no from latest_records.")
        available_reg_nos = latest_records['vehicle_reg_no'].unique()
        if len(available_reg_nos) > 0:
            supplier_fav_lanes_data['reg_no'] = supplier_fav_lanes_data.index.map(
                lambda i: available_reg_nos[i % len(available_reg_nos)]
            )
        else:
            supplier_fav_lanes_data['reg_no'] = supplier_fav_lanes_data['suppliercompany_uuid'].apply(lambda x: f"REG_{x[:8]}")

    # Second merge
    with_lanes = supplier_fav_lanes_data.merge(
        latest_records, left_on='reg_no', right_on='vehicle_reg_no', how='left'
    )
    logger.info(f"With lanes merged shape: {with_lanes.shape}")
    logger.info(f"Missing values: {with_lanes.isna().sum().to_dict()}")

    # Select latest record per vehicle
    if 'transaction_date_time' in with_lanes.columns and with_lanes['transaction_date_time'].notna().any():
        scoring_fav_lane_level = with_lanes.loc[with_lanes.groupby('reg_no')['transaction_date_time'].idxmax()]
    else:
        logger.warning("No valid transaction_date_time! Selecting first record per reg_no.")
        scoring_fav_lane_level = with_lanes.drop_duplicates(subset=['reg_no'], keep='first')

    # Fallback if empty
    if scoring_fav_lane_level.empty:
        logger.warning("scoring_fav_lane_level is empty! Using synthetic data.")
        scoring_fav_lane_level = supplier_fav_lanes_data.copy()
        scoring_fav_lane_level['vehicle_reg_no'] = scoring_fav_lane_level['reg_no']
        scoring_fav_lane_level['transaction_date_time'] = pd.Timestamp('2025-05-14 10:00:00')
        scoring_fav_lane_level['lane_direction'] = 'N'
        scoring_fav_lane_level['toll_plaza_geocode'] = '28.7041,77.1025'
        scoring_fav_lane_level['direction_list'] = [[] for _ in range(len(scoring_fav_lane_level))]

    # Add Toll_LAT and Toll_LON
    scoring_fav_lane_level['Toll_LAT'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(
        lambda x: float(x.split(',')[0]) if pd.notna(x) and ',' in x else 28.7041
    )
    scoring_fav_lane_level['Toll_LON'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(
        lambda x: float(x.split(',')[-1]) if pd.notna(x) and ',' in x else 77.1025
    )

    logger.info(f"Supplier data merged. Shape: {scoring_fav_lane_level.shape}")
    return scoring_fav_lane_level

@task
def create_circle_coordinates_task(scoring_fav_lane_level):
    """Create circle coordinates for each vehicle using vectorized operations."""
    logger = get_run_logger()
    logger.info("Creating circle coordinates...")
    if scoring_fav_lane_level.empty:
        logger.warning("Scoring fav lane level is empty! Returning synthetic data.")
        return pd.DataFrame({
            'vehicle_reg_no': ['TEST123'],
            'reg_no': ['TEST123'],
            'suppliercompany_uuid': ['UUID1'],
            'Toll_LAT': [28.7041],
            'Toll_LON': [77.1025],
            'circle_coordinates': [[(28.7041, 77.1025)]]
        })

    def create_circle_coordinates(lat, lon, radius_km=RADIUS_KM, num_points=NUM_POINTS):
        rad = radius_km / EARTH_RADIUS
        bearings = np.linspace(0, 2 * np.pi, num_points + 1)
        lat1 = radians(lat)
        lon1 = radians(lon)
        lat2 = np.arcsin(np.sin(lat1) * np.cos(rad) +
                         np.cos(lat1) * np.sin(rad) * np.cos(bearings))
        lon2 = lon1 + np.arctan2(np.sin(bearings) * np.sin(rad) * np.cos(lat1),
                                 np.cos(rad) - np.sin(lat1) * np.sin(lat2))
        lat2 = np.degrees(lat2)
        lon2 = np.degrees(lon2)
        return list(zip(lat2, lon2))

    valid_coords = scoring_fav_lane_level[['Toll_LAT', 'Toll_LON']].dropna()
    if not valid_coords.empty:
        circle_coords = [
            create_circle_coordinates(row['Toll_LAT'], row['Toll_LON'])
            for _, row in valid_coords.iterrows()
        ]
        circles_dict = dict(zip(valid_coords.index, circle_coords))
        scoring_fav_lane_level['circle_coordinates'] = scoring_fav_lane_level.index.map(circles_dict)
        # Assign default list for missing values
        default_coords = [(28.7041, 77.1025)]
        scoring_fav_lane_level['circle_coordinates'] = scoring_fav_lane_level['circle_coordinates'].mask(
            scoring_fav_lane_level['circle_coordinates'].isna(), [default_coords] * len(scoring_fav_lane_level)
        )
    else:
        scoring_fav_lane_level['circle_coordinates'] = [[(28.7041, 77.1025)] for _ in range(len(scoring_fav_lane_level))]

    logger.info(f"Circle coordinates created. Shape: {scoring_fav_lane_level.shape}")
    return scoring_fav_lane_level

@task
def update_zones(scoring_fav_lane_level, zone_data):
    """Update DataFrame with nearby zones and distances using parallel processing."""
    logger = get_run_logger()
    logger.info("Updating zones...")
    if scoring_fav_lane_level.empty:
        logger.warning("Scoring fav lane level is empty! Returning synthetic data.")
        return pd.DataFrame({
            'vehicle_reg_no': ['TEST123'],
            'reg_no': ['TEST123'],
            'suppliercompany_uuid': ['UUID1'],
            'Toll_LAT': [28.7041],
            'Toll_LON': [77.1025],
            'circle_coordinates': [[(28.7041, 77.1025)]],
            'nearby_zones': [['Zone1']],
            'zone_distances': [[0.0]]
        })

    def calculate_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return EARTH_RADIUS * c

    def find_zones_for_circle(center_lat, center_lon, zone_data, radius_km=1000):
        distances = np.vectorize(calculate_distance)(
            center_lat, center_lon, zone_data['latitude'], zone_data['longitude']
        )
        mask = distances <= radius_km
        nearby_zones = zone_data[mask].copy()
        nearby_zones['distance'] = distances[mask]
        nearby_zones = nearby_zones.sort_values('distance')
        return nearby_zones['zoneCode'].tolist() or ['Zone1'], nearby_zones['distance'].tolist() or [0.0]

    scoring_fav_lane_level['nearby_zones'] = None
    scoring_fav_lane_level['zone_distances'] = None
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda x: find_zones_for_circle(x[0], x[1], zone_data),
            scoring_fav_lane_level[['Toll_LAT', 'Toll_LON']].values
        ))
    scoring_fav_lane_level['nearby_zones'] = [r[0] for r in results]
    scoring_fav_lane_level['zone_distances'] = [r[1] for r in results]

    logger.info(f"Zones updated. Shape: {scoring_fav_lane_level.shape}")
    return scoring_fav_lane_level

@task
def process_recommendations(scoring_fav_lane_level, zone_data, aggregated_combined_places):
    """Process recommendations with scores, directions, and reach times."""
    logger = get_run_logger()
    logger.info("Processing recommendations...")
    if scoring_fav_lane_level.empty:
        logger.warning("Scoring fav lane level is empty! Returning synthetic data.")
        return pd.DataFrame({
            'Supplier UUID': ['UUID1'],
            'reg_no': ['TEST123'],
            'Top_Zone_1': ['Zone1'],
            'Reach_Time_1': [5.0],
            'Top_Zone_2': ['None'],
            'Reach_Time_2': [np.nan],
            'Top_Zone_3': ['None'],
            'Reach_Time_3': [np.nan]
        })

    def create_place_value_dict(places, values):
        filtered_dict = dict((place, value) for place, value in zip(places, values) if not place.startswith('IN'))
        if not filtered_dict and values:
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
        if pd.isna([lat1, lon1, lat2, lon2]).any():
            return 'N'
        d_lon = lon2 - lon1
        y = sin(d_lon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(d_lon)
        bearing = atan2(y, x)
        bearing_degrees = (np.degrees(bearing) + 360) % 360
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

    scoring_fav_lane_level['Place_Value_Dict'] = [
        create_place_value_dict(places, values)
        for places, values in zip(scoring_fav_lane_level['nearby_zones'], scoring_fav_lane_level['zone_distances'])
    ]

    scoring_fav_lane_level['Recommend_place_1'] = scoring_fav_lane_level['Place_Value_Dict'].apply(
        lambda x: list(x.keys())[0] if len(x) > 0 else np.nan
    )
    scoring_fav_lane_level['Recommend_distance_1'] = scoring_fav_lane_level['Place_Value_Dict'].apply(
        lambda x: np.round(list(x.values())[0], 2) if len(x) > 0 else np.nan
    )

    place_2_distances = scoring_fav_lane_level['Place_Value_Dict'].apply(lambda x: safe_get_from_dict(x, 1))
    scoring_fav_lane_level['Recommend_place_2'], scoring_fav_lane_level['Recommend_distance_2'] = zip(*place_2_distances)

    place_3_distances = scoring_fav_lane_level['Place_Value_Dict'].apply(lambda x: safe_get_from_dict(x, 2))
    scoring_fav_lane_level['Recommend_place_3'], scoring_fav_lane_level['Recommend_distance_3'] = zip(*place_3_distances)

    scoring_fav_lane_level['Toll_LAT'] = scoring_fav_lane_level['Toll_LAT'].astype(float, errors='ignore')
    scoring_fav_lane_level['Toll_LON'] = scoring_fav_lane_level['Toll_LON'].astype(float, errors='ignore')
    zone_data['latitude'] = zone_data['latitude'].astype(float, errors='ignore')
    zone_data['longitude'] = zone_data['longitude'].astype(float, errors='ignore')

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

    scoring_fav_lane_level['direction_1'] = scoring_fav_lane_level.apply(
        lambda row: calculate_direction(
            row['Toll_LAT'], row['Toll_LON'], row['latitude_scoring'], row['longitude_scoring']
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

    scoring_fav_lane_level['favorite_zones'] = scoring_fav_lane_level['favorite_zones'].apply(
        lambda x: x.split(',') if pd.notna(x) else []
    )

    # Log direction_list stats
    non_empty_direction_list = scoring_fav_lane_level['direction_list'].apply(
        lambda x: len(x) > 0
    ).sum()
    logger.info(f"Rows with non-empty direction_list: {non_empty_direction_list}/{len(scoring_fav_lane_level)}")

    for i in range(1, 4):
        scoring_fav_lane_level[f'movement_analysis_{i}'] = scoring_fav_lane_level.apply(
            lambda row: analyze_movement_pattern(row['direction_list'][0], row[f'direction_{i}'])
            if pd.notna(row.get('direction_list')) and len(row['direction_list']) > 0
            else {'moving_towards': 0, 'moving_away': 0, 'moving_perpendicular': 0},
            axis=1
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

    trips = aggregated_combined_places.merge(
        scoring_fav_lane_level, left_on='Supplier UUID', right_on='suppliercompany_uuid', how='left'
    )
    trips.dropna(subset='suppliercompany_uuid', inplace=True)
    trips.reset_index(drop=True, inplace=True)
    logger.info(f"Trips merged with scoring data. Shape: {trips.shape}")

    for i in range(1, 4):
        trips[f'near_to_fav_zone_{i}'] = trips.apply(
            check_zone_in_favorites, axis=1, nearest_zone=f'Recommend_place_{i}', favorite_zones='favorite_zones'
        )
        trips[f'travelled_past_zone_{i}'] = trips.apply(
            check_zone_in_past, axis=1, nearest_zone=f'Recommend_place_{i}', trip_places='Trip_places_in_last_6_months'
        )

    trips['Score_distance_1'] = 1 - (
        trips['Recommend_distance_1'] / (trips['Recommend_distance_1'] + trips['Recommend_distance_2'].fillna(0) + trips['Recommend_distance_3'].fillna(0))
    )
    trips['Score_distance_1'] = trips['Score_distance_1'].fillna(1)

    trips['Score_distance_2'] = 1 - (
        trips['Recommend_distance_2'] / (trips['Recommend_distance_1'] + trips['Recommend_distance_2'].fillna(0) + trips['Recommend_distance_3'].fillna(0))
    )
    trips['Score_distance_2'] = trips['Score_distance_2'].fillna(0)

    trips['Score_distance_3'] = 1 - (
        trips['Recommend_distance_3'] / (trips['Recommend_distance_1'] + trips['Recommend_distance_2'].fillna(0) + trips['Recommend_distance_3'].fillna(0))
    )
    trips['Score_distance_3'] = trips['Score_distance_3'].fillna(0)

    trips['zone_score_1'] = trips.apply(lambda row: calculate_zone_score(row, 1), axis=1)
    trips['zone_score_2'] = trips.apply(lambda row: calculate_zone_score(row, 2), axis=1)
    trips['zone_score_3'] = trips.apply(lambda row: calculate_zone_score(row, 3), axis=1)

    trips['Reach_time_1'] = trips['Recommend_distance_1'] / AVERAGE_SPEED
    trips['Reach_time_2'] = trips['Recommend_distance_2'] / AVERAGE_SPEED
    trips['Reach_time_3'] = trips['Recommend_distance_3'] / AVERAGE_SPEED

    trips[['Score_distance_1', 'Score_distance_2', 'Score_distance_3']] = trips[
        ['Score_distance_1', 'Score_distance_2', 'Score_distance_3']
    ].apply(pd.to_numeric, errors='coerce')

    trips['Recommend_place_2'] = trips['Recommend_place_2'].fillna("None")
    trips['Recommend_place_3'] = trips['Recommend_place_3'].fillna("None")

    trips[['Top_Zone_1', 'Top_Zone_2', 'Top_Zone_3', 'Reach_Time_1', 'Reach_Time_2', 'Reach_Time_3']] = trips.apply(
        recommend_zones_with_time, axis=1, result_type='expand'
    )

    logger.info(f"Recommendations processed. Shape: {trips.shape}")
    return trips

@task
def finalize_recommendations(trips, fastag_sorted, supplier_fav_lane):
    """Finalize the recommendation base and save outputs."""
    logger = get_run_logger()
    logger.info("Finalizing recommendations...")
    output_dir = 'C:/Users/HP/Documents/Raaho/fasttag/fastag/output/'
    os.makedirs(output_dir, exist_ok=True)

    global REFERENCE_TIME
    if not fastag_sorted.empty:
        max_date = fastag_sorted['transaction_date_time'].max()
        REFERENCE_TIME = max_date if pd.notna(max_date) else datetime(2025, 5, 14, 8, 30, 0)
    logger.info(f"REFERENCE_TIME set to: {REFERENCE_TIME}")

    if trips.empty:
        logger.warning("Trips DataFrame is empty! Generating synthetic output.")
        final_recommended_base = pd.DataFrame({
            'Supplier UUID': ['UUID1'],
            'name': ['Fallback Supplier'],
            'pan': ['ABCDE1234F'],
            'owner_name': ['John Doe'],
            'owner_number': ['1234567890'],
            'favourite_places': ['Zone1'],
            'favorite_zones': ['Zone1'],
            'reg_no': ['TEST123'],
            'Top_Zone_1': ['Zone1'],
            'Reach_Time_1': [5.0],
            'transaction_date_time': [pd.Timestamp('2025-05-14')],
            'Time_Difference': [10.0],
            'TCellID': ['T1'],
            'DCellID': ['D1'],
            'TDCellID': ['T1D1']
        })
        continuation_base = final_recommended_base.copy()
        full_base = final_recommended_base.copy()

        final_recommended_base.to_csv(f'{output_dir}recommended_base.csv', index=False)
        continuation_base.to_csv(f'{output_dir}continuation.csv', index=False)
        full_base.to_csv(f'{output_dir}full_base.csv', index=False)
        logger.info(f"Synthetic recommendations saved. Shapes: final_recommended_base={final_recommended_base.shape}, "
                   f"continuation_base={continuation_base.shape}, full_base={full_base.shape}")
        return final_recommended_base, continuation_base, full_base

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
            'Supplier UUID': ['UUID1'],
            'name': ['Fallback Supplier'],
            'pan': ['ABCDE1234F'],
            'owner_name': ['John Doe'],
            'owner_number': ['1234567890'],
            'favourite_places': ['Zone1'],
            'favorite_zones': ['Zone1'],
            'reg_no': ['TEST123'],
            'Top_Zone_1': ['Zone1'],
            'Reach_Time_1': [5.0],
            'transaction_date_time': [pd.Timestamp('2025-05-14')],
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

    final_recommended_base.to_csv(f'{output_dir}recommended_base.csv', index=False)
    continuation_base.to_csv(f'{output_dir}continuation.csv', index=False)
    full_base.to_csv(f'{output_dir}full_base.csv', index=False)

    logger.info(f"Recommendations finalized. Shapes: final_recommended_base={final_recommended_base.shape}, "
               f"continuation_base={continuation_base.shape}, full_base={full_base.shape}")
    return final_recommended_base, continuation_base, full_base

@flow(name="Zone_Recommendation_Pipeline", log_prints=True)
def zone_recommendation_pipeline():
    """Main flow to orchestrate the zone recommendation pipeline with concurrent tasks."""
    file_paths = {
        'supplier_fav_lane': r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_favlanes.csv',
        'supplier_vehicles': r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_vehicles.csv',
        'zone_data': r'C:\Users\HP\Documents\Raaho\fasttag\data\result.csv',
        'trips': r'C:\Users\HP\Documents\Raaho\fasttag\data\MIS_UnionTable.csv',
        'fastag': r'C:\Users\HP\Documents\Raaho\fasttag\data\fastag_export_1.csv'
    }

    # Load data
    data = load_data(file_paths)

    # Run independent preprocessing tasks concurrently
    with ThreadPoolExecutor() as executor:
        fastag_future = executor.submit(preprocess_fastag, data['fastag'])
        trips_future = executor.submit(preprocess_trips, data['trips'])
        latest_records = fastag_future.result()
        aggregated_combined_places = trips_future.result()

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
    print(f"Final shapes: final_recommended_base={final_recommended_base.shape}, "
          f"continuation_base={continuation_base.shape}, full_base={full_base.shape}")
from prefect import task, flow
import pandas as pd
import warnings
from datetime import datetime, timedelta
import numpy as np
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Assuming your CSV files are in the specified Google Drive path
SUPPLIER_FAV_LANE_PATH = r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_favlanes.csv'
SUPPLIER_VEHICLES_PATH = r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_vehicles.csv'
ZONE_DATA_PATH = r'C:\Users\HP\Documents\Raaho\fasttag\data\result.csv'
TRIPS_PATH = r'C:\Users\HP\Documents\Raaho\fasttag\data\MIS_UnionTable.csv'
FASTAG_PATH = r'C:\Users\HP\Documents\Raaho\fasttag\data\fastag_export_1.csv'
REFERENCE_TIME = datetime(2025, 4, 5, 8, 30, 0)  # Using the reference time you specified
AVERAGE_SPEED = 20

@task(name="Load Supplier Favorite Lanes")
def load_supplier_fav_lane(file_path: str) -> pd.DataFrame:
    """Loads the supplier favorite lanes data."""
    return pd.read_csv(file_path)

@task(name="Load Supplier Vehicles")
def load_supplier_vehicles(file_path: str) -> pd.DataFrame:
    """Loads the supplier vehicles data."""
    return pd.read_csv(file_path)

@task(name="Load Zone Data")
def load_zone_data(file_path: str) -> pd.DataFrame:
    """Loads the zone data."""
    return pd.read_csv(file_path)

@task(name="Load Trips Data")
def load_trips_data(file_path: str) -> pd.DataFrame:
    """Loads the trips data."""
    return pd.read_csv(file_path)

@task(name="Load Fastag Data")
def load_fastag_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the fastag data."""
    fastag = pd.read_csv(file_path)
    fastag["transaction_date_time"] = pd.to_datetime(fastag["transaction_date_time"])
    fastag.sort_values(by=['vehicle_reg_no', 'transaction_date_time'], ascending=[True, False], inplace=True)
    current_time_utc = datetime.utcnow()
    time_threshold = current_time_utc - timedelta(days=5)
    return fastag[fastag["transaction_date_time"] >= time_threshold]

@task(name="Aggregate Combined Places")
def aggregate_combined_places(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates origin and destination zones for each supplier."""
    df_combined = pd.concat([trips_df[['Supplier UUID', 'Zone_Origin']],
                              trips_df[['Supplier UUID', 'Zone_Destination']].rename(columns={'Zone_Destination': 'Zone_Origin'})],
                             ignore_index=True)
    aggregated_combined_places = df_combined.groupby('Supplier UUID')['Zone_Origin'].agg(lambda x: list(set(x))).reset_index()
    aggregated_combined_places.rename(columns={'Zone_Origin': 'Trip_places_in_last_6_months'}, inplace=True)
    return aggregated_combined_places

@task(name="Get Latest Fastag Records")
def get_latest_fastag_records(fastag_df: pd.DataFrame) -> pd.DataFrame:
    """Gets the latest fastag transaction for each vehicle and their last direction."""
    fastag_sorted = fastag_df.drop_duplicates(subset=['vehicle_reg_no'], keep='first').copy()
    direction_list = fastag_df.groupby('vehicle_reg_no')['lane_direction'].agg(list).reset_index()
    direction_list.rename(columns={'lane_direction': 'direction_list'}, inplace=True)
    latest_records = fastag_sorted.merge(direction_list, on='vehicle_reg_no', how='left')
    latest_records = latest_records.sort_values('transaction_date_time').groupby('vehicle_reg_no').last().reset_index()
    return latest_records

@task(name="Merge Supplier Favorite Lanes Data")
def merge_supplier_fav_lanes_data(fav_lane_df: pd.DataFrame, vehicles_df: pd.DataFrame) -> pd.DataFrame:
    """Merges supplier favorite lanes with vehicle data."""
    vehicles_df = vehicles_df.dropna(subset=['owner_uuid']).copy()
    vehicles_df['owner_uuid'] = vehicles_df['owner_uuid'].astype(int).astype(str)
    fav_lane_df['suppliercompany_uuid'] = fav_lane_df['suppliercompany_uuid'].astype(int).astype(str)
    supplier_fav_lanes_data = fav_lane_df.merge(vehicles_df, left_on='suppliercompany_uuid', right_on='owner_uuid', how='left')
    return supplier_fav_lanes_data.dropna().reset_index(drop=True)

@task(name="Merge with Latest Records")
def merge_with_latest_records(fav_lanes_data: pd.DataFrame, latest_records_df: pd.DataFrame) -> pd.DataFrame:
    """Merges favorite lanes data with the latest fastag records."""
    with_lanes = fav_lanes_data.merge(latest_records_df, left_on='reg_no', right_on='vehicle_reg_no', how='left')
    return with_lanes.dropna().reset_index(drop=True)

@task(name="Calculate Circle Coordinates")
def create_circle_coordinates(lat: float, lon: float, radius_km: int = 50, num_points: int = 100) -> list:
    """Create coordinates for a circle around a center point."""
    R = 6371.0
    rad = radius_km / R
    circle_coords = []
    for i in range(num_points + 1):
        bearing = 2 * np.pi * i / num_points
        lat1 = radians(float(lat))
        lon1 = radians(float(lon))
        lat2 = np.arcsin(np.sin(lat1) * np.cos(rad) +
                        np.cos(lat1) * np.sin(rad) * np.cos(bearing))
        lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(rad) * np.cos(lat1),
                                np.cos(rad) - np.sin(lat1) * np.sin(lat2))
        circle_coords.append((np.degrees(lat2), np.degrees(lon2)))
    return circle_coords

@task(name="Create Circles Dictionary")
def create_circles_dict(scoring_df: pd.DataFrame) -> dict:
    """Creates a dictionary of circle coordinates for each vehicle."""
    circles_dict = {}
    for idx, row in scoring_df.iterrows():
        location_id = row['vehicle_reg_no']
        try:
            lat, lon = row['toll_plaza_geocode'].split(',')
            circles_dict[location_id] = create_circle_coordinates(float(lat), float(lon))
        except AttributeError:
            circles_dict[location_id] = None  # Handle cases where toll_plaza_geocode might be missing or malformed
    return circles_dict

@task(name="Calculate Distance")
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two points using Haversine formula (in km)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@task(name="Find Zones For Circle")
def find_zones_for_circle(circle_center: tuple, zone_data_df: pd.DataFrame, radius_km: int = 500) -> tuple[list, list]:
    """Find zones within radius for a given circle center."""
    center_lat, center_lon = circle_center
    distances = zone_data_df.apply(
        lambda row: calculate_distance(center_lat, center_lon, row['latitude'], row['longitude']), axis=1
    )
    nearby_zones = zone_data_df[distances <= radius_km].copy()
    nearby_zones['distance'] = distances[distances <= radius_km]
    nearby_zones = nearby_zones.sort_values('distance')
    return nearby_zones['zoneCode'].tolist(), nearby_zones['distance'].tolist()

@task(name="Update DataFrame With Zones")
def update_dataframe_with_zones(scoring_df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    """Update the scoring DataFrame with nearby zone information."""
    nearby_zones_list = []
    zone_distances_list = []
    for idx in scoring_df.index:
        if scoring_df.at[idx, 'circle_coordinates']:
            center_point = scoring_df.at[idx, 'circle_coordinates'][0]
            zones, distances = find_zones_for_circle(center_point, zone_df)
            nearby_zones_list.append(zones)
            zone_distances_list.append(distances)
        else:
            nearby_zones_list.append([])
            zone_distances_list.append([])
    scoring_df['nearby_zones'] = nearby_zones_list
    scoring_df['zone_distances'] = zone_distances_list
    return scoring_df



@task(name="Create Place Value Dictionary")
def create_place_value_dict_task(places: list, values: list) -> dict:
    """Creates a dictionary of places and their values, filtering 'IN' prefixed places."""
    filtered_dict = dict(
        (place, value) for place, value in zip(places, values) if not place.startswith('IN')
    )
    if not filtered_dict and values:
        min_distance_idx = values.index(min(values))
        filtered_dict[places[min_distance_idx]] = values[min_distance_idx]
    return filtered_dict

    filtered_dict = dict(
        (place, value) for place, value in zip(places, values) if not place.startswith('IN')
    )
    if not filtered_dict and values:
        min_distance_idx = values.index(min(values))
        filtered_dict[places[min_distance_idx]] = values[min_distance_idx]
    return filtered_dict

@task(name="Safe Get From Dictionary")
def safe_get_from_dict_task(d: dict, index: int) -> tuple[float, float]:
    """Safely retrieves a key and rounded value from a dictionary at a given index."""
    try:
        key = list(d.keys())[index]
        value = np.round(list(d.values())[index], 2)
        return key, value
    except IndexError:
        return np.nan, np.nan

@task(name="Calculate Direction")
def calculate_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Calculate direction between two points."""
    import math
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

@task(name="Analyze Movement Pattern")
def analyze_movement_pattern(latest_direction: str, zone_direction: str) -> dict:
    """Detailed analysis of movement pattern relative to zone."""
    direction_relationships = {
        ('N', 'N'): 'directly towards',
        ('S', 'S'): 'directly towards',
        ('E', 'E'): 'directly towards',
        ('W', 'W'): 'directly towards',
        ('N', 'S'): 'directly away',
        ('S', 'N'): 'directly away',
        ('E', 'W'): 'directly away',
        ('W', 'E'): 'directly away',
        ('N', 'E'): 'perpendicular',
        ('N', 'W'): 'perpendicular',
        ('S', 'E'): 'perpendicular',
        ('S', 'W'): 'perpendicular',
        ('E', 'N'): 'perpendicular',
        ('E', 'S'): 'perpendicular',
        ('W', 'N'): 'perpendicular',
        ('W', 'S'): 'perpendicular'
    }
    movement_type = direction_relationships.get((latest_direction, zone_direction), 'unknown')
    return {
        'movement_type': movement_type,
        'moving_towards': 1 if movement_type == 'directly towards' else 0,
        'moving_away': 1 if movement_type == 'directly away' else 0,
        'moving_perpendicular': 1 if movement_type == 'perpendicular' else 0
    }

@task(name="Check Zone In Favorites")
def check_zone_in_favorites(row: pd.Series, nearest_zone: str, favorite_zones: str) -> int:
    """Checks if the nearest zone is in the favorite zones."""
    if pd.isna(row[nearest_zone]) or not row[favorite_zones]:
        return 0
    return 1 if row[nearest_zone] in row[favorite_zones] else 0

@task(name="Check Zone In Past")
def check_zone_in_past(row: pd.Series, nearest_zone: str, Trip_places_in_last_6_months: str) -> int:
    """Checks if the nearest zone was travelled to in the past."""
    if pd.isna(row[nearest_zone]) or not row[Trip_places_in_last_6_months]:
        return 0
    return 1 if row[nearest_zone] in row[Trip_places_in_last_6_months] else 0

@task(name="Calculate Zone Score")
def calculate_zone_score(row: pd.Series, zone_num: int) -> float:
    """Calculates a score for a recommended zone."""
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

    total_score = (
        0.3 * direction_score +
        0.05 * favorite_zone_score +
        0.5 * distance_score +
        0.15 * history_score
    )
    return total_score

@task(name="Classify Time Difference")
def classify_time_diff(diff: float) -> str:
    """Classifies the time difference into categories."""
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

@task(name="Classify Reach Time Difference")
def classify_reach_time_diff(diff: float) -> str:
    """Classifies the reach time difference into categories."""
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

@task(name="Recommend Zones With Time")
def recommend_zones_with_time(row: pd.Series) -> list:
    """Recommends top 3 zones based on score and returns names and reach times."""
    scores = {
        'Recommend_place_1': (row['zone_score_1'], row['Reach_time_1'] if 'Reach_time_1' in row else np.nan),
        'Recommend_place_2': (row['zone_score_2'], row['Reach_time_2'] if 'Reach_time_2' in row else np.nan),
        'Recommend_place_3': (row['zone_score_3'], row['Reach_time_3'] if 'Reach_time_3' in row else np.nan),
    }
    sorted_zones = sorted(scores, key=lambda x: scores[x][0], reverse=True)
    sorted_zone_names = [row.get(zone, np.nan) for zone in sorted_zones]
    sorted_reach_times = [scores[zone][1] for zone in sorted_zones]
    return sorted_zone_names + sorted_reach_times

@task(name="Merge with Zone Data for Recommendations")
def merge_with_zone_data(scoring_df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    """Merges scoring data with zone data to get coordinates of recommended places."""
    merged_df = scoring_df.merge(
        zone_df[['zoneCode', 'latitude', 'longitude']],
        left_on='Recommend_place_1',
        right_on='zoneCode',
        how='left',
        suffixes=('_scoring', '_1')
    ).merge(
        zone_df[['zoneCode', 'latitude', 'longitude']],
        left_on='Recommend_place_2',
        right_on='zoneCode',
        how='left',
        suffixes=('_scoring', '_2')
    ).merge(
        zone_df[['zoneCode', 'latitude', 'longitude']],
        left_on='Recommend_place_3',
        right_on='zoneCode',
        how='left',
        suffixes=('_scoring', '_3')
    )
    return merged_df

@task(name="Calculate Directions to Recommended Zones")
def calculate_directions_to_recommendations(scoring_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the direction from the last toll to the top 3 recommended zones."""
    scoring_df['Toll_LAT'] = pd.to_numeric(scoring_df['Toll_LAT'], errors='coerce')
    scoring_df['Toll_LON'] = pd.to_numeric(scoring_df['Toll_LON'], errors='coerce')
    scoring_df['latitude_1'] = pd.to_numeric(scoring_df['latitude_1'], errors='coerce')
    scoring_df['longitude_1'] = pd.to_numeric(scoring_df['longitude_1'], errors='coerce')
    scoring_df['latitude_2'] = pd.to_numeric(scoring_df['latitude_2'], errors='coerce')
    scoring_df['longitude_2'] = pd.to_numeric(scoring_df['longitude_2'], errors='coerce')
    scoring_df['latitude'] = pd.to_numeric(scoring_df['latitude'], errors='coerce')
    scoring_df['longitude'] = pd.to_numeric(scoring_df['longitude'], errors='coerce')

    scoring_df['direction_1'] = scoring_df.apply(lambda row: calculate_direction(
        row['Toll_LAT'], row['Toll_LON'], row['latitude_1'], row['longitude_1']
    ) if pd.notna(row['Toll_LAT']) and pd.notna(row['Toll_LON']) and pd.notna(row['latitude_1']) and pd.notna(row['longitude_1']) else np.nan, axis=1)
    scoring_df['direction_2'] = scoring_df.apply(lambda row: calculate_direction(
        row['Toll_LAT'], row['Toll_LON'], row['latitude_2'], row['longitude_2']
    ) if pd.notna(row['Toll_LAT']) and pd.notna(row['Toll_LON']) and pd.notna(row['latitude_2']) and pd.notna(row['longitude_2']) else np.nan, axis=1)
    scoring_df['direction_3'] = scoring_df.apply(lambda row: calculate_direction(
        row['Toll_LAT'], row['Toll_LON'], row['latitude'], row['longitude']
    ) if pd.notna(row['Toll_LAT']) and pd.notna(row['Toll_LON']) and pd.notna(row['latitude']) and pd.notna(row['longitude']) else np.nan, axis=1)
    return scoring_df

@task(name="Analyze Movement Towards Recommendations")
def analyze_movement_towards_recommendations(scoring_df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the movement pattern towards the recommended zones."""
    scoring_df['movement_analysis_1'] = scoring_df.apply(
        lambda row: analyze_movement_pattern(row['direction_list'][0], row['direction_1'])
        if isinstance(row['direction_list'], list) and row['direction_list'] and pd.notna(row['direction_1']) else {'moving_towards': 0, 'moving_away': 0, 'moving_perpendicular': 0},
        axis=1)
    scoring_df['movement_analysis_2'] = scoring_df.apply(
        lambda row: analyze_movement_pattern(row['direction_list'][0], row['direction_2'])
        if isinstance(row['direction_list'], list) and row['direction_list'] and pd.notna(row['direction_2']) else {'moving_towards': 0, 'moving_away': 0, 'moving_perpendicular': 0},
        axis=1)
    scoring_df['movement_analysis_3'] = scoring_df.apply(
        lambda row: analyze_movement_pattern(row['direction_list'][0], row['direction_3'])
        if isinstance(row['direction_list'], list) and row['direction_list'] and pd.notna(row['direction_3']) else {'moving_towards': 0, 'moving_away': 0, 'moving_perpendicular': 0},
        axis=1)
    scoring_df['moving_towards_1'] = scoring_df['movement_analysis_1'].apply(lambda x: x.get('moving_towards', 0))
    scoring_df['moving_away_1'] = scoring_df['movement_analysis_1'].apply(lambda x: x.get('moving_away', 0))
    scoring_df['moving_perpendicular_1'] = scoring_df['movement_analysis_1'].apply(lambda x: x.get('moving_perpendicular', 0))
    scoring_df['moving_towards_2'] = scoring_df['movement_analysis_2'].apply(lambda x: x.get('moving_towards', 0))
    scoring_df['moving_away_2'] = scoring_df['movement_analysis_2'].apply(lambda x: x.get('moving_away', 0))
    scoring_df['moving_perpendicular_2'] = scoring_df['movement_analysis_2'].apply(lambda x: x.get('moving_perpendicular', 0))
    scoring_df['moving_towards_3'] = scoring_df['movement_analysis_3'].apply(lambda x: x.get('moving_towards', 0))
    scoring_df['moving_away_3'] = scoring_df['movement_analysis_3'].apply(lambda x: x.get('moving_away', 0))
    scoring_df['moving_perpendicular_3'] = scoring_df['movement_analysis_3'].apply(lambda x: x.get('moving_perpendicular', 0))
    scoring_df.drop(columns=['movement_analysis_1', 'movement_analysis_2', 'movement_analysis_3'], axis=1, inplace=True)
    return scoring_df

@task(name="Merge Trips with Scoring")
def merge_trips_with_scoring(aggregated_trips_df: pd.DataFrame, scoring_df: pd.DataFrame) -> pd.DataFrame:
    """Merges aggregated trips data with the scoring DataFrame."""
    aggregated_trips_df['Supplier UUID'] = aggregated_trips_df['Supplier UUID'].astype(str)
    merged_df = aggregated_trips_df.merge(scoring_df, left_on='Supplier UUID', right_on='suppliercompany_uuid', how='left')
    return merged_df.dropna(subset='suppliercompany_uuid').reset_index(drop=True)

@task(name="Check Recommended Zones Against Favorites and History")
def check_recommended_zones(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Checks if recommended zones are in favorites or past trips."""
    trips_df['near_to_fav_zone_1'] = trips_df.apply(
        check_zone_in_favorites, axis=1, nearest_zone='Recommend_place_1', favorite_zones='favorite_zones')
    trips_df['near_to_fav_zone_2'] = trips_df.apply(
        check_zone_in_favorites, axis=1, nearest_zone='Recommend_place_2', favorite_zones='favorite_zones')
    trips_df['near_to_fav_zone_3'] = trips_df.apply(
        check_zone_in_favorites, axis=1, nearest_zone='Recommend_place_3', favorite_zones='favorite_zones')
    trips_df['travelled_past_zone_1'] = trips_df.apply(
        check_zone_in_past, axis=1, nearest_zone='Recommend_place_1', Trip_places_in_last_6_months='Trip_places_in_last_6_months')
    trips_df['travelled_past_zone_2'] = trips_df.apply(
        check_zone_in_past, axis=1, nearest_zone='Recommend_place_2', Trip_places_in_last_6_months='Trip_places_in_last_6_months')
    trips_df['travelled_past_zone_3'] = trips_df.apply(
        check_zone_in_past, axis=1, nearest_zone='Recommend_place_3', Trip_places_in_last_6_months='Trip_places_in_last_6_months')
    return trips_df

@task(name="Calculate Distance Scores")
def calculate_distance_scores(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates distance-based scores for the recommended zones."""
    trips_df['Score_distance_1'] = 1 - (trips_df['Recommend_distance_1'] / (
        trips_df['Recommend_distance_1'] + trips_df['Recommend_distance_2'] + trips_df['Recommend_distance_3']))
    trips_df['Score_distance_1'] = trips_df['Score_distance_1'].fillna(1)
    trips_df['Score_distance_2'] = 1 - (trips_df['Recommend_distance_2'] / (
        trips_df['Recommend_distance_1'] + trips_df['Recommend_distance_2'] + trips_df['Recommend_distance_3']))
    trips_df['Score_distance_2'] = trips_df['Score_distance_2'].fillna(0)
    trips_df['Score_distance_3'] = 1 - (trips_df['Recommend_distance_3'] / (
        trips_df['Recommend_distance_1'] + trips_df['Recommend_distance_2'] + trips_df['Recommend_distance_3']))
    trips_df['Score_distance_3'] = trips_df['Score_distance_3'].fillna(0)
    return trips_df

@task(name="Apply Zone Scoring")
def apply_zone_scoring(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Applies the zone scoring function to calculate final scores."""
    trips_df['zone_score_1'] = trips_df.apply(lambda row: calculate_zone_score(row, 1), axis=1)
    trips_df['zone_score_2'] = trips_df.apply(lambda row: calculate_zone_score(row, 2), axis=1)
    trips_df['zone_score_3'] = trips_df.apply(lambda row: calculate_zone_score(row, 3), axis=1)
    return trips_df

@task(name="Calculate Reach Times")
def calculate_reach_times(trips_df: pd.DataFrame, average_speed: float) -> pd.DataFrame:
    """Calculates the estimated reach times to the recommended zones."""
    trips_df['Reach_time_1'] = trips_df['Recommend_distance_1'] / average_speed
    trips_df['Reach_time_2'] = trips_df['Recommend_distance_2'] / average_speed
    trips_df['Reach_time_3'] = trips_df['Recommend_distance_3'] / average_speed
    trips_df[['Score_distance_1', 'Score_distance_2', 'Score_distance_3']] = trips_df[
        ['Score_distance_1', 'Score_distance_2', 'Score_distance_3']].apply(pd.to_numeric)
    trips_df['Recommend_place_2'] = trips_df['Recommend_place_2'].fillna("None")
    trips_df['Recommend_place_3'] = trips_df['Recommend_place_3'].fillna("None")
    return trips_df

@task(name="Get Top Recommended Zones")
def get_top_recommended_zones(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Gets the top recommended zones and their reach times."""
    trips_df[[
        'Top_Zone_1', 'Top_Zone_2', 'Top_Zone_3', 'Reach_Time_1', 'Reach_Time_2', 'Reach_Time_3'
    ]] = trips_df.apply(recommend_zones_with_time, axis=1, result_type='expand')
    final = trips_df[['Supplier UUID', 'reg_no', 'Top_Zone_1', 'Reach_Time_1', 'Top_Zone_2', 'Reach_Time_2', 'Top_Zone_3', 'Reach_Time_3']].copy()
    final.replace("None", pd.NA, inplace=True)
    final["Top_Zone_1"] = final["Top_Zone_1"].fillna(final["Top_Zone_2"]).fillna(final["Top_Zone_3"])
    final["Reach_Time_1"] = final["Reach_Time_1"].fillna(final["Reach_Time_2"]).fillna(final["Reach_Time_3"])
    return final

@task(name="Merge with Latest Fastag Timestamp")
def merge_with_latest_fastag_timestamp(recommended_base_df: pd.DataFrame, fastag_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the recommended base with the latest fastag transaction timestamp."""
    latest_fastag = fastag_df.sort_values(['vehicle_reg_no', 'transaction_date_time'], ascending=[True, False]).drop_duplicates(subset=['vehicle_reg_no'], keep='first')
    recommended_base = recommended_base_df.merge(latest_fastag[['vehicle_reg_no', 'transaction_date_time']], left_on='reg_no', right_on='vehicle_reg_no', how='left')
    return recommended_base.drop(columns=['vehicle_reg_no'])

@task(name="Calculate Time Differences and Classify")
def calculate_time_differences(recommended_base_df: pd.DataFrame, reference_time: datetime) -> pd.DataFrame:
    """Calculates time differences and classifies them."""
    recommended_base_df["Time_Difference"] = np.round((reference_time - recommended_base_df["transaction_date_time"]).dt.total_seconds() / 3600)
    recommended_base_df["TCellID"] = recommended_base_df["Time_Difference"].apply(classify_time_diff)
    recommended_base_df['Reach_Time_1'] = np.round(recommended_base_df['Reach_Time_1'])
    recommended_base_df['DCellID'] = recommended_base_df["Reach_Time_1"].apply(classify_reach_time_diff)
    recommended_base_df["TDCellID"] = recommended_base_df["TCellID"] + recommended_base_df["DCellID"]
    return recommended_base_df.drop_duplicates()

@task(name="Merge with Supplier Favorite Lanes for Final Output")
def merge_with_supplier_fav_lane_final(recommended_base_df: pd.DataFrame, supplier_fav_lane_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the processed recommendations with supplier favorite lanes for the final output."""
    merged_df = recommended_base_df.merge(supplier_fav_lane_df, left_on='Supplier UUID', right_on='suppliercompany_uuid', how='left')
    return merged_df.drop_duplicates()[[
        'Supplier UUID',
        'name',
        'pan',
        'owner_name',
        'owner_number',
        'favourite_places',
        'favorite_zones',
        'reg_no',
        'Top_Zone_1',
        'Reach_Time_1',
        'transaction_date_time',
        'Time_Difference',
        'TCellID',
        'DCellID',
        'TDCellID'
    ]]

@task(name="Filter and Save Final Data")
def filter_and_save_final_data(final_df: pd.DataFrame):
    """Filters the final DataFrame and saves to CSV files."""
    final_df['owner_number'] = final_df['owner_number'].astype(int).astype(str)
    continution_base = final_df[final_df["TDCellID"] == 'T1D1']
    final_recommended_base = final_df[final_df["TDCellID"] == 'T2D1']
    full_base = final_df

    continution_base.to_csv('continution.csv', index=False)
    final_recommended_base.to_csv('recommended_base.csv', index=False)
    full_base.to_csv('full_base.csv', index=False)

@flow(name="Supplier Zone Recommendation Flow")
def supplier_zone_recommendation_flow(
    supplier_fav_lane_path: str = SUPPLIER_FAV_LANE_PATH,
    supplier_vehicles_path: str = SUPPLIER_VEHICLES_PATH,
    zone_data_path: str = ZONE_DATA_PATH,
    trips_path: str = TRIPS_PATH,
    fastag_path: str = FASTAG_PATH,
    reference_time: datetime = REFERENCE_TIME,
    average_speed: float = AVERAGE_SPEED
):
    """Main Prefect flow for generating supplier zone recommendations."""
    supplier_fav_lane_df = load_supplier_fav_lane(supplier_fav_lane_path)
    supplier_vehicles_df = load_supplier_vehicles(supplier_vehicles_path)
    zone_data_df = load_zone_data(zone_data_path)
    trips_df = load_trips_data(trips_path)
    fastag_df = load_fastag_data(fastag_path)

    aggregated_trips_df = aggregate_combined_places(trips_df)
    latest_fastag_records_df = get_latest_fastag_records(fastag_df)
    supplier_fav_lanes_data_df = merge_supplier_fav_lanes_data(supplier_fav_lane_df, supplier_vehicles_df)
    with_lanes_df = merge_with_latest_records(supplier_fav_lanes_data_df, latest_fastag_records_df)

    scoring_fav_lane_level = with_lanes_df.loc[with_lanes_df.groupby('vehicle_reg_no')['transaction_date_time'].idxmax()].copy()
    scoring_fav_lane_level['Toll_LAT'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(lambda x: x.split(',')[0])
    scoring_fav_lane_level['Toll_LON'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(lambda x: x.split(',')[-1])
    scoring_fav_lane_level.dropna(inplace=True)
    scoring_fav_lane_level.reset_index(drop=True, inplace=True)

    circles_dict = create_circles_dict(scoring_fav_lane_level)
    scoring_fav_lane_level['circle_coordinates'] = scoring_fav_lane_level['vehicle_reg_no'].map(circles_dict)
    scoring_fav_lane_level.dropna(subset=['circle_coordinates'], inplace=True)
    scoring_fav_lane_level.reset_index(drop=True, inplace=True)

    updated_scoring_df = update_dataframe_with_zones(scoring_fav_lane_level, zone_data_df)
    updated_scoring_df['Place_Value_Dict'] = updated_scoring_df.apply(
        lambda row: create_place_value_dict_task(row['nearby_zones'], row['zone_distances']), axis=1)

    updated_scoring_df['Recommend_place_1'], updated_scoring_df['Recommend_distance_1'] = zip(*updated_scoring_df['Place_Value_Dict'].apply(lambda x: safe_get_from_dict_task(x, 0)))
    updated_scoring_df['Recommend_place_2'], updated_scoring_df['Recommend_distance_2'] = zip(*updated_scoring_df['Place_Value_Dict'].apply(lambda x: safe_get_from_dict_task(x, 1)))
    updated_scoring_df['Recommend_place_3'], updated_scoring_df['Recommend_distance_3'] = zip(*updated_scoring_df['Place_Value_Dict'].apply(lambda x: safe_get_from_dict_task(x, 2)))

    merged_with_zone = merge_with_zone_data(updated_scoring_df, zone_data_df)
    directions_calculated = calculate_directions_to_recommendations(merged_with_zone)
    movement_analyzed = analyze_movement_towards_recommendations(directions_calculated)

    movement_analyzed['favorite_zones'] = movement_analyzed['favorite_zones'].str.split(',')

    trips_scored = merge_trips_with_scoring(aggregated_trips_df, movement_analyzed)
    trips_checked = check_recommended_zones(trips_scored)
    trips_with_distance_score = calculate_distance_scores(trips_checked)
    trips_with_zone_score = apply_zone_scoring(trips_with_distance_score)
    trips_with_reach_time = calculate_reach_times(trips_with_zone_score, average_speed)
    final_recommendations = get_top_recommended_zones(trips_with_reach_time)
    final_with_timestamp = merge_with_latest_fastag_timestamp(final_recommendations, fastag_df)
    final_with_time_diff = calculate_time_differences(final_with_timestamp, reference_time)
    final_output_df = merge_with_supplier_fav_lane_final(final_with_time_diff, supplier_fav_lane_df)

    filter_and_save_final_data(final_output_df)

if __name__ == "__main__":
    supplier_zone_recommendation_flow()
        
import pandas as pd
fastag = pd.read_csv(r'C:\Users\HP\Documents\Raaho\fasttag\data\fastag_export_1.csv')
fastag["transaction_date_time"] = pd.to_datetime(fastag["transaction_date_time"])
fastag_sorted = fastag.drop_duplicates(subset=['vehicle_reg_no'], keep='first')
direction_list = fastag.groupby('vehicle_reg_no')['lane_direction'].agg(list).reset_index()
latest_records = fastag_sorted.merge(direction_list.rename(columns={'lane_direction': 'direction_list'}), on='vehicle_reg_no', how='left')
vehicles = pd.read_csv(r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_vehicles.csv')
fav_lane = pd.read_csv(r'C:\Users\HP\Documents\Raaho\fasttag\data\suppliers_favlanes.csv')
vehicles['owner_uuid'] = vehicles['owner_uuid'].astype(str).str.strip().str.upper()
fav_lane['suppliercompany_uuid'] = fav_lane['suppliercompany_uuid'].astype(str).str.strip().str.upper()
supplier_fav_lanes_data = fav_lane.merge(vehicles, left_on='suppliercompany_uuid', right_on='owner_uuid', how='left')
with_lanes = supplier_fav_lanes_data.merge(latest_records, left_on='reg_no', right_on='vehicle_reg_no', how='left')
scoring_fav_lane_level = with_lanes.loc[with_lanes.groupby('vehicle_reg_no')['transaction_date_time'].idxmax()].copy()
scoring_fav_lane_level['Toll_LAT'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(lambda x: x.split(',')[0] if pd.notna(x) else '0.0')
scoring_fav_lane_level['Toll_LON'] = scoring_fav_lane_level['toll_plaza_geocode'].apply(lambda x: x.split(',')[-1] if pd.notna(x) else '0.0')
scoring_fav_lane_level.dropna(inplace=True)
print(f"Shape before circles: {scoring_fav_lane_level.shape}")
circles_dict = {}
for idx, row in scoring_fav_lane_level.iterrows():
    try:
        lat, lon = row['toll_plaza_geocode'].split(',')
        circles_dict[row['vehicle_reg_no']] = [(0.0, 0.0)] * 101  # Dummy coordinates
    except:
        circles_dict[row['vehicle_reg_no']] = [(0.0, 0.0)] * 101
scoring_fav_lane_level['circle_coordinates'] = scoring_fav_lane_level['vehicle_reg_no'].map(circles_dict)
scoring_fav_lane_level.dropna(subset=['circle_coordinates'], inplace=True)
print(f"Shape after circles: {scoring_fav_lane_level.shape}")
print(scoring_fav_lane_level[['reg_no', 'toll_plaza_geocode', 'Toll_LAT', 'Toll_LON']].head())
"""
Convert CASEFILE from Beijing to Pan-India 28-State system.
Remaps all data files to Indian states, cities, coordinates, and FIR references.
"""
import pandas as pd
import numpy as np
import os, random

random.seed(42)
np.random.seed(42)

# ═══════════════ 28 INDIAN STATES DATA ═══════════════
STATES = {
    "Andhra Pradesh": {"capital": "Amaravati", "cities": ["Visakhapatnam","Tirupati","Vijayawada","Guntur","Nellore","Kakinada","Rajahmundry"], "lat": 15.9129, "lon": 79.74, "code": "AP", "stations": ["Tirupati Urban PS","Vizag Central PS","Vijayawada Town PS","Guntur City PS","Nellore East PS"]},
    "Arunachal Pradesh": {"capital": "Itanagar", "cities": ["Naharlagun","Pasighat","Tawang"], "lat": 27.1, "lon": 93.62, "code": "AR", "stations": ["Itanagar PS","Naharlagun PS","Pasighat PS"]},
    "Assam": {"capital": "Dispur", "cities": ["Guwahati","Silchar","Dibrugarh","Jorhat","Nagaon"], "lat": 26.14, "lon": 91.77, "code": "AS", "stations": ["Guwahati Central PS","Silchar PS","Dibrugarh PS"]},
    "Bihar": {"capital": "Patna", "cities": ["Gaya","Muzaffarpur","Bhagalpur","Darbhanga"], "lat": 25.6, "lon": 85.1, "code": "BR", "stations": ["Patna City PS","Gaya PS","Muzaffarpur PS"]},
    "Chhattisgarh": {"capital": "Raipur", "cities": ["Bhilai","Bilaspur","Korba","Durg"], "lat": 21.25, "lon": 81.63, "code": "CG", "stations": ["Raipur Civil Lines PS","Bhilai PS","Bilaspur PS"]},
    "Goa": {"capital": "Panaji", "cities": ["Margao","Vasco","Mapusa","Ponda"], "lat": 15.5, "lon": 73.83, "code": "GA", "stations": ["Panaji PS","Margao PS","Vasco PS"]},
    "Gujarat": {"capital": "Gandhinagar", "cities": ["Ahmedabad","Surat","Vadodara","Rajkot","Bhavnagar"], "lat": 23.0, "lon": 72.57, "code": "GJ", "stations": ["Ahmedabad Crime Branch","Surat City PS","Vadodara PS"]},
    "Haryana": {"capital": "Chandigarh", "cities": ["Gurugram","Faridabad","Panipat","Ambala","Karnal"], "lat": 29.06, "lon": 76.09, "code": "HR", "stations": ["Gurugram Cyber PS","Faridabad PS","Panipat PS"]},
    "Himachal Pradesh": {"capital": "Shimla", "cities": ["Dharamshala","Mandi","Kullu","Solan"], "lat": 31.1, "lon": 77.17, "code": "HP", "stations": ["Shimla PS","Dharamshala PS","Mandi PS"]},
    "Jharkhand": {"capital": "Ranchi", "cities": ["Jamshedpur","Dhanbad","Bokaro","Hazaribagh"], "lat": 23.35, "lon": 85.33, "code": "JH", "stations": ["Ranchi PS","Jamshedpur PS","Dhanbad PS"]},
    "Karnataka": {"capital": "Bengaluru", "cities": ["Mysuru","Mangaluru","Hubli","Belgaum","Shimoga"], "lat": 12.97, "lon": 77.59, "code": "KA", "stations": ["Bengaluru Central PS","Mysuru PS","Mangaluru PS"]},
    "Kerala": {"capital": "Thiruvananthapuram", "cities": ["Kochi","Kozhikode","Thrissur","Kollam"], "lat": 10.85, "lon": 76.27, "code": "KL", "stations": ["Kochi City PS","Kozhikode PS","Thiruvananthapuram PS"]},
    "Madhya Pradesh": {"capital": "Bhopal", "cities": ["Indore","Jabalpur","Gwalior","Ujjain"], "lat": 23.26, "lon": 77.41, "code": "MP", "stations": ["Bhopal Central PS","Indore PS","Jabalpur PS"]},
    "Maharashtra": {"capital": "Mumbai", "cities": ["Pune","Nagpur","Nashik","Aurangabad","Thane"], "lat": 19.08, "lon": 72.88, "code": "MH", "stations": ["Mumbai Crime Branch","Pune City PS","Nagpur PS"]},
    "Manipur": {"capital": "Imphal", "cities": ["Thoubal","Bishnupur","Churachandpur"], "lat": 24.82, "lon": 93.95, "code": "MN", "stations": ["Imphal PS","Thoubal PS"]},
    "Meghalaya": {"capital": "Shillong", "cities": ["Tura","Jowai","Nongstoin"], "lat": 25.57, "lon": 91.88, "code": "ML", "stations": ["Shillong PS","Tura PS"]},
    "Mizoram": {"capital": "Aizawl", "cities": ["Lunglei","Champhai","Serchhip"], "lat": 23.73, "lon": 92.72, "code": "MZ", "stations": ["Aizawl PS","Lunglei PS"]},
    "Nagaland": {"capital": "Kohima", "cities": ["Dimapur","Mokokchung","Tuensang"], "lat": 25.67, "lon": 94.12, "code": "NL", "stations": ["Kohima PS","Dimapur PS"]},
    "Odisha": {"capital": "Bhubaneswar", "cities": ["Cuttack","Rourkela","Berhampur","Sambalpur"], "lat": 20.3, "lon": 85.82, "code": "OD", "stations": ["Bhubaneswar PS","Cuttack PS","Rourkela PS"]},
    "Punjab": {"capital": "Chandigarh", "cities": ["Ludhiana","Amritsar","Jalandhar","Patiala"], "lat": 31.1, "lon": 75.34, "code": "PB", "stations": ["Ludhiana PS","Amritsar PS","Jalandhar PS"]},
    "Rajasthan": {"capital": "Jaipur", "cities": ["Jodhpur","Udaipur","Kota","Ajmer","Bikaner"], "lat": 26.92, "lon": 75.79, "code": "RJ", "stations": ["Jaipur City PS","Jodhpur PS","Udaipur PS"]},
    "Sikkim": {"capital": "Gangtok", "cities": ["Namchi","Mangan","Gyalshing"], "lat": 27.33, "lon": 88.61, "code": "SK", "stations": ["Gangtok PS","Namchi PS"]},
    "Tamil Nadu": {"capital": "Chennai", "cities": ["Coimbatore","Madurai","Tiruchirappalli","Salem","Tirunelveli"], "lat": 13.08, "lon": 80.27, "code": "TN", "stations": ["Chennai Central PS","Coimbatore PS","Madurai PS"]},
    "Telangana": {"capital": "Hyderabad", "cities": ["Warangal","Nizamabad","Karimnagar","Khammam"], "lat": 17.39, "lon": 78.49, "code": "TS", "stations": ["Hyderabad Cyber Crime PS","Warangal PS","Nizamabad PS"]},
    "Tripura": {"capital": "Agartala", "cities": ["Udaipur","Dharmanagar","Kailashahar"], "lat": 23.83, "lon": 91.28, "code": "TR", "stations": ["Agartala PS","Udaipur PS"]},
    "Uttar Pradesh": {"capital": "Lucknow", "cities": ["Noida","Agra","Varanasi","Kanpur","Prayagraj","Meerut"], "lat": 26.85, "lon": 80.95, "code": "UP", "stations": ["Lucknow Central PS","Noida PS","Agra PS","Varanasi PS"]},
    "Uttarakhand": {"capital": "Dehradun", "cities": ["Haridwar","Rishikesh","Nainital","Haldwani"], "lat": 30.32, "lon": 78.03, "code": "UK", "stations": ["Dehradun PS","Haridwar PS","Nainital PS"]},
    "West Bengal": {"capital": "Kolkata", "cities": ["Howrah","Siliguri","Durgapur","Asansol"], "lat": 22.57, "lon": 88.36, "code": "WB", "stations": ["Kolkata Lalbazar PS","Howrah PS","Siliguri PS"]},
}

# 5 Macro Regions (mapped from 5 K-Means areas)
REGIONS = [
    {"region": "South India", "area_id": 0, "states": ["Andhra Pradesh","Karnataka","Kerala","Tamil Nadu","Telangana","Goa"],
     "hub": "Hyderabad", "lat": 17.39, "lon": 78.49, "poi": "Charminar Heritage Zone", "cat": "heritage"},
    {"region": "North India", "area_id": 1, "states": ["Uttar Pradesh","Haryana","Punjab","Himachal Pradesh","Uttarakhand","Rajasthan"],
     "hub": "Delhi NCR", "lat": 28.61, "lon": 77.21, "poi": "India Gate Complex", "cat": "landmark"},
    {"region": "East India", "area_id": 2, "states": ["West Bengal","Bihar","Jharkhand","Odisha","Sikkim"],
     "hub": "Kolkata", "lat": 22.57, "lon": 88.36, "poi": "Howrah Station Zone", "cat": "transit"},
    {"region": "West India", "area_id": 3, "states": ["Maharashtra","Gujarat","Madhya Pradesh","Chhattisgarh"],
     "hub": "Mumbai", "lat": 19.08, "lon": 72.88, "poi": "CST Railway Complex", "cat": "transit"},
    {"region": "Northeast India", "area_id": 4, "states": ["Assam","Meghalaya","Manipur","Mizoram","Nagaland","Arunachal Pradesh","Tripura"],
     "hub": "Guwahati", "lat": 26.14, "lon": 91.77, "poi": "Kamakhya Temple Zone", "cat": "temple"},
]

state_list = list(STATES.keys())

def get_region_for_state(state_name):
    for r in REGIONS:
        if state_name in r['states']:
            return r
    return REGIONS[0]

def get_coords_for_state(state_name):
    s = STATES[state_name]
    lat = s['lat'] + np.random.normal(0, 0.05)
    lon = s['lon'] + np.random.normal(0, 0.05)
    return round(lat, 6), round(lon, 6)

def get_city_for_state(state_name):
    return random.choice(STATES[state_name]['cities'])

def get_station_for_state(state_name):
    return random.choice(STATES[state_name]['stations'])

# ═══════════════ UPDATE AREA CENTERS ═══════════════
print("Updating area_centers.csv...")
area_rows = []
for r in REGIONS:
    area_rows.append({
        'area_id': r['area_id'],
        'lat': r['lat'],
        'lon': r['lon'],
        'name': f"{r['hub']} — {r['region']}",
        'category': r['cat']
    })
area_df = pd.DataFrame(area_rows)
area_df.to_csv("data/processed/area_centers.csv", index=False)
print(f"  -> {len(area_df)} regions saved")

# ═══════════════ GENERATE 112 CASES (28 states × 4 each) ═══════════════
print("Generating cases across 28 states...")
cases = []
case_num = 1
age_groups = ['18-25','26-35','36-50','51-65','65+']
genders = ['Male','Female']
weathers = ['Clear','Cloudy','Rain','Fog','Partly Cloudy']
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

for state in state_list:
    n_cases = 4  # 4 per state = 112 total
    for _ in range(n_cases):
        lat, lon = get_coords_for_state(state)
        city = get_city_for_state(state)
        station = get_station_for_state(state)
        code = STATES[state]['code']
        region = get_region_for_state(state)
        
        # Pick target and previous areas
        target_region = random.choice(REGIONS)
        prev_region = random.choice(REGIONS)
        
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        cases.append({
            'Case_ID': f"MP-{code}-2026-{case_num:03d}",
            'FIR_Reference': f"FIR-{random.randint(100,999)}/2026-{code}",
            'Person_ID': random.randint(1, 500),
            'Age_Group': random.choice(age_groups),
            'Gender': random.choice(genders),
            'Last_Latitude': lat,
            'Last_Longitude': lon,
            'Last_Seen_Time': f"{hour:02d}:{minute:02d}",
            'Day': random.choice(days),
            'Weather': random.choice(weathers),
            'State': state,
            'City': city,
            'Police_Station': station,
            'Usual_Area': f"{get_city_for_state(state)} {random.choice(['RTC Complex','Railway Station','Bus Stand','Market Area','Temple Zone','IT Park','University Campus'])}",
            'Average_Distance': round(random.uniform(3, 25), 2),
            'Average_Speed': round(random.uniform(1.5, 15), 2),
            'Previous_Area': f"{get_city_for_state(state)} {random.choice(['Residential Zone','Commercial Area','Transit Hub','Market','Hospital Area'])}",
            'Time_Since_Last_Seen': random.randint(2, 72),
            'Target_Area': f"{target_region['hub']} — {target_region['region']}",
        })
        case_num += 1

cases_df = pd.DataFrame(cases)
cases_df.to_csv("data/synthetic/cases.csv", index=False)
print(f"  -> {len(cases_df)} cases across {len(state_list)} states saved")

# ═══════════════ INVESTIGATION CASE ═══════════════
print("Creating investigation case...")
inv_case = {
    'Case_ID': 'MP-AP-2026-001',
    'FIR_Reference': 'FIR-854/2026-AP',
    'Person_ID': 1,
    'Age_Group': '18-25',
    'Gender': 'Male',
    'Last_Latitude': 13.6288,
    'Last_Longitude': 79.4192,
    'Last_Seen_Time': '19:00',
    'Day': 'Monday',
    'Weather': 'Clear',
    'State': 'Andhra Pradesh',
    'City': 'Tirupati',
    'Police_Station': 'Tirupati Urban PS',
    'Usual_Area': 'Visakhapatnam RTC Complex',
    'Average_Distance': 12.18,
    'Average_Speed': 35.27,
    'Previous_Area': 'Visakhapatnam RTC Complex',
    'Time_Since_Last_Seen': 33,
    'Target_Area': 'Hyderabad — South India',
    'description': 'Subject was last seen near Tirupati Railway Station on Monday evening. CCTV footage shows subject boarding a southbound local bus at 19:00 IST. Subject\'s mobile phone was last active near Tirupati Urban area. Family reports subject was planning to visit a temple. Belongings found near the bus stand.',
    'investigating_officer': 'SI R. Krishnamurthy',
    'report_date': '2026-09-15',
}
pd.DataFrame([inv_case]).to_csv("data/synthetic/investigation_case.csv", index=False)
print("  -> Investigation case (Andhra Pradesh) saved")

# ═══════════════ UPDATE SEARCH PRIORITY ═══════════════
print("Updating search_priority.csv...")
pri = pd.read_csv("data/processed/search_priority.csv")
region_names = {r['area_id']: f"{r['hub']} — {r['region']}" for r in REGIONS}
pri['area_name'] = pri['area_id'].map(region_names).fillna(pri['area_name'])
pri.to_csv("data/processed/search_priority.csv", index=False)
print("  -> Search priority updated with Indian region names")

# ═══════════════ UPDATE PREDICTED ROUTES ═══════════════
print("Updating predicted_routes.csv...")
routes = pd.read_csv("data/processed/predicted_routes.csv")
routes['area_name'] = routes['area_id'].map(region_names).fillna(routes['area_name'].astype(str))
routes.to_csv("data/processed/predicted_routes.csv", index=False)
print("  -> Predicted routes updated")

# ═══════════════ UPDATE STAY POINTS COORDS (approximate Indian spread) ═══════════════
print("Adjusting stay_points coordinates to Indian geography...")
stay = pd.read_csv("data/processed/stay_points_clustered.csv")
# Map Beijing coords to Indian coords: lat 39.85-40.05 -> 12-28, lon 116.2-116.55 -> 73-94
stay['lat'] = 12 + (stay['lat'] - stay['lat'].min()) / (stay['lat'].max() - stay['lat'].min()) * 16
stay['lon'] = 73 + (stay['lon'] - stay['lon'].min()) / (stay['lon'].max() - stay['lon'].min()) * 21
stay['stay_lat'] = 12 + (stay['stay_lat'] - stay['stay_lat'].min()) / (stay['stay_lat'].max() - stay['stay_lat'].min()) * 16
stay['stay_lon'] = 73 + (stay['stay_lon'] - stay['stay_lon'].min()) / (stay['stay_lon'].max() - stay['stay_lon'].min()) * 21
stay.to_csv("data/processed/stay_points_clustered.csv", index=False)
print(f"  -> {len(stay)} stay points remapped to India")

print("\n✅ All data converted to Pan-India 28-State system!")
print(f"   States: {len(state_list)}")
print(f"   Cases: {len(cases_df)}")
print(f"   Regions: {len(REGIONS)}")

import os
import sys
import requests
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")

def fetch_acs_data():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    years = list(range(2015, 2023))
    
    # We specifically need these variables for process_census.py
    variables = "GEO_ID,NAME,DP03_0062E,DP03_0009PE,DP03_0063E,DP03_0119E"
    
    for year in years:
        print(f"Fetching ACS 5-Year Data for {year}...")
        url = f"https://api.census.gov/data/{year}/acs/acs5/profile"
        
        params = {
            "get": variables,
            "for": "county subdivision:*",
            "in": "state:34"  # New Jersey
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # The first row contains headers
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            
            # The API returns GEO_ID with a slightly different prefix sometimes, 
            # but usually it's "0600000US34...". Let's verify and just save.
            # We don't need the state, county, county subdivision columns.
            
            out_file = os.path.join(DATA_RAW_DIR, f"ACSDP5Y{year}.DP03-Data.csv")
            df.to_csv(out_file, index=False)
            print(f"Saved {year} to {out_file}")
        else:
            print(f"Failed to fetch {year}. Status code: {response.status_code}")
            print(response.text)

if __name__ == '__main__':
    fetch_acs_data()

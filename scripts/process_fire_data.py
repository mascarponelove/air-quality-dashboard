#!/usr/bin/env python3
"""
Fire Detection Data Processor for Punjab & Haryana
Processes NASA FIRMS VIIRS data and generates district-wise fire counts
"""

import requests
import zipfile
import io
import geopandas as gpd
import pandas as pd
import json
from datetime import datetime
import os
import sys

# =============================================================================
# CONFIGURATION - EDIT THESE IF NEEDED
# =============================================================================

FIRMS_URL = 'https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/shapes/zips/J1_VIIRS_C2_South_Asia_7d.zip'

# Shapefile paths (relative to repository root)
PUNJAB_SHAPEFILE = 'shapefiles/Punjab_District.shp'
HARYANA_SHAPEFILE = 'shapefiles/Haryana_District.shp'

# Output paths
OUTPUT_JSON = 'data/fire_counts.json'
SUMMARY_JSON = 'data/fire_counts_summary.json'

# District name column - CHANGE THIS if your shapefile uses different column name
DISTRICT_COLUMN = 'dtname'  # Common alternatives: 'DISTRICT', 'NAME', 'dist_name'

# =============================================================================
# FUNCTIONS
# =============================================================================

def print_step(message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}")

def download_and_extract_fire_data():
    """Download VIIRS fire data from FIRMS and extract shapefile"""
    print_step("STEP 1: Downloading VIIRS Fire Data from NASA FIRMS")
    
    try:
        print(f"Downloading from: {FIRMS_URL}")
        response = requests.get(FIRMS_URL, timeout=120)
        response.raise_for_status()
        
        print(f"Download complete. Size: {len(response.content) / 1024 / 1024:.2f} MB")
        
        # Extract ZIP file
        print("Extracting shapefile...")
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall('temp_fire_data')
        
        shapefile_path = 'temp_fire_data/J1_VIIRS_C2_South_Asia_7d.shp'
        
        if not os.path.exists(shapefile_path):
            raise FileNotFoundError(f"Expected shapefile not found: {shapefile_path}")
        
        print(f"✓ Fire data extracted successfully")
        return shapefile_path
        
    except Exception as e:
        print(f"✗ ERROR downloading fire data: {e}")
        sys.exit(1)

def load_district_shapefiles():
    """Load and combine Punjab and Haryana district shapefiles"""
    print_step("STEP 2: Loading District Boundary Shapefiles")
    
    try:
        # Check if files exist
        if not os.path.exists(PUNJAB_SHAPEFILE):
            raise FileNotFoundError(f"Punjab shapefile not found: {PUNJAB_SHAPEFILE}")
        if not os.path.exists(HARYANA_SHAPEFILE):
            raise FileNotFoundError(f"Haryana shapefile not found: {HARYANA_SHAPEFILE}")
        
        # Load shapefiles
        print(f"Loading Punjab districts from: {PUNJAB_SHAPEFILE}")
        punjab_districts = gpd.read_file(PUNJAB_SHAPEFILE)
        print(f"  Found {len(punjab_districts)} Punjab districts")
        
        print(f"Loading Haryana districts from: {HARYANA_SHAPEFILE}")
        haryana_districts = gpd.read_file(HARYANA_SHAPEFILE)
        print(f"  Found {len(haryana_districts)} Haryana districts")
        
        # Check if district column exists
        if DISTRICT_COLUMN not in punjab_districts.columns:
            print(f"\n✗ ERROR: Column '{DISTRICT_COLUMN}' not found in Punjab shapefile")
            print(f"Available columns: {list(punjab_districts.columns)}")
            sys.exit(1)
        
        if DISTRICT_COLUMN not in haryana_districts.columns:
            print(f"\n✗ ERROR: Column '{DISTRICT_COLUMN}' not found in Haryana shapefile")
            print(f"Available columns: {list(haryana_districts.columns)}")
            sys.exit(1)
        
        # Add state identifiers
        punjab_districts['state'] = 'Punjab'
        haryana_districts['state'] = 'Haryana'
        
        # Combine districts
        all_districts = pd.concat([punjab_districts, haryana_districts], ignore_index=True)
        
        print(f"\n✓ Total districts loaded: {len(all_districts)}")
        print(f"  CRS: {all_districts.crs}")
        
        return all_districts
        
    except Exception as e:
        print(f"✗ ERROR loading district shapefiles: {e}")
        sys.exit(1)

def process_fire_counts(fire_shapefile, districts):
    """Process fire points and count by district"""
    print_step("STEP 3: Processing Fire Points - Spatial Analysis")
    
    try:
        # Read fire points
        print(f"Reading fire detection points from: {fire_shapefile}")
        points = gpd.read_file(fire_shapefile)
        print(f"  Total fire points detected: {len(points)}")
        
        if len(points) == 0:
            print("⚠ WARNING: No fire points found in dataset")
            return pd.DataFrame(columns=['ACQ_DATE', 'state', DISTRICT_COLUMN, 'fire_count'])
        
        # Sort by acquisition date
        points = points.sort_values(by='ACQ_DATE')
        print(f"  Date range: {points['ACQ_DATE'].min()} to {points['ACQ_DATE'].max()}")
        
        # Ensure CRS match for spatial join
        print(f"  Fire points CRS: {points.crs}")
        print(f"  Districts CRS: {districts.crs}")
        
        if points.crs != districts.crs:
            print("  ⚠ CRS mismatch detected - reprojecting fire points...")
            points = points.to_crs(districts.crs)
            print("  ✓ Reprojection complete")
        
        # Perform spatial join
        print("\n  Performing spatial join (this may take 30-60 seconds)...")
        points_districts = gpd.sjoin(points, districts, how='inner', predicate='within')
        print(f"  ✓ Spatial join complete: {len(points_districts)} fires matched to districts")
        
        if len(points_districts) == 0:
            print("  ⚠ WARNING: No fires matched any districts!")
            print("  This might indicate a CRS or geometry issue.")
        
        # Group by date, state, and district
        print("\n  Aggregating fire counts by district and date...")
        district_counts = points_districts.groupby(['ACQ_DATE', 'state', DISTRICT_COLUMN]).size().reset_index(name='fire_count')
        district_counts = district_counts.sort_values(['ACQ_DATE', 'state', 'fire_count'], ascending=[True, True, False])
        
        print(f"  ✓ Generated {len(district_counts)} district-date records")
        
        # Summary statistics
        total_fires = district_counts['fire_count'].sum()
        affected_districts = district_counts[DISTRICT_COLUMN].nunique()
        print(f"\n  SUMMARY:")
        print(f"    Total fires: {total_fires}")
        print(f"    Affected districts: {affected_districts}")
        print(f"    Punjab fires: {district_counts[district_counts['state']=='Punjab']['fire_count'].sum()}")
        print(f"    Haryana fires: {district_counts[district_counts['state']=='Haryana']['fire_count'].sum()}")
        
        return district_counts
        
    except Exception as e:
        print(f"✗ ERROR during fire processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_json_output(district_counts):
    """Create structured JSON output for dashboard"""
    print_step("STEP 4: Generating JSON Output Files")
    
    if len(district_counts) == 0:
        print("⚠ No fire data to process - creating empty dataset")
        return {
            'last_updated': datetime.utcnow().isoformat() + 'Z',
            'data_source': 'NASA FIRMS VIIRS (NOAA-20)',
            'time_period': '7 days',
            'daily_data': []
        }
    
    output_data = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'data_source': 'NASA FIRMS VIIRS (NOAA-20)',
        'time_period': '7 days',
        'daily_data': []
    }
    
    # Process each date
    unique_dates = sorted(district_counts['ACQ_DATE'].unique())
    print(f"Processing {len(unique_dates)} dates...")
    
    for date in unique_dates:
        # FIXED: Handle both string and datetime objects
        if isinstance(date, str):
            date_str = date  # Already a string
        else:
            date_str = date.strftime('%Y-%m-%d')  # Convert datetime to string
        
        date_data = district_counts[district_counts['ACQ_DATE'] == date]
        
        punjab_data = date_data[date_data['state'] == 'Punjab']
        haryana_data = date_data[date_data['state'] == 'Haryana']
        
        punjab_districts = [
            {
                'district': row[DISTRICT_COLUMN],
                'fire_count': int(row['fire_count'])
            }
            for _, row in punjab_data.iterrows()
        ]
        
        haryana_districts = [
            {
                'district': row[DISTRICT_COLUMN],
                'fire_count': int(row['fire_count'])
            }
            for _, row in haryana_data.iterrows()
        ]
        
        daily_entry = {
            'date': date_str,
            'punjab': {
                'districts': punjab_districts,
                'total': int(punjab_data['fire_count'].sum())
            },
            'haryana': {
                'districts': haryana_districts,
                'total': int(haryana_data['fire_count'].sum())
            },
            'combined_total': int(punjab_data['fire_count'].sum() + haryana_data['fire_count'].sum())
        }
        
        output_data['daily_data'].append(daily_entry)
        print(f"  {date_str}: {daily_entry['combined_total']} fires")
    
    print("✓ Daily data JSON created")
    return output_data

def create_summary_output(district_counts):
    """Create summary statistics for the entire period"""
    
    if len(district_counts) == 0:
        return {
            'last_updated': datetime.utcnow().isoformat() + 'Z',
            'total_fire_count': 0,
            'date_range': {'start': None, 'end': None},
            'top_districts': [],
            'state_totals': {'punjab': 0, 'haryana': 0}
        }
    
    # FIXED: Handle string dates properly
    unique_dates = sorted(district_counts['ACQ_DATE'].unique())
    
    # Convert dates to strings if they aren't already
    if len(unique_dates) > 0:
        if isinstance(unique_dates[0], str):
            start_date = unique_dates[0]
            end_date = unique_dates[-1]
        else:
            start_date = unique_dates[0].strftime('%Y-%m-%d')
            end_date = unique_dates[-1].strftime('%Y-%m-%d')
    else:
        start_date = None
        end_date = None
    
    summary = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'total_fire_count': int(district_counts['fire_count'].sum()),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'top_districts': []
    }
    
    # Calculate top 20 districts by total fire count
    top_districts = district_counts.groupby(['state', DISTRICT_COLUMN])['fire_count'].sum().reset_index()
    top_districts = top_districts.sort_values('fire_count', ascending=False).head(20)
    
    summary['top_districts'] = [
        {
            'state': row['state'],
            'district': row[DISTRICT_COLUMN],
            'total_fires': int(row['fire_count'])
        }
        for _, row in top_districts.iterrows()
    ]
    
    # State-wise totals
    state_totals = district_counts.groupby('state')['fire_count'].sum()
    summary['state_totals'] = {
        'punjab': int(state_totals.get('Punjab', 0)),
        'haryana': int(state_totals.get('Haryana', 0))
    }
    
    print("✓ Summary data JSON created")
    return summary

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("  FIRE DETECTION DATA PROCESSOR")
    print("  Punjab & Haryana Agricultural Fires")
    print("  Data Source: NASA FIRMS VIIRS (NOAA-20)")
    print("="*60)
    
    try:
        # Create output directory
        os.makedirs('data', exist_ok=True)
        
        # Step 1: Download fire data
        fire_shapefile = download_and_extract_fire_data()
        
        # Step 2: Load district boundaries
        districts = load_district_shapefiles()
        
        # Step 3: Process fire counts
        district_counts = process_fire_counts(fire_shapefile, districts)
        
        # Step 4: Generate JSON outputs
        daily_output = create_json_output(district_counts)
        summary_output = create_summary_output(district_counts)
        
        # Save JSON files
        print_step("STEP 5: Saving Output Files")
        
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(daily_output, f, indent=2)
        print(f"✓ Saved: {OUTPUT_JSON}")
        
        with open(SUMMARY_JSON, 'w') as f:
            json.dump(summary_output, f, indent=2)
        print(f"✓ Saved: {SUMMARY_JSON}")
        
        # Final summary
        print_step("PROCESSING COMPLETE ✓")
        print(f"Total fires detected: {summary_output['total_fire_count']}")
        print(f"Date range: {summary_output['date_range']['start']} to {summary_output['date_range']['end']}")
        print(f"Punjab: {summary_output['state_totals']['punjab']} fires")
        print(f"Haryana: {summary_output['state_totals']['haryana']} fires")
        print("\nOutput files ready for dashboard!")
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

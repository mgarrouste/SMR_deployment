import pandas as pd
import numpy as np

#EPA data
epa_df = pd.read_excel('./epa_flight_iron_and_steel_2022_emissions.xls', skiprows=5)
epa_df['FACILITY NAME'] = epa_df['FACILITY NAME'].apply(lambda x: x.upper().replace('U.S.','US'))
epa_df['Latitude rounded'] = epa_df['LATITUDE'].apply(lambda x: np.round(x,1))
epa_df['Longitude rounded'] = epa_df['LONGITUDE'].apply(lambda x: np.round(x,1))
epa_df.rename(columns={'LATITUDE':'latitude', 'LONGITUDE':'longitude'}, inplace=True)


# global tracker data: main production process mapping, select bf/bof
global_tracker_df = pd.read_excel('./Global-Steel-Plant-Tracker-2023-03-2.xlsx', sheet_name='Steel Plants')
# Select only U.S. plants and BF/BOF ones in global tracker data
global_tracker_df = global_tracker_df[global_tracker_df['Country'] == 'United States']
global_tracker_df['FACILITY NAME'] = global_tracker_df['Plant name (English)'].apply(lambda x: x.upper().replace('U.S.','US'))
global_tracker_df = global_tracker_df[(global_tracker_df['Main production process'] == 'integrated (BF)') | (global_tracker_df['Main production process'] == 'oxygen')]
# US Steel Great Lakes closed down in 2020 


#Merge EPA and Global tracker data on latitude, longitude and plant name
global_tracker_df['Latitude rounded'] = global_tracker_df['Coordinates'].apply(lambda x:np.round(float(str(x).split(',')[0]),1))
global_tracker_df['Longitude rounded'] = global_tracker_df['Coordinates'].apply(lambda x:np.round(float(str(x).split(',')[1]),1))
global_tracker_df['Name 2 letters'] = global_tracker_df['FACILITY NAME'].apply(lambda x:x[:2])
epa_df['Name 2 letters'] = epa_df['FACILITY NAME'].apply(lambda x:x[:2])

merged  = pd.merge(global_tracker_df, epa_df, on=['Latitude rounded', 'Longitude rounded','Name 2 letters'])
# Get the production process capacity
merged['Steel production capacity (ttpa)'] = (
  np.where(
    merged['Main production process'] == 'oxygen',
    merged['Nominal BOF steel capacity (ttpa)'],
    np.where(
      merged['Main production process'] =='integrated (BF)',
      merged['Nominal BF capacity (ttpa)'], None
    )
))

# Check matching between the two dataset
merged = merged[['Plant name (English)', 'FACILITY NAME_x', 'FACILITY NAME_y', 'Coordinates', 'latitude', 'longitude', 
                 'Location address', 'REPORTED ADDRESS', 'CITY NAME', 'COUNTY NAME', 'STATE', 'Subnational unit (province/state)','ZIP CODE',
                 'Status','Main production process', 'Main production equipment','GHG QUANTITY (METRIC TONS CO2e)','Steel production capacity (ttpa)']]
# Select relevant columns for final dataset
clean_df = merged[['Plant name (English)', 'latitude', 'longitude', 'REPORTED ADDRESS', 'CITY NAME', 'COUNTY NAME', 'ZIP CODE', 'STATE',
                    'Status', 'Main production process', 'GHG QUANTITY (METRIC TONS CO2e)', 
                   'Steel production capacity (ttpa)']]



# calculation of equivalent hydrogen demadn
hydrogen_rate = 67.095 #kg-H2/ton_DRI cf research notes
ratio_steel_dri = 0.9311 #t_steel/t_dri
clean_df['Hydrogen demand (kg/day)'] = clean_df['Steel production capacity (ttpa)']*1e3*hydrogen_rate/(365*ratio_steel_dri)

# Calculation of electricity demand from auxiliary components
aux_elec_demand_rate = 0.1025 #kWh/ton-DRI cf research notes
eaf_elec_demand_rate = 0.461 #kWh/tsteel cf research note
clean_df['Electricity demand (MWe)'] = clean_df['Steel production capacity (ttpa)']*1e3*(eaf_elec_demand_rate + aux_elec_demand_rate/ratio_steel_dri)/(365*24)

clean_df.rename(columns={'Plant name (English)':'Plant'}, inplace=True)
clean_df.to_excel('../h2_demand_bfbof_steel_us_2022.xlsx', sheet_name='processed', index=False)

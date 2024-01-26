import pandas as pd 
import numpy as np 

heat_df = pd.read_csv('./industry_heat_demand_characterization_nrel.csv', index_col=0, encoding='windows-1252',header=0)

# SElect high temperature 
# Drop null demand and convert total from TJ to MJ 
heat_df.drop(heat_df.index[(heat_df["Total"] ==0)],axis=0,inplace=True)
heat_df['Heat demand (MJ/year)'] = heat_df.apply(lambda row:row['Total']*1e6, axis=1)
heat_df.drop(columns=['Total'], inplace=True)
# We are interested in displacing natural gas for temperatures higher than outlet temperatures of ANRs
# Max delivered temperature is : thermal transfer efficiency x max outlet temperature = TTE(HTGR) x Outlet temp (HTGR) = 
# 90% x 950 deg C = 855 deg C
CUTOFF = 855
print('All entries {}'.format(len(heat_df)))
hot_df = heat_df[heat_df['Temp_degC']>=CUTOFF]
print('Only above cutoff ({} deg C): {}, {} % of the data'.format(CUTOFF, len(hot_df), np.round(100*len(hot_df)/len(heat_df),2) ))
NON_AUTHORIZED_FUEL_TYPE = ['Blast Furnace Gas', 'Coke Oven Gas', 'Fuel Gas', '0', 'Used Oil']
NON_AUTHORIZED_FUEL_TYPE_BLEND = ['Process Off-gas', 'Scrap Fat (from W-4 Tank)', 'Nitrile Pitch (from W-3 Tank)','Biogenic Process Derived Fuel (Glidfuel)',
                            'Biogenic Process Derived Fuel', 'Sweet Gas  Return', 'Biogeninc Process Derived Fuel (PDF)',
                             'Biogenic Process Derived Fuel (PDF)','FurnGas', 'Fuel Gas']
ng_df = hot_df[hot_df['FUEL_TYPE'].isin(['Natural Gas (Weighted U.S. Average)', 'Mixed (Industrial sector)'])]
ng_df = ng_df[~(ng_df['FUEL_TYPE_BLEND'].isin(NON_AUTHORIZED_FUEL_TYPE_BLEND))]
print(len(ng_df))
print(np.round(100*len(ng_df)/len(heat_df),2), '% of total entries')
print(np.round(100*len(ng_df)/len(hot_df),2), '% of hot entries')


# Convert remaining data in equivalent hydrogen demand
ng_df = ng_df[['CITY', 'COUNTY', 'FACILITY_ID', 'FUEL_TYPE', 'REPORTING_YEAR', 'STATE', 'Temp_degC',  'Natural_gas', 'Other',
               'UNIT_NAME', 'UNIT_TYPE', 'MMTCO2E', 'Heat demand (MJ/year)']]
# Load the conversion spreadsheet for hhv of hydrogen
hhv_map = pd.read_excel('fuels_HHV_industry_heat.xlsx', sheet_name='hhv', skiprows=1, index_col='fuel')
hhv_hydrogen = hhv_map.at['Hydrogen','HHV (MJ/kg)']
# Convert heat demand to mass of hydrogen demand
ng_df['H2 demand (kg/year)'] = ng_df['Heat demand (MJ/year)']/hhv_hydrogen


# The data includes 6 reporting years: 2010 to 2015. We include the min, avg and max of heat demand in the pre-processed demand spreadsheet
stats_df = ng_df[['FACILITY_ID', 'Natural_gas', 'Other', 'MMTCO2E', 'Heat demand (MJ/year)', 'H2 demand (kg/year)']]
max_df = stats_df.groupby('FACILITY_ID').max()
avg_df = stats_df.groupby('FACILITY_ID').mean()
min_df = stats_df.groupby('FACILITY_ID').min()

# Save everything
with pd.ExcelWriter('../h2_demand_industry_heat.xlsx') as writer: 
  ng_df.to_excel(writer, index=False, sheet_name='all_years')
  max_df.to_excel(writer, sheet_name='max')
  avg_df.to_excel(writer, sheet_name='mean')
  min_df.to_excel(writer, sheet_name='min')
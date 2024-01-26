import pandas as pd
import numpy as np

ammonia_df = pd.read_excel('./statistic_id1266392_ammonia-plant-production-capacity-in-the-us-2022.xlsx', sheet_name='Data', header=None, skiprows=5, usecols=[1,2])
ammonia_df.rename(columns={1:'Plant',2:'Capacity'}, inplace=True)
ammonia_df['City'] = ammonia_df.apply(lambda x:x['Plant'].split('(')[1].split(',')[0], axis=1)
ammonia_df['State'] = ammonia_df.apply(lambda x:x['Plant'].split('(')[1][:-1].split(',')[1], axis=1)
ammonia_df['Plant'] = ammonia_df.apply(lambda x:x['Plant'].split('(')[0][:-1], axis=1)
ammonia_df['Capacity (tNH3/year)'] = ammonia_df['Capacity']*1e3 # Original data in thousands MT
ammonia_df.drop(columns=['Capacity'], inplace=True)
# Create id with plant name and city
ammonia_df['plant_id'] = ammonia_df.apply(lambda x: x['Plant'][:2]+'_'+x['City'][:2], axis=1)



# Hydrogen equivalent demand
# Values in modeling paper based on LHV of materials
h2_lhv = pd.read_excel('./fuels_HHV_industry_heat.xlsx', sheet_name='lhv', skiprows= 1, index_col=0).loc['Hydrogen', 'HHV (MJ/kg)']
h2_to_nh3 = 23.67*1e3 #MJ/tNH3
h2_to_nh3_ratio = h2_to_nh3/h2_lhv #kgH2/tNH3

ammonia_df['H2 Dem. (kg/year)'] = ammonia_df['Capacity (tNH3/year)']*h2_to_nh3_ratio
elec_to_nh3 = 0.119 # MWh/tNH3
ammonia_df['Electricity demand (MWe)'] = elec_to_nh3*ammonia_df['Capacity (tNH3/year)']/(365*24)


# save
ammonia_df.to_excel('../h2_demand_ammonia_us_2022.xlsx', sheet_name='processed', index=False)
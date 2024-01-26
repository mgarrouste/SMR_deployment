import pandas as pd
import numpy as np 

# Data from the EIA: Captive h2 production at refineries
eia_df = pd.read_excel('./hydrogen_production_capacities_at_US_refineries_EIA_2022.xlsx', skiprows=3,header=0)
eia_df = eia_df.iloc[:-2,:] # Remove totals
eia_df = eia_df[['State', 'Company', 'City', 2022]] # Only keep data for year 2022
eia_df.dropna(axis=0, ignore_index=True, inplace=True) # Drop rows with null production 


#convert hydrogen demand in kg/day
hydrogen_density = 0.002408 # kg/ standard cubic feet
clean_df = eia_df[['State', 'Company','City']]
clean_df['2022'] = eia_df[2022]*1e6*hydrogen_density

# check NG consumptions and equivalent NG demand

# Map each refinery to corresponding padd
map_padd = pd.read_excel('./map_state_padd.xlsx', skiprows=1)
refineries_mapped = pd.merge(clean_df, map_padd, on='State')
# Sum of hydrogen demand per padd
sum_ref_padd = refineries_mapped.groupby('PADD').sum('2022')
# and convert to equivalent natural gas demand in MMCf
# from H2A tool 0.1578 MMBtu/kgH2
h2_to_ng_smr = 0.1578 #MMBtu NG/kgH2
# NG: 1 Mcf = 1.038 MMBtu
vol_to_nrj_ng = 1038 # MMBtu/MMcf
sum_ref_padd['NG eq demand 2022(MMcf/year)'] = 365*sum_ref_padd['2022']*h2_to_ng_smr/vol_to_nrj_ng
# Load data natural gas feedstock for h2 production at refineries
published_ng_feedstock_padd = pd.read_excel('./nat_gas_feedstock_h2_prod_refineries_padd_level_eia.xls', sheet_name='Data 1', skiprows=2)
published_ng_feedstock_padd = published_ng_feedstock_padd.iloc[-1, 2:]
published_ng_feedstock_padd = published_ng_feedstock_padd.to_frame()
published_ng_feedstock_padd.set_index(pd.Series([1,2,3,4,5]),inplace=True)
published_ng_feedstock_padd.rename(columns={14:'Published NG Feedstock (MMcf/year)'}, inplace=True)
published_ng_feedstock_padd.reset_index(names=['PADD'], inplace=True)
merged_ng_demands = pd.merge(published_ng_feedstock_padd, sum_ref_padd, on='PADD')
merged_ng_demands.drop(columns=['2022'], inplace=True)

#compute ratio of imported NG for h2 production at padd level
merged_ng_demands['Ratio imported NG 2022'] = merged_ng_demands['Published NG Feedstock (MMcf/year)']/merged_ng_demands['NG eq demand 2022(MMcf/year)']
# For each refinery, depending on their PADD, we assume the average ratio of imported NG calculated at the PADD level can be applied
ratio_mapped = merged_ng_demands[['PADD', 'Ratio imported NG 2022']]
ref_ratio = pd.merge(refineries_mapped, ratio_mapped, on='PADD')
ref_ratio['Corrected 2022 demand (kg/day)'] = ref_ratio['2022']*ref_ratio['Ratio imported NG 2022']
def create_id(company, city): 
  return company[:2]+'_'+city[:3]
ref_ratio['refinery_id'] = ref_ratio.apply(lambda x: create_id(x['Company'], x['City']), axis=1)

# SAve results
ref_ratio.to_excel('../h2_demand_refineries.xlsx', sheet_name='processed', index=False)
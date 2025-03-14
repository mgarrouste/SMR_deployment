import pandas as pd
import numpy as np
import math
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def import_smr_h2_data(OAK):
    h2_techs = pd.read_excel('./h2_tech.xlsx', sheet_name = 'Summary')
    h2_techs.rename(columns={'CAPEX ($/MWe)':'H2 CAPEX ($/MWe)',
                    'FOM ($/MWe-year)':'H2 FOM ($/MWe-year)',
                    'VOM ($/MWhe)':'H2 VOM ($/MWhe)'}, inplace=True)
    SMRs = pd.read_excel('./SMR_inputs.xlsx', sheet_name=OAK)
    SMRs.rename(columns={'CAPEX $/MWe':'SMR CAPEX ($/MWe)',
                    'FOPEX $/MWe-y':'SMR FOM ($/MWe-year)',
                    'VOM in $/MWh-e':'SMR VOM ($/MWhe)', 
                    'Startupfixedcost in $':'Start Cost ($)'}, inplace=True)
    techs = pd.merge(h2_techs, SMRs, left_on='SMR', right_on='Type')
    return techs

def get_ng_prices(year):
    ng_prices = pd.read_excel('./input_data/eia_aeo_industrial_sector_ng_prices.xlsx', sheet_name='state_prices')
    # 2024 prices
    ng_prices = ng_prices[ng_prices.year == year]
    ng_prices.rename(columns={'price 2020USD/MMBtu':'NG price ($/MMBtu)'}, inplace=True)
    return ng_prices

def compute_ng_multiplier(temp, AHF_coeffs=[0, -.00038, 0.90556]):
  # Available Heat Fraction
  AHF = AHF_coeffs[0]*(int(temp)**2)+ AHF_coeffs[1]*int(temp) + AHF_coeffs[2]
  multiplier = AHF/utils.mmbtu_to_mj # Unit: MmBtu/MJ
  return multiplier

def get_coordinates_facilities():
    locations_file = './results/process_heat/heat_facilities_locations.csv'
    if not os.path.isfile(locations_file):
        loc_data = pd.read_excel('./input_data/direct_heat_maxv/facs_batched.xlsx')[['CITY', 'STATE']].drop_duplicates(ignore_index=True)
        geolocator = Nominatim(user_agent="your_app_name")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        # Function to apply geocoding
        def geocode_location(row):
            # Try to geocode using city and state, else return NaN
            try:
                location = geocode(f"{row['CITY']}, {row['STATE']}, USA")
                return location.latitude, location.longitude
            except:
                return pd.NA, pd.NA

        # Apply the geocoding function to each row
        loc_data['latitude'], loc_data['longitude'] = zip(*loc_data.apply(geocode_location, axis=1))
        loc_data.to_csv(locations_file, index=False)
    else:
        loc_data = pd.read_csv(locations_file)
    return loc_data


def get_nrel_data():
    nrel_data = pd.read_excel('./input_data/direct_heat_maxv/facs_batched.xlsx')
    nrel_data.drop(columns=['Unnamed: 0', 'Total'], inplace=True)
    return nrel_data

def get_direct_heat_results():
    max_results = pd.read_csv('./input_data/direct_heat_maxv/Full_Spread_SMRs.csv')
    max_results.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
    return max_results


def compute_h2_demand(heat, temp, AHF_coeffs = [0, -0.00038, .90556]):
  """Computes the equivalent hydrogen demand to produce heat 
  Args:
    - heat (float): heat demand in MW
    - temp (float): temperature in degree Celsius
    - AHF_coeffs (list(float)): coefficients for Available Heat Fraction calculation
  Returns:
    - h2_demand (float): hydrogen demand in kg/h
  """
  AHF = AHF_coeffs[0]*(int(temp)^2) + AHF_coeffs[1]*int(temp) + AHF_coeffs[2]
  h2_demand = heat*utils.mwh_to_mj/(utils.h2_hhv*AHF)
  return h2_demand


def compute_smr_depl(direct_heat, techs):
    techs_to_merge = techs[['SMR', 'H2Cap (kgh2/h)', 'H2Cap (MWe)', 'Power in MWe', 
                            'SMR CAPEX ($/MWe)', 'H2 CAPEX ($/MWe)', 'SMR FOM ($/MWe-year)',
                            'H2 FOM ($/MWe-year)', 'Eq tot H2ElecCons (MWhe/kgh2)', 'SMR VOM ($/MWhe)',
                            'H2 VOM ($/MWhe)']]
    smr_depl = direct_heat.merge(techs_to_merge, left_on='Type', right_on='SMR')
    smr_depl['H2 Modules'] = smr_depl.apply(lambda x: math.ceil(x['Remaining H2 Dem. (kg/h)']/x['H2Cap (kgh2/h)']), axis=1)
    smr_depl['Depl. H2 Cap. (kgh2/h)'] = smr_depl['H2 Modules']*smr_depl['H2Cap (kgh2/h)']
    smr_depl['Depl. H2 Cap. (MWe)'] = smr_depl['H2 Modules']*smr_depl['H2Cap (MWe)']
    smr_depl['SMR Modules'] = smr_depl.apply(lambda x: math.ceil(x['Depl. H2 Cap. (MWe)']/x['Power in MWe']), axis=1)
    smr_depl['Depl. SMR Cap. (MWe)'] = smr_depl['SMR Modules']*smr_depl['Power in MWe']
    smr_depl['Depl. SMR Cap. (MWt)'] = smr_depl['SMR Modules']*smr_depl['Power in MWt']
    smr_depl['Surplus SMR Cap. (MWe)'] = smr_depl['Depl. SMR Cap. (MWe)']-smr_depl['Depl. H2 Cap. (MWe)']
    smr_depl['Surplus SMR Cap. (MWt)'] = smr_depl['Surplus SMR Cap. (MWe)']/smr_depl['Thermal Efficiency']
    return smr_depl


def compute_average_electricity_prices(cambium_scenario, year):
  folder = f'./input_data/cambium_{cambium_scenario.lower()}_state_hourly_electricity_prices'
  list_csv_files = glob.glob(folder+'/Cambium*.csv')
  state_prices = pd.DataFrame(columns=['average price ($/MWhe)', 'state'])
  state_prices.set_index('state', inplace=True)
  for file in list_csv_files:
    if str(year) in file:
      state = file.split('_')[-2]
      avg_price = pd.read_csv(file, skiprows=5)['energy_cost_enduse'].mean()
      state_prices.loc[state, 'average price ($/MWhe)'] = avg_price
  state_prices.to_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx')


def compute_cogen_revenues(df, surplus_cap_col_name, state_col_name, cambium_scenario, year):
  try:
    elec_prices_df = pd.read_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx', index_col=0)
  except FileNotFoundError:
    compute_average_electricity_prices(cambium_scenario, year)
    elec_prices_df = pd.read_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx', index_col=0)
  df['Electricity revenues ($/y)'] = df.apply(lambda x: x[surplus_cap_col_name]*elec_prices_df.loc[x[state_col_name]]*8760, axis=1)
  return df

def compute_cashflows(smr_depl,with_PTC,cogen,cambium_scenario,year):
    # Cashflows for SMR for direct heat: Annual_CAPEX, FOPEX, VOPEX, and Revenues (from avoided NG costs)
    # Compute costs in $/year
    # Capital recovery factor
    IR = utils.WACC
    smr_depl['SMR CRF'] = (IR/(1-((1+IR)**(-1*(smr_depl['Life (y)'])))))
    smr_depl['H2 CRF'] = (IR/(1-((1+IR)**(-1*(smr_depl['Life (y)'])))))
    itc_SMR = utils.ITC_SMR
    itc_h2 = utils.ITC_H2

    ## CAPEX
    smr_depl['Tot SMR CAPEX'] = smr_depl['Depl. SMR Cap. (MWe)']*smr_depl['SMR CAPEX ($/MWe)']*(1-itc_SMR)
    smr_depl['Annual SMR CAPEX'] = smr_depl['Tot SMR CAPEX']*smr_depl['SMR CRF']
    smr_depl['Tot H2 CAPEX'] = smr_depl['Depl. H2 Cap. (MWe)']*smr_depl['H2 CAPEX ($/MWe)']*(1-itc_h2)
    smr_depl['Annual H2 CAPEX'] = smr_depl['Tot H2 CAPEX']*smr_depl['H2 CRF']
    smr_depl['Annual SMR-H2 CAPEX'] = smr_depl['Annual SMR CAPEX']+smr_depl['Annual H2 CAPEX']

    ## FOM
    smr_depl['SMR FOM'] = smr_depl['Depl. SMR Cap. (MWe)']*smr_depl['SMR FOM ($/MWe-year)']
    smr_depl['H2 FOM'] = smr_depl['Depl. H2 Cap. (MWe)']*smr_depl['H2 FOM ($/MWe-year)']
    smr_depl['SMR-H2 FOM'] = smr_depl['SMR FOM']+smr_depl['H2 FOM']

    ## VOM
    smr_depl['SMR VOM'] =smr_depl['Remaining H2 Dem. (kg/h)']*8760*smr_depl['Eq tot H2ElecCons (MWhe/kgh2)']*smr_depl['SMR VOM ($/MWhe)']
    smr_depl['H2 VOM'] =smr_depl['Remaining H2 Dem. (kg/h)']*8760*smr_depl['H2 VOM ($/MWhe)']
    smr_depl['SMR-H2 VOM'] = smr_depl['SMR VOM']+smr_depl['H2 VOM']

    ## Conversion costs
    gf_crf = IR/ (1 - (1/(1+IR)**utils.GFLT) ) 
    smr_depl['Conversion'] = utils.GFCAPEX*gf_crf*(1-itc_h2)*smr_depl['Remaining_Heat_MW']
    smr_depl['Conversion total'] = utils.GFCAPEX*(1-itc_h2)*smr_depl['Remaining_Heat_MW']

    # Initial investment 
    smr_depl['Initial investment ($)'] = smr_depl['Conversion total']+smr_depl['Tot SMR CAPEX']+smr_depl['Tot H2 CAPEX']+smr_depl[f'Total_CAPEX']

    # Total cost
    smr_depl['SMR-H2 Total Cost ($/year)'] = smr_depl['Annual SMR-H2 CAPEX'] + smr_depl['SMR-H2 FOM'] + smr_depl['SMR-H2 VOM'] + smr_depl['Conversion']
    # Compute revenues in $/year
    # Avoided NG costs
    smr_depl['NG Mult (MMBtu/MJ)'] = smr_depl.apply(lambda x: compute_ng_multiplier(x['Remaining_temp_degC']), axis=1) 

    smr_depl['Avoided NG Cost'] = smr_depl['NG price ($/MMBtu)']*smr_depl['NG Mult (MMBtu/MJ)']*smr_depl['Remaining_Heat_MW']*utils.mwh_to_mj*8760
    # Total avoided NG cost
    smr_depl['Avoided NG Cost ($/y)'] = smr_depl['Avoided NG Cost']+smr_depl['Revenues']

    # H2 PTC
    smr_depl['H2 PTC'] = utils.h2_ptc*smr_depl['Remaining H2 Dem. (kg/h)']*8760

    # Compute total deployed SMR capacity 
    smr_depl['Depl. SMR Cap. (MWe)'] = smr_depl['Depl. SMR Cap. (MWe)']+smr_depl[ 'SMR_Capacity_e']
    smr_depl['Depl. SMR Cap. (MWt)'] = smr_depl['Depl. SMR Cap. (MWt)']+smr_depl[ 'SMR_Capacity']
    smr_depl['Surplus SMR Cap. (MWe)'] = smr_depl['Surplus SMR Cap. (MWe)']+smr_depl['Surplus_Capacity_e']
    smr_depl['Surplus SMR Cap. (MWt)'] = smr_depl['Surplus SMR Cap. (MWt)']+smr_depl['Surplus_Capacity']
    smr_depl['SMR Modules'] = smr_depl['SMR Modules']+smr_depl['Modules']

    smr_depl['SMR direct heat cost ($/year)'] = smr_depl['Annual_CAPEX']+smr_depl[f'FOPEX']+smr_depl[f'VOPEX']
    if with_PTC:
        smr_depl['Net Ann. Rev. ($/year)'] = -smr_depl['SMR-H2 Total Cost ($/year)']+smr_depl['H2 PTC']+smr_depl['Avoided NG Cost']\
                                        -smr_depl['SMR direct heat cost ($/year)']
    else:
        smr_depl['Net Ann. Rev. ($/year)'] = -smr_depl['SMR-H2 Total Cost ($/year)']+smr_depl['Avoided NG Cost']\
                                        -smr_depl['SMR direct heat cost ($/year)']
    if cogen: 
        smr_depl = compute_cogen_revenues(smr_depl, 'Surplus SMR Cap. (MWe)', 'STATE',cambium_scenario, year)
        smr_depl['Net Ann. Rev. ($/year)'] += smr_depl['Electricity revenues ($/y)']
    return smr_depl


def compute_irr(smr_depl):
    smr_depl['IRR w PTC'] = smr_depl.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC'], x['Avoided NG Cost ($/y)']), axis=1)
    smr_depl['IRR wo PTC'] = smr_depl.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC'], x['Avoided NG Cost ($/y)'], ptc=False), axis=1)
    return smr_depl


def select_best_h2(smr_depl):
    smr_depl.reset_index(inplace=True,drop=True)
    idx = smr_depl.groupby(['FACILITY_ID', 'Remaining_temp_degC', 'SMR'])['Net Ann. Rev. ($/year)'].idxmax()
    max_h2 = smr_depl.loc[idx]
    max_h2 = max_h2.reset_index(drop=True)
    return max_h2

def select_best_smr(smr_depl):
   # Select SMR design corresponding to the maximum net annual revenue
    smr_depl.reset_index(inplace=True, drop=True)
    idx = smr_depl.groupby(['FACILITY_ID'])['Net Ann. Rev. ($/year)'].idxmax()
    max_SMR = smr_depl.loc[idx]
    max_SMR = max_SMR.reset_index(drop=True)  
    return max_SMR   
    
def compute_ng_breakeven(smr_depl,cogen):
    if cogen: 
        smr_depl['Breakeven NG price ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion']+smr_depl[f'SMR direct heat cost ($/year)']\
                                                        -smr_depl['H2 PTC']-smr_depl['Electricity revenues ($/y)'])/\
                                                    (smr_depl['NG_HLMP_mod']*(smr_depl['Heat_demand_MWh/hr']+smr_depl['Remaining_Heat_MW'])*8760)
        smr_depl['BE wo PTC ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion']+smr_depl[f'SMR direct heat cost ($/year)']\
                                                        -smr_depl['Electricity revenues ($/y)'])/(smr_depl['NG_HLMP_mod']*(smr_depl['Heat_demand_MWh/hr']+smr_depl['Remaining_Heat_MW'])*8760)
    else: 
        smr_depl['Breakeven NG price ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion']+smr_depl[f'SMR direct heat cost ($/year)']\
                                                        -smr_depl['H2 PTC'])/(smr_depl['NG_HLMP_mod']*(smr_depl['Heat_demand_MWh/hr']+smr_depl['Remaining_Heat_MW'])*8760)
        smr_depl['BE wo PTC ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion']+smr_depl[f'SMR direct heat cost ($/year)'])/\
                                                    (smr_depl['NG_HLMP_mod']*(smr_depl['Heat_demand_MWh/hr']+smr_depl['Remaining_Heat_MW'])*8760)
    return smr_depl


def main(OAK,with_PTC,cogen,cambium_scenario,year):
    if cogen: cogen_tag = 'cogen'
    else: cogen_tag = 'nocogen'
    if with_PTC: ptc_tag = 'PTC'
    else: ptc_tag = 'noPTC'
    # Import SMR-H2 data
    techs = import_smr_h2_data(OAK)
    # Import NG prices
    ng_prices = get_ng_prices(year)
    # Get coordinates of facilities
    loc_data = get_coordinates_facilities()
    # Load batched facility data
    nrel_data = get_nrel_data()
    # Load max results computing the cashflows for each type of SMR for each facility
    direct_heat_results = get_direct_heat_results()
    # Compute H2 demand for unserved heat demand
    direct_heat_results['Remaining H2 Dem. (kg/h)'] = direct_heat_results.apply(lambda x:compute_h2_demand(x['Remaining_Heat_MW'], x['Remaining_temp_degC']), axis=1)
    # Do not include HI or AK, no elec price data for those states
    direct_heat_results = direct_heat_results[~direct_heat_results['STATE'].isin(['HI', 'AK'])]
    # Compute the deployment of SMR required to serve the remaining H2 demand
    smr_depl = compute_smr_depl(direct_heat_results, techs)
    # Compute cashflows and IRR
    smr_depl = compute_cashflows(smr_depl,with_PTC,cogen,cambium_scenario,year)
    smr_depl = compute_irr(smr_depl)
    # Select best h2 technologies
    smr_depl = select_best_h2(smr_depl)
    # Select best SMR design at each location
    smr_depl = select_best_smr(smr_depl)
    print(smr_depl)
    # Add location data
    smr_depl = smr_depl.merge(loc_data, on=['CITY', 'STATE'])
    # Compute BE NG price
    smr_depl = compute_ng_breakeven(smr_depl,cogen)
    smr_depl.to_csv(f'./results/process_heat_direct_heat_h2comp_{OAK}_{ptc_tag}_{cogen_tag}.csv')



if __name__ == '__main__':
    OAK = 'FOAK_act'
    with_PTC = False
    cogen = True
    cambium_scenario = 'MidCase'
    year = 2024
    main(OAK,with_PTC,cogen,cambium_scenario,year)
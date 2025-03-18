import pandas as pd
import numpy as np
import math
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from process_heat_direct_heat_h2comp import compute_h2_demand, get_coordinates_facilities, import_smr_h2_data, compute_cogen_revenues

def get_nrel_data():
    nrel_data = pd.read_csv('./input_data/direct_heat_maxv/NREL_base_facilities_2.csv', encoding = "ISO-8859-1")
    nrel_data = nrel_data[['FACILITY_ID', 'FUEL_TYPE',
        'FUEL_TYPE_BLEND', 'FUEL_TYPE_OTHER', 
        'Natural_gas', 'Other', 'REPORTING_YEAR', 'CITY',
        'STATE', 'Temp_degC', 'Total', 'UNIT_NAME', 'MMTCO2E']]
    nrel_data = nrel_data.loc[nrel_data.FUEL_TYPE.isin(['Natural Gas (Weighted U.S. Average)'])]
    nrel_data = nrel_data[nrel_data.REPORTING_YEAR ==2015]
    nrel_data.drop(nrel_data.index[(nrel_data["Total"] ==0)],axis=0,inplace=True)
    nrel_data.drop(columns=['FUEL_TYPE',
        'FUEL_TYPE_BLEND', 'FUEL_TYPE_OTHER', 
        'Natural_gas', 'Other', 'REPORTING_YEAR'], inplace=True)
    # Group units with same temperature in same facility
    nrel_data['Heat Demand (MW)'] = nrel_data.apply(lambda x: x['Total']*1.1*277.778/8670, axis=1)
    nrel_data = nrel_data.groupby(['FACILITY_ID', 'STATE', 'CITY', 'Temp_degC']).sum(numeric_only=True)
    # compute MW from Total in TJ/y, distribution losses add 10%
    nrel_data.reset_index(inplace=True)
    return nrel_data


def get_ng_prices(year):
    ng_prices = pd.read_excel('./input_data/eia_aeo_industrial_sector_ng_prices.xlsx', sheet_name='state_prices')
    # 2024 prices
    ng_prices = ng_prices[ng_prices.year == year]
    ng_prices.rename(columns={'price 2020USD/MMBtu':'NG price ($/MMBtu)'}, inplace=True)
    return ng_prices


def compute_eq_h2_dem(nrel_data):
    # For each facility, temperature and heat compute hydrogen demand and sum up for each facility (sum emissions too)
    nrel_data['Total H2 Dem. (kg/h)'] = nrel_data.apply(lambda x:compute_h2_demand(x['Heat Demand (MW)'], x['Temp_degC']), axis=1)
    return nrel_data


def compute_avoided_ng_costs(nrel_data, ng_prices):
    def NGTempCostCurve(Temp,NG_Cost = 1.0, AHF_Coeffs = [0,-0.00038,0.90556]):
        HHV = 54 # MJ/kg
        Density = 0.68 # kg/m3
        cfTom3 = 35.31 # Unit conversion
        AHF = AHF_Coeffs[0]*(int(Temp)^2) + AHF_Coeffs[1]*int(Temp) + AHF_Coeffs[2] # avaialble Heat fraction - Deep Patel Equation
        
        HHV = HHV*Density*(1/cfTom3)*(1/1000000)*(1000) # returns  TJ/thousand cf 
        Cost = NG_Cost*(1/HHV)*(1/AHF)*(1/277.778) # returns the Cost in $/MWh
        return Cost
    nrel_data['NG_HLMP_mod'] = nrel_data.apply(lambda x: NGTempCostCurve(x['Temp_degC']), axis=1) 
    nrel_data = nrel_data.merge(ng_prices, left_on='STATE', right_on='state')
    nrel_data = nrel_data[~nrel_data['STATE'].isin(['HI', 'AK'])]
    nrel_data['Avoided NG Cost ($/y)'] = nrel_data['NG price ($/MMBtu)']*nrel_data['NG_HLMP_mod']*nrel_data['Heat Demand (MW)']*8760
    return nrel_data


def compute_mean_ng_hlmp_mod(nrel_data):
    ng_hlmp = nrel_data[['FACILITY_ID', 'NG_HLMP_mod']]
    mean_hlmp = ng_hlmp.groupby(['FACILITY_ID']).max()
    mean_hlmp.rename(columns={'NG_HLMP_mod':'mean_NG_HLMP_mod'}, inplace=True)
    mean_hlmp.reset_index(inplace=True)
    return mean_hlmp


def compute_cashflows(smr_depl, cogen, with_PTC, ITC, cambium_scenario, year):
    smr_depl['H2 Modules'] = smr_depl.apply(lambda x: math.ceil(x['Total H2 Dem. (kg/h)']/x['H2Cap (kgh2/h)']), axis=1)
    smr_depl['Depl. H2 Cap. (kgh2/h)'] = smr_depl['H2 Modules']*smr_depl['H2Cap (kgh2/h)']
    smr_depl['Depl. H2 Cap. (MWe)'] = smr_depl['H2 Modules']*smr_depl['H2Cap (MWe)']
    smr_depl['SMR Modules'] = smr_depl.apply(lambda x: math.ceil(x['Depl. H2 Cap. (MWe)']/x['Power in MWe']), axis=1)
    smr_depl['Depl. SMR Cap. (MWe)'] = smr_depl['SMR Modules']*smr_depl['Power in MWe']
    smr_depl['Depl. SMR Cap. (MWt)'] = smr_depl['SMR Modules']*smr_depl['Power in MWt']
    smr_depl['Surplus SMR Cap. (MWe)'] = smr_depl['Depl. SMR Cap. (MWe)']-smr_depl['Depl. H2 Cap. (MWe)']
    smr_depl['Surplus SMR Cap. (MWt)'] = smr_depl['Surplus SMR Cap. (MWe)']/smr_depl['Thermal Efficiency']
    # Compute costs in $/year
    # Capital recovery factor
    IR = utils.WACC
    smr_depl['SMR CRF'] = (IR/(1-((1+IR)**(-1*(smr_depl['Life (y)_y'])))))
    smr_depl['H2 CRF'] = (IR/(1-((1+IR)**(-1*(smr_depl['Life (y)_x'])))))

    # ITC
    itc_SMR, itc_h2 = ITC, ITC

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
    smr_depl['SMR VOM'] =smr_depl['Total H2 Dem. (kg/h)']*8760*smr_depl['Eq tot H2ElecCons (MWhe/kgh2)']*smr_depl['SMR VOM ($/MWhe)']
    smr_depl['H2 VOM'] =smr_depl['Total H2 Dem. (kg/h)']*8760*smr_depl['H2 VOM ($/MWhe)']
    smr_depl['SMR-H2 VOM'] = smr_depl['SMR VOM']+smr_depl['H2 VOM']

    ## Conversion costs
    gf_crf = IR/ (1 - (1/(1+IR)**utils.GFLT) ) 
    smr_depl['Conversion'] = utils.GFCAPEX*gf_crf*(1-itc_h2)*smr_depl['Heat Demand (MW)']
    smr_depl['Conversion total'] = utils.GFCAPEX*(1-itc_h2)*smr_depl['Heat Demand (MW)']

    # Total initial investment
    smr_depl['Initial investment ($)'] = smr_depl['Conversion total']+smr_depl['Tot SMR CAPEX']+smr_depl['Tot H2 CAPEX']


    # Total cost
    smr_depl['SMR-H2 Total Cost ($/year)'] = smr_depl['Annual SMR-H2 CAPEX'] + smr_depl['SMR-H2 FOM'] + smr_depl['SMR-H2 VOM'] +smr_depl['Conversion']
    # Compute revenues in $/year
    # Avoided NG cost computed from input data
    # H2 PTC
    smr_depl['H2 PTC'] = utils.h2_ptc*smr_depl['Total H2 Dem. (kg/h)']*8760

    # Cogeneration of electricity
    if with_PTC:
        smr_depl['SMR-H2 Net Ann. Rev. ($/year)'] = -smr_depl['SMR-H2 Total Cost ($/year)']+smr_depl['H2 PTC']+smr_depl['Avoided NG Cost ($/y)']
    else: 
        smr_depl['SMR-H2 Net Ann. Rev. ($/year)'] = -smr_depl['SMR-H2 Total Cost ($/year)']+smr_depl['Avoided NG Cost ($/y)']
    if cogen: 
        smr_depl = compute_cogen_revenues(smr_depl, 'Surplus SMR Cap. (MWe)', 'STATE',cambium_scenario,year)
        smr_depl['SMR-H2 Net Ann. Rev. ($/year)'] += smr_depl['Electricity revenues ($/y)']
    # Compute IRR
    smr_depl['IRR w PTC'] = smr_depl.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC'], x['Avoided NG Cost ($/y)']), axis=1)
    smr_depl['IRR wo PTC'] = smr_depl.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC'], x['Avoided NG Cost ($/y)'], ptc=False), axis=1)
    return smr_depl


def select_best_h2(smr_depl):
    # Select h2 tech corresponding to the maximum net annual revenue
    smr_depl.reset_index(inplace=True, drop=True)
    idx = smr_depl.groupby(['FACILITY_ID', 'SMR'])['SMR-H2 Net Ann. Rev. ($/year)'].idxmax()
    max_h2 = smr_depl.loc[idx]
    smr_depl = max_h2.reset_index(drop=True) 
    return smr_depl 


def select_best_smr(smr_depl):
    # Select SMR design corresponding to the maximum net annual revenue
    smr_depl.reset_index(inplace=True, drop=True)
    idx = smr_depl.groupby(['FACILITY_ID'])['SMR-H2 Net Ann. Rev. ($/year)'].idxmax()
    max_SMR = smr_depl.loc[idx]
    smr_depl = max_SMR.reset_index(drop=True)     
    return smr_depl


def compute_be_ng_prices(smr_depl, cogen):
    # Compute breakeven ng price
    if cogen: 
        smr_depl['Breakeven NG price ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion']\
                                                        -smr_depl['H2 PTC']-smr_depl['Electricity revenues ($/y)'])/(smr_depl['mean_NG_HLMP_mod']*smr_depl['Heat Demand (MW)']*8760)
        smr_depl['BE wo PTC ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion']\
                                    -smr_depl['Electricity revenues ($/y)'])/(smr_depl['mean_NG_HLMP_mod']*smr_depl['Heat Demand (MW)']*8760)
    else: 
        smr_depl['Breakeven NG price ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion']\
                                                        -smr_depl['H2 PTC'])/(smr_depl['mean_NG_HLMP_mod']*smr_depl['Heat Demand (MW)']*8760)
        smr_depl['BE wo PTC ($/MMBtu)'] = (smr_depl['Annual SMR-H2 CAPEX']+smr_depl['SMR-H2 FOM']+smr_depl['SMR-H2 FOM']+smr_depl['Conversion'])/\
                                                    (smr_depl['mean_NG_HLMP_mod']*smr_depl['Heat Demand (MW)']*8760)
    return smr_depl


def main(OAK,with_PTC,cogen,ITC,cambium_scenario,year):
    if cogen: cogen_tag = 'cogen'
    else: cogen_tag = 'nocogen'
    if with_PTC: ptc_tag = 'PTC'
    else: ptc_tag = 'noPTC'
    # Get heat demand data
    nrel_data = get_nrel_data()
    # Compute equivalent h2 demand to serve all the heat demand
    nrel_data['Total H2 Dem. (kg/h)'] = nrel_data.apply(lambda x:compute_h2_demand(x['Heat Demand (MW)'], x['Temp_degC']), axis=1)
    # Compute avoided NG costs
    nrel_data = compute_avoided_ng_costs(nrel_data, ng_prices=get_ng_prices(year))
    
    # Merge NG HMLP and location data to demand data
    nrel_data = nrel_data.groupby(['FACILITY_ID', 'STATE', 'CITY']).sum()
    nrel_data.reset_index(inplace=True)
    nrel_data = nrel_data.merge(compute_mean_ng_hlmp_mod(nrel_data), on='FACILITY_ID')
    nrel_data.drop(columns=['Temp_degC', 'NG_HLMP_mod','state', 'year'], inplace=True)
    nrel_data = nrel_data.merge(get_coordinates_facilities(), on=['STATE', 'CITY'])
    # Get SMR and H2 techs data
    techs = import_smr_h2_data(OAK)
    # Get the cross production of demand and SMR-H2 data to consider all possible combinations
    smr_depl = nrel_data.merge(techs, how='cross') # full facilities
    # Compute cashflows in $/year
    smr_depl = compute_cashflows(smr_depl, cogen, with_PTC, ITC, cambium_scenario, year)
    # Select best h2 technologies
    smr_depl = select_best_h2(smr_depl)
    # Select best SMR design
    smr_depl = select_best_smr(smr_depl)
    # Compute BE NG prices
    smr_depl = compute_be_ng_prices(smr_depl,cogen)
    smr_depl.to_csv(f'./results/process_heat_h2all_{OAK}_{ptc_tag}_{cogen_tag}.csv', index=False)

if __name__ == '__main__':
    OAK = utils.LEARNING
    with_PTC = utils.with_PTC
    ITC = utils.ITC
    cogen = True
    cambium_scenario = 'MidCase'
    year = 2024
    main(OAK,with_PTC,cogen,ITC,cambium_scenario,year)
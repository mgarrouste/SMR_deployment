import pandas as pd
from process_heat_direct_heat_h2comp import main as run_direct_heat_h2comp
from process_heat_h2all import main as run_h2all


def load_results(OAK,with_PTC,cogen,cambium_scenario,year):
    if cogen: cogen_tag = 'cogen'
    else: cogen_tag = 'nocogen'
    if with_PTC: ptc_tag = 'PTC'
    else: ptc_tag = 'noPTC'
    try:
        h2comp = pd.read_csv(f'./results/process_heat_direct_heat_h2comp_{OAK}_{ptc_tag}_{cogen_tag}.csv')
    except FileNotFoundError:
        run_direct_heat_h2comp(OAK,with_PTC,cogen,cambium_scenario,year)
        h2comp = pd.read_csv(f'./results/process_heat_direct_heat_h2comp_{OAK}_{ptc_tag}_{cogen_tag}.csv')
    try:
        h2all = pd.read_csv(f'./results/process_heat_h2all_{OAK}_{ptc_tag}_{cogen_tag}.csv')
    except FileNotFoundError:
        run_h2all(OAK,with_PTC,cogen,cambium_scenario,year)
        h2all = pd.read_csv(f'./results/process_heat_h2all_{OAK}_{ptc_tag}_{cogen_tag}.csv')
    h2comp['CAPEX ($/y)'] = h2comp['Annual_CAPEX']+h2comp['Annual SMR-H2 CAPEX']
    h2comp['O&M ($/y)'] =h2comp['FOPEX']+h2comp['VOPEX']+h2comp['SMR-H2 FOM']+h2comp['SMR-H2 VOM']
    h2comp.rename(columns={'Remaining H2 Dem. (kg/h)':'H2 Dem. (kg/h)', 'Heat_demand_MWh/hr':'Heat Demand (MW)', 'Highest_Temp_served_degC':'max_temp_degC', 'SMR':'SMR'}, inplace=True)
    h2comp['Pathway'] = 'Direct heat+H2'
    h2comp['Pathway Net Ann. Rev. ($/year)'] = h2comp['Net Ann. Rev. ($/year)']
    h2all['O&M ($/y)'] = h2all['SMR-H2 FOM']+h2all['SMR-H2 VOM']
    h2all.rename(columns={'Annual SMR-H2 CAPEX':'CAPEX ($/y)', 'MMTCO2E':'Emissions_mmtco2/y', 'Total H2 Dem. (kg/h)':'H2 Dem. (kg/h)', 
                   'SMR':'SMR'}, inplace=True)
    h2all['Pathway'] = 'H2'
    h2all['Pathway Net Ann. Rev. ($/year)'] = h2all['SMR-H2 Net Ann. Rev. ($/year)']
    return h2comp, h2all

def main(OAK,with_PTC,cogen,cambium_scenario,year):
    if cogen: cogen_tag = 'cogen'
    else: cogen_tag = 'nocogen'
    if with_PTC: ptc_tag = 'PTC'
    else: ptc_tag = 'noPTC'
    h2comp, h2all = load_results(OAK,with_PTC,cogen,cambium_scenario,year)
    comparison = pd.concat([h2comp,h2all],ignore_index=True)
    comparison.reset_index(inplace=True, drop=True)
    idx = comparison.groupby(['FACILITY_ID'])['Pathway Net Ann. Rev. ($/year)'].idxmax()
    best_pathway = comparison.loc[idx]
    best_pathway['Pathway Net Ann. Rev. (M$/y)'] = best_pathway['Pathway Net Ann. Rev. ($/year)']/1e6
    best_pathway['Pathway Net Ann. Rev. (M$/y/MWt)'] = best_pathway['Pathway Net Ann. Rev. (M$/y)']/best_pathway['Depl. SMR Cap. (MWt)']
    best_pathway['Pathway Net Ann. Rev. (M$/y/MWe)'] = best_pathway['Pathway Net Ann. Rev. (M$/y)']/best_pathway['Depl. SMR Cap. (MWe)']
    best_pathway.to_csv(f'./results/process_heat_{OAK}_{ptc_tag}_{cogen_tag}.csv')
    


if __name__ == '__main__':
    OAK = 'FOAK_act'
    with_PTC = False
    cogen = True
    cambium_scenario = 'MidCase'
    year = 2024
    main(OAK,with_PTC,cogen,cambium_scenario,year)
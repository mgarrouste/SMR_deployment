import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def pp_refining(results_path, clean_save_path, OAK, ITC):
    df = pd.read_excel(results_path, sheet_name='refining')
    df.sort_values(by=['Breakeven price ($/MMBtu)'], inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    smr_in = pd.read_excel('SMR_inputs.xlsx', sheet_name=OAK)
    smr_thermal_power = smr_in[['Power in MWe', 'Type']]
    df = df.merge(smr_thermal_power, how='inner', left_on=['SMR type'], right_on=['Type'])
    df.sort_values(by=['Breakeven price ($/MMBtu)'], inplace=True)
    df['Deployed Power (MWe)'] = df['Power in MWe']*df['# SMR modules']
    df['SMR Nameplate Capacity (GWe)'] = df['Deployed Power (MWe)'].cumsum()/1000
    df['Cum h2 dem (t/day)'] = df['H2 Dem. (kg/day)'].cumsum()/1000
    df['Cum h2 dem (%)'] = 100*df['H2 Dem. (kg/day)'].cumsum()/df['H2 Dem. (kg/day)'].sum()
    df['Viable SMR modules count'] = df['# SMR modules'].cumsum()
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df['Viable SMR refineries count'] = df.index+1
    SMR_carbon_intensity = 11.888 #kgCO2eq/kgH2
    df['Ann. avoided CO2 emissions (MMT-CO2/year)'] = (df['H2 Dem. (kg/day)']*365*SMR_carbon_intensity - df['Ann. CO2 emissions (kgCO2eq/year)'])/1e9
    df['Viable avoided emissions (MMT-CO2/year)'] = df['Ann. avoided CO2 emissions (MMT-CO2/year)'].cumsum()
    df[['Breakeven price ($/MMBtu)', 'Viable avoided emissions (MMT-CO2/year)']].tail(110)
    res_be = pd.read_csv('./results/res_be_refining.csv', index_col='RES')
    def find_be_h2_demand(df, percentage):
        indleft = df['Cum h2 dem (%)'].sub(percentage).abs().idxmin()
        be = df['Breakeven price ($/MMBtu)'][indleft]
        return be
    for percent in [10, 25, 50, 75, 90,100]:
        res_be.loc[str(percent), 'Breakeven price ($/MMBtu)'] = find_be_h2_demand(df, percent)
    sheetn= 'refining'
    excelf = './results/res_be_comparison.xlsx'
    try:
        with pd.ExcelFile(excelf, engine='openpyxl') as xls:
            with pd.ExcelWriter(excelf, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                res_be.to_excel(writer, sheet_name=sheetn)
    except FileNotFoundError:
        res_be.to_excel(excelf, sheet_name=sheetn)
    df = utils.compute_cogen(df, surplus_cap_col_name='Surplus SMR Cap. (MWe)', state_col_name='state', \
                            cambium_scenario=cambium_scenario, year = year)
    df['Net Revenues with H2 PTC with elec ($/year)'] = df['Net Revenues with H2 PTC ($/year)']+df['Electricity revenues ($/y)']
    def compute_ref_capex_breakeven(SMR_crf, capacity, h2_capex, SMR_om, h2_om, conv, affc, h2ptc, elec, ptc=True, ITC=ITC):
        alpha = SMR_crf*capacity*(1-ITC)
        # Costs without SMR capex
        SMRh2_costs = h2_capex+SMR_om+h2_om+conv
        # BE CAPEX $/MWe
        if ptc: 
            be_capex = (affc + elec+ h2ptc - SMRh2_costs)/alpha
        else:
            be_capex = (affc+ elec - SMRh2_costs)/alpha
        return be_capex
    df['Breakeven CAPEX ($/MWe)'] = df.apply(lambda x: compute_ref_capex_breakeven(
                                    x['SMR CRF'], x['Depl. SMR Cap. (MWe)'], x['H2 CAPEX ($/year)'], x['SMR O&M ($/year)'], x['H2 O&M ($/year)'], 
                                    x['Conversion costs ($/year)'], x['Avoided NG costs ($/year)'], x['H2 PTC Revenues ($/year)'], 
                                    x['Electricity revenues ($/y)']), axis=1)
    df['Breakeven CAPEX wo PTC ($/MWe)'] = df.apply(lambda x: compute_ref_capex_breakeven(
                                    x['SMR CRF'], x['Depl. SMR Cap. (MWe)'], x['H2 CAPEX ($/year)'], x['SMR O&M ($/year)'], x['H2 O&M ($/year)'], 
                                    x['Conversion costs ($/year)'], x['Avoided NG costs ($/year)'], x['H2 PTC Revenues ($/year)'], 
                                    x['Electricity revenues ($/y)'], ptc=False), axis=1)
    df['IRR w PTC'] = df.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC Revenues ($/year)'], x['Avoided NG costs ($/year)']), axis=1)
    df['IRR wo PTC'] = df.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC Revenues ($/year)'], x['Avoided NG costs ($/year)'], ptc=False), axis=1)
    df_clean = df[["id", 'state',  'latitude', 'longitude','H2 Dem. (kg/day)', 'HTSE', 'SMR type', '# SMR modules', \
                'Electricity revenues ($/y)','Net Revenues ($/year)','Net Revenues with H2 PTC ($/year)', 'Breakeven CAPEX ($/MWe)',\
                    'Ann. avoided CO2 emissions (MMT-CO2/year)', 'Breakeven price ($/MMBtu)',\
                        'Viable avoided emissions (MMT-CO2/year)','Net Revenues with H2 PTC with elec ($/year)']]

    df_clean['H2 Dem. (kg/day)'] /=1e3
    df_clean['HTSE'] = df_clean['HTSE'].apply(lambda x: int(x))
    df_clean['H2 Dem. (kg/day)'] = df_clean['H2 Dem. (kg/day)'].apply(lambda x:np.round(x,1))
    df_clean['Net Revenues ($/year)'] /=1e6
    df_clean['Avoided cost of CO2 ($/ton)'] = df_clean['Net Revenues ($/year)']/df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)']
    df_clean['Avoided cost of CO2 ($/ton)'] = df_clean['Avoided cost of CO2 ($/ton)'].apply(lambda x:np.round(np.abs(x),1))
    df_clean['Net Revenues ($/year)'] = df_clean['Net Revenues ($/year)'].apply(lambda x: np.abs(np.round(x,1)))
    df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)']= df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)'].apply(lambda x: np.round(x,1))
    df_clean['Breakeven price ($/MMBtu)'] = df_clean.apply(lambda x:np.round(x['Breakeven price ($/MMBtu)'],1), axis=1)
    df_clean.rename(columns={'RH2 Dem. (kg/day)': 'Demand (MT H2/day)',
                            'SMR type': 'SMR', 
                            '# SMR modules': '#'}, inplace=True)
    df_clean.sort_values(by=['Breakeven price ($/MMBtu)'], inplace=True)
    sheet_name = 'refining'
    excel_file = clean_save_path
    try:
    # Load the existing Excel file
        with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
            # Check if the sheet exists
            if sheet_name in xls.sheet_names:
                # If the sheet exists, replace the data
                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name)
            else:
                # If the sheet doesn't exist, create a new sheet
                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                    df.to_excel(writer, sheet_name=sheet_name)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and write the DataFrame to it
        df.to_excel(excel_file, sheet_name=sheet_name)


def pp_ammonia(results_path, clean_save_path, OAK, ITC):
    df = pd.read_excel(results_path, sheet_name='ammonia')
    df.sort_values(by=['Breakeven price ($/MMBtu)'], inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    smr_in = pd.read_excel('SMR_inputs.xlsx', sheet_name=OAK)
    SMR_thermal_power = smr_in[['Power in MWt', 'Power in MWe', 'Type']]
    df = df.merge(SMR_thermal_power, how='inner', left_on=['SMR type'], right_on=['Type'])
    df.sort_values(by=['Breakeven price ($/MMBtu)'], inplace=True)
    df['Deployed Power (MWt)'] = df['Power in MWt']*df['# SMR modules']
    df['Deployed Power (MWe)'] = df['Power in MWe']*df['# SMR modules']
    df['SMR Nameplate Capacity (GWe)'] = df['Deployed Power (MWe)'].cumsum()/1000
    df['Cum h2 dem (t/day)'] =df['H2 Dem. (kg/day)'].cumsum()/1000
    df['Cum h2 dem (%)'] = 100*df['H2 Dem. (kg/day)'].cumsum()/df['H2 Dem. (kg/day)'].sum()
    df['Viable SMR modules count'] = df['# SMR modules'].cumsum()
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df['Viable SMR ammonia plants count'] = df.index+1
    ammonia_demand_df = pd.read_excel('./h2_demand_ammonia_us_2022.xlsx', sheet_name='processed')
    df = df.merge(ammonia_demand_df, on='id')
    ratio_co2_to_nh3 = 2.30 # tCO2eq/tNH3
    df['NG path GHG (tCO2/year)'] = df.apply(lambda x:x['Capacity (tNH3/year)']*2.30, axis=1)
    df['Ann. avoided CO2 emissions (MMT-CO2/year)'] = (df['NG path GHG (tCO2/year)'] - (df['Ann. CO2 emissions (kgCO2eq/year)']/1e3))/1e6
    df['Viable avoided emissions (MMT-CO2/year)'] = df['Ann. avoided CO2 emissions (MMT-CO2/year)'].cumsum()
    df[['Breakeven price ($/MMBtu)', 'Viable avoided emissions (MMT-CO2/year)']].tail(50)
    res_be = pd.read_csv('./results/res_be_ammonia.csv', index_col='RES')
    def find_be_h2_demand(df, percentage):
        indleft = df['Cum h2 dem (%)'].sub(percentage).abs().idxmin()
        be = df['Breakeven price ($/MMBtu)'][indleft]
        return be
    for percent in [10, 25, 50, 75, 90,100]:
        res_be.loc[str(percent), 'Breakeven price ($/MMBtu)'] = find_be_h2_demand(df, percent)
    sheetn= 'ammonia'
    excelf = './results/res_be_comparison.xlsx'
    try:
        with pd.ExcelFile(excelf, engine='openpyxl') as xls:
            with pd.ExcelWriter(excelf, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                res_be.to_excel(writer, sheet_name=sheetn)
    except FileNotFoundError:
        res_be.to_excel(excelf, sheet_name=sheetn)
    df = utils.compute_cogen(df, surplus_cap_col_name='Surplus SMR Cap. (MWe)', state_col_name='state', \
                         cambium_scenario=cambium_scenario, year = year)
    df['Net Revenues with H2 PTC with elec ($/year)'] = df['Net Revenues with H2 PTC ($/year)']+df['Electricity revenues ($/y)']
    def compute_ammonia_capex_breakeven(SMR_crf, capacity, h2_capex, SMR_om, h2_om, conv, affc, h2ptc, elec, ptc=True, ITC=ITC):
        alpha = SMR_crf*capacity*(1-ITC)
        # Costs without SMR capex
        SMRh2_costs = h2_capex+SMR_om+h2_om+conv
        # BE CAPEX $/MWe
        if ptc: 
            be_capex = (affc + elec+ h2ptc - SMRh2_costs)/alpha
        else:
            be_capex = (affc+ elec - SMRh2_costs)/alpha
        return be_capex
    df['Breakeven CAPEX ($/MWe)'] = df.apply(lambda x: compute_ammonia_capex_breakeven(
                                            x['SMR CRF'], x['Depl. SMR Cap. (MWe)'], x['H2 CAPEX ($/year)'], x['SMR O&M ($/year)'], x['H2 O&M ($/year)'], 
                                            x['Conversion costs ($/year)'], x['Avoided NG costs ($/year)'], x['H2 PTC Revenues ($/year)'], 
                                            x['Electricity revenues ($/y)']), axis=1)
    df['Breakeven CAPEX wo PTC ($/MWe)'] = df.apply(lambda x: compute_ammonia_capex_breakeven(
                                            x['SMR CRF'], x['Depl. SMR Cap. (MWe)'], x['H2 CAPEX ($/year)'], x['SMR O&M ($/year)'], x['H2 O&M ($/year)'], 
                                            x['Conversion costs ($/year)'], x['Avoided NG costs ($/year)'], x['H2 PTC Revenues ($/year)'], 
                                            x['Electricity revenues ($/y)'], ptc=False), axis=1)
    df['IRR w PTC'] = df.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC Revenues ($/year)'], x['Avoided NG costs ($/year)']), axis=1)
    df['IRR wo PTC'] = df.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC Revenues ($/year)'], x['Avoided NG costs ($/year)'], ptc=False), axis=1)
    df_clean = df[["id", 'state', 'latitude_x', 'longitude_x','Capacity (tNH3/year)', 'H2 Dem. (kg/day)', 'HTSE', 'SMR type', '# SMR modules', \
               'Net Revenues ($/year)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 'Breakeven price ($/MMBtu)', 'Breakeven CAPEX ($/MWe)',\
                'Electricity revenues ($/y)','Net Revenues with H2 PTC with elec ($/year)',\
                'Viable avoided emissions (MMT-CO2/year)', 'Cum h2 dem (%)', 'Net Revenues with H2 PTC ($/year)']]
    df_clean['H2 Dem. (kg/day)'] /=1e3

    df_clean['HTSE'] = df_clean['HTSE'].apply(lambda x: int(x))

    df_clean['H2 Dem. (kg/day)'] = df_clean['H2 Dem. (kg/day)'].apply(lambda x:np.round(x,1))

    df_clean['Avoided cost of CO2 ($/ton)'] = df_clean['Net Revenues ($/year)']/(1e6*df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)'])
    df_clean['Avoided cost of CO2 ($/ton)'] = df_clean['Avoided cost of CO2 ($/ton)'].apply(lambda x:np.round(np.abs(x),1))

    df_clean['Net Revenues ($/year)'] = df_clean['Net Revenues ($/year)'].apply(lambda x: np.round(x/1e6,1))

    df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)']= df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)'].apply(lambda x: np.round(x,1))

    df_clean['Breakeven price ($/MMBtu)'] = df_clean['Breakeven price ($/MMBtu)'].apply(lambda x : np.round(x,1))

    df_clean.rename(columns={'H2 Dem. (kg/day)': 'Demand (MT H2/day)',
                            'Net Rev. ($/year)': 'Net Rev. (M$/year)', 
                            'SMR type': 'SMR', 
                            'latitude_x':'latitude', 'longitude_x':'longitude',
                            '# SMR modules': '#'}, inplace=True)
    df.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'}, inplace=True)
    sheet_name = 'ammonia'
    excel_file = clean_save_path
    try:
        # Load the existing Excel file
        with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and write the DataFrame to it
        df.to_excel(excel_file, sheet_name=sheet_name)


def pp_steel(results_path, clean_save_path, OAK, ITC):
    df = pd.read_excel(results_path, sheet_name='steel')
    df.sort_values(by=['Breakeven price ($/MMBtu)'], inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    SMR_param = pd.read_excel('SMR_inputs.xlsx', sheet_name=OAK)
    SMR_thermal_power = SMR_param[['Power in MWe', 'Type']]
    SMR_thermal_power
    df = df.merge(SMR_thermal_power, how='inner', left_on=['SMR type'], right_on=['Type'])
    df.sort_values(by=['Breakeven price ($/MMBtu)'], inplace=True)
    df['Deployed Power (MWe)'] = df['Power in MWe']*df['# SMR modules']
    df['SMR Nameplate Capacity (GWe)'] = df['Deployed Power (MWe)'].cumsum()/1000
    df['Cum h2 dem (t/day)'] = df['H2 Dem. (kg/day)'].cumsum()/1000
    df['Cum h2 dem (%)'] = 100*df['H2 Dem. (kg/day)'].cumsum()/df['H2 Dem. (kg/day)'].sum()
    df['Viable SMR modules count'] = df['# SMR modules'].cumsum()
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df['Viable SMR steel plants count'] = df.index+1
    demand_steel_df = pd.read_excel('./h2_demand_bfbof_steel_us_2022.xlsx', sheet_name='processed')
    df = df.merge(demand_steel_df, left_on='id', right_on='Plant')
    df['Ann. avoided CO2 emissions (MMT-CO2/year)'] = (df['GHG QUANTITY (METRIC TONS CO2e)'] - (df['Ann. CO2 emissions (kgCO2eq/year)']/1e3))/1e6
    df['Viable avoided emissions (MMT-CO2/year)'] = df['Ann. avoided CO2 emissions (MMT-CO2/year)'].cumsum()
    df = utils.compute_cogen(df, surplus_cap_col_name='Surplus SMR Cap. (MWe)', state_col_name='state', \
                         cambium_scenario=cambium_scenario, year = year)
    df['Net Revenues with H2 PTC with elec ($/year)'] = df['Net Revenues with H2 PTC ($/year)']+df['Electricity revenues ($/y)']
    df['Net Revenues wo PTC with elec ($/year)'] = df['Net Revenues ($/year)']+df['Electricity revenues ($/y)']
    df[['Net Revenues with H2 PTC with elec ($/year)', 'Net Revenues wo PTC with elec ($/year)']]
    def compute_breakeven_price(rev_woptc, steel_prod):
        costs = -rev_woptc# NEt revenues Negative by convention
        plant_cap = steel_prod
        COAL_CONS_RATE = 0.463
        breakeven_price_per_ton = (costs - utils.iron_ore_cost*utils.bfbof_iron_cons*plant_cap - utils.om_bfbof*plant_cap)/(COAL_CONS_RATE*plant_cap)
        breakeven_price = breakeven_price_per_ton/utils.coal_heat_content
        return breakeven_price
    df.drop(columns=['Breakeven price ($/MMBtu)'], inplace=True)
    df['Breakeven price ($/MMBtu)'] = df.apply(lambda x: compute_breakeven_price(x['Net Revenues with H2 PTC with elec ($/year)'], x['Steel prod. (ton/year)']), axis=1)
    def compute_be_wo_PTC(rev_woptc, steel_prod):
        costs = rev_woptc # NEt revenues Negative by convention
        plant_cap = steel_prod
        COAL_CONS_RATE = 0.463
        breakeven_price_per_ton = -(costs - utils.iron_ore_cost*utils.bfbof_iron_cons*plant_cap - utils.om_bfbof*plant_cap)/(COAL_CONS_RATE*plant_cap)
        breakeven_price = breakeven_price_per_ton/utils.coal_heat_content
        return breakeven_price
    df.drop(columns=['BE wo PTC ($/MMBtu)'], inplace=True)
    df['BE wo PTC ($/MMBtu)'] = df.apply(lambda x: compute_be_wo_PTC(x['Net Revenues wo PTC with elec ($/year)'], x['Steel prod. (ton/year)']), axis=1)
    df[['Net Revenues with H2 PTC with elec ($/year)', 'Net Revenues wo PTC with elec ($/year)', 'BE wo PTC ($/MMBtu)', 'Breakeven price ($/MMBtu)']]
    res_be = pd.read_csv('./results/res_be_steel.csv', index_col='RES')
    def find_be_h2_demand(df, percentage):
        indleft = df['Cum h2 dem (%)'].sub(percentage).abs().idxmin()
        be = df['Breakeven price ($/MMBtu)'][indleft]
        return be
    for percent in [10, 25, 50, 75, 90,100]:
        res_be.loc[str(percent), 'Breakeven price ($/MMBtu)'] = find_be_h2_demand(df, percent)
    sheetn= 'steel'
    excelf = './results/res_be_comparison.xlsx'
    try:
        with pd.ExcelFile(excelf, engine='openpyxl') as xls:
            with pd.ExcelWriter(excelf, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                res_be.to_excel(writer, sheet_name=sheetn)
    except FileNotFoundError:
        res_be.to_excel(excelf, sheet_name=sheetn)
    def compute_steel_capex_breakeven(SMR_crf, capacity, h2_capex, SMR_om, h2_om, conv, affc, h2ptc, elec, ptc=True, ITC=ITC):
        alpha = SMR_crf*capacity*(1-ITC)
        # Costs without SMR capex
        SMRh2_costs = h2_capex+SMR_om+h2_om+conv
        # BE CAPEX $/MWe
        if ptc: 
            be_capex = (affc + elec+ h2ptc - SMRh2_costs)/alpha
        else:
            be_capex = (affc+ elec - SMRh2_costs)/alpha
        return be_capex
    df['Breakeven CAPEX ($/MWe)'] = df.apply(lambda x: compute_steel_capex_breakeven(
                                        x['SMR CRF'], x['Depl. SMR Cap. (MWe)'], x['H2 CAPEX ($/year)'], x['SMR O&M ($/year)'], x['H2 O&M ($/year)'], 
                                        x['Conversion costs ($/year)'], x['Avoided NG costs ($/year)'], x['H2 PTC Revenues ($/year)'], 
                                        x['Electricity revenues ($/y)']), axis=1)
    df['Breakeven CAPEX wo PTC ($/MWe)'] = df.apply(lambda x: compute_steel_capex_breakeven(
                                        x['SMR CRF'], x['Depl. SMR Cap. (MWe)'], x['H2 CAPEX ($/year)'], x['SMR O&M ($/year)'], x['H2 O&M ($/year)'], 
                                        x['Conversion costs ($/year)'], x['Avoided NG costs ($/year)'], x['H2 PTC Revenues ($/year)'], 
                                        x['Electricity revenues ($/y)'], ptc=False), axis=1)
    df['IRR w PTC'] = df.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC Revenues ($/year)'], x['Avoided NG costs ($/year)']), axis=1)
    df['IRR wo PTC'] = df.apply(lambda x: utils.calculate_irr(x['Initial investment ($)'], x['Electricity revenues ($/y)'], x['H2 PTC Revenues ($/year)'], x['Avoided NG costs ($/year)'], ptc=False), axis=1)
    df_clean = df[["id", 'state',  'latitude_x', 'longitude_x','Steel production capacity (ttpa)', 'Electricity revenues ($/y)',\
                'H2 Dem. (kg/day)', 'HTSE', 'SMR type', '# SMR modules', 'Net Revenues ($/year)','Breakeven CAPEX ($/MWe)',\
                  'Net Revenues with H2 PTC ($/year)','Ann. avoided CO2 emissions (MMT-CO2/year)',\
                      'Breakeven price ($/MMBtu)', 'Viable avoided emissions (MMT-CO2/year)',
                      'Net Revenues with H2 PTC with elec ($/year)']]

    df_clean['H2 Dem. (kg/day)'] /=1e3
    df_clean['HTSE'] = df_clean['HTSE'].apply(lambda x: int(x))
    df_clean['H2 Dem. (kg/day)'] = df_clean['H2 Dem. (kg/day)'].apply(lambda x:np.round(x,1))
    df_clean['Net Revenues ($/year)'] = df_clean['Net Revenues ($/year)'].apply(lambda x: np.round(x/1e6,1))
    df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)']= df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)'].apply(lambda x: np.round(x,1))
    df_clean['Avoided cost of CO2 ($/ton)'] = df_clean['Net Revenues ($/year)']/df_clean['Ann. avoided CO2 emissions (MMT-CO2/year)']
    df_clean['Avoided cost of CO2 ($/ton)'] = df_clean['Avoided cost of CO2 ($/ton)'].apply(lambda x:np.round(np.abs(x),1))
    df_clean['Breakeven price ($/MMBtu)'] = df_clean['Breakeven price ($/MMBtu)'].apply(lambda x : np.round(x,1))
    df_clean.rename(columns={'H2 Dem (kg/day)': 'Demand (MT H2/day)', 
                            'SMR type': 'SMR', 'latitude_x':'latitude', 'longitude_x':'longitude',
                            '# SMR modules': '#'}, inplace=True)
    df.rename(columns={'latitude_x':'latitude', 'longitude_x':'longitude'}, inplace=True)
    sheet_name = 'steel'
    excel_file = clean_save_path
    try:
        # Load the existing Excel file
        with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
            # Check if the sheet exists
            if sheet_name in xls.sheet_names:
                # If the sheet exists, replace the data
                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # If the sheet doesn't exist, create a new sheet
                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and write the DataFrame to it
        df.to_excel(excel_file, sheet_name=sheet_name, index=False)


def main(OAK,wacc,ITC,cambium_scenario,year):
    results_path = f'./results/raw_results_SMR_{OAK}_ITC_{ITC}.xlsx'
    clean_save_path = f'./results/clean_results_SMR_{OAK}_ITC_{ITC}.xlsx'
    pp_refining(results_path, clean_save_path, OAK, ITC)
    pp_ammonia(results_path,clean_save_path, OAK, ITC)
    pp_steel(results_path, clean_save_path, OAK, ITC)


if __name__ == '__main__':
    OAK = utils.LEARNING
    with_PTC = utils.with_PTC
    ITC = utils.ITC
    cogen = True
    wacc = utils.WACC
    cambium_scenario = 'MidCase'
    year = 2024
    main(OAK,wacc,ITC,cambium_scenario,year)
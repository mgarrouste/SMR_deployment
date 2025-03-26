import pandas as pd
from process_heat_pp import main as run_heat_analysis
from utils import WACC


def load_h2_results(OAK, with_PTC, ITC):
    h2_results_path = f'./results/clean_results_SMR_{OAK}_ITC_{ITC}.xlsx'
    industries = ['refining','steel','ammonia']
    list_df = []
    for ind in industries:
        df = pd.read_excel(h2_results_path, sheet_name=ind, index_col='id')
        list_cols = ['state', 'latitude', 'longitude','Net Revenues ($/year)','Electricity revenues ($/y)','IRR',
                    'Net Annual Revenues with H2 PTC ($/MWe/y)','Depl. SMR Cap. (MWe)', 'SMR type', 
                    '# SMR modules', 'Breakeven price ($/MMBtu)', 'Net Revenues with H2 PTC with elec ($/year)', 'Breakeven CAPEX ($/MWe)']
        if with_PTC: 
            df = df.drop(columns=['BE wo PTC ($/MMBtu)', 'IRR wo PTC'])
            df = df.rename(columns = {'IRR w PTC':'IRR'})
        else:
            df = df.drop(columns = ['Breakeven price ($/MMBtu)', 'IRR w PTC','Breakeven CAPEX ($/MWe)'] )
            df = df.rename(columns={'BE wo PTC ($/MMBtu)':'Breakeven price ($/MMBtu)', 'IRR wo PTC':'IRR','Breakeven CAPEX wo PTC ($/MWe)':'Breakeven CAPEX ($/MWe)'})
        df = df[list_cols]
        df['Application'] = f'H2 - {ind}' 
        list_df.append(df)
    all_df = pd.concat(list_df)
    all_df['SMR'] = all_df['SMR type']
    all_df = all_df.drop(columns=['SMR type'])
    all_df['# SMR modules'] = all_df['# SMR modules'].astype(float)

    if with_PTC:
        all_df['Annual Net Revenues (M$/MWe/y)'] = all_df['Net Revenues with H2 PTC with elec ($/year)']/(1e6*all_df['Depl. SMR Cap. (MWe)'])
        all_df['Annual Net Revenues (M$/y)'] = all_df['Net Revenues with H2 PTC with elec ($/year)']/1e6
    else:
        all_df['Annual Net Revenues (M$/y)'] = all_df.apply(lambda x: (x['Net Revenues ($/year)']+x['Electricity revenues ($/y)'])/1e6, axis=1)
        all_df['Annual Net Revenues (M$/MWe/y)'] = all_df['Annual Net Revenues (M$/y)']/all_df['Depl. SMR Cap. (MWe)']
    all_df = all_df.drop(columns=['Electricity revenues ($/y)','Net Revenues with H2 PTC with elec ($/year)',
                                  'Net Annual Revenues with H2 PTC ($/MWe/y)','Net Revenues ($/year)'])
    all_df.sort_values(by='Breakeven price ($/MMBtu)', inplace=True)
    all_df.reset_index(inplace=True)
    return all_df


def load_heat_results(OAK, with_PTC=True, ITC=0.3):
    """Loads direct process heat results and returns them sorted by breakeven prices"""
    if with_PTC: ptc_tag = 'PTC'
    else: ptc_tag = 'noPTC'
    heat_results_path = f'./results/process_heat_{OAK}_{ptc_tag}_cogen_ITC_{ITC}.csv'
    try:
        heat_df = pd.read_csv(heat_results_path, index_col='FACILITY_ID')
    except FileNotFoundError:
        run_heat_analysis(OAK,with_PTC,cogen=True,ITC=ITC)
        heat_df = pd.read_csv(heat_results_path, index_col='FACILITY_ID')
    heat_df['Annual Net Revenues (M$/MWe/y)']  = heat_df['Pathway Net Ann. Rev. (M$/y)']/heat_df['Depl. SMR Cap. (MWe)']
    heat_df['Annual Net Revenues (M$/y)'] = heat_df['Pathway Net Ann. Rev. (M$/y)']
    heat_df['Application'] = 'Process Heat'
    list_cols = ['STATE','latitude', 'longitude', 'SMR','# SMR modules','Breakeven NG price ($/MMBtu)',
											 'Depl. SMR Cap. (MWe)','Annual Net Revenues (M$/y)', 'Annual Net Revenues (M$/MWe/y)','Application']
    if with_PTC: list_cols += ['IRR w PTC']
    else: list_cols += ['IRR wo PTC']
    heat_df = heat_df[list_cols]
    if with_PTC: heat_df = heat_df.rename(columns={'IRR w PTC':'IRR'})
    else: heat_df = heat_df.rename(columns={'IRR wo PTC':'IRR'})
    heat_df = heat_df.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)', 
                                      'STATE':'state'})
    heat_df.reset_index(inplace=True, names='id')
    return heat_df

def load_results(OAK,with_PTC,ITC):
    heat = load_heat_results(OAK,with_PTC,ITC)
    h2 = load_h2_results(OAK,with_PTC,ITC)
    df = pd.concat([heat, h2], ignore_index=True)
    return df


def exclude_foak_sites(noak_results, foak_results, tag):
    print(f"Exluding FOAK profitable sites from NOAK deployement phase for scenario {tag}")
    foak_deployed = foak_results[(foak_results['IRR']>=WACC*100)]
    foak_deployed_sites = foak_deployed['id'].unique()
    print(f'FOAK sites profitably deployed {len(foak_deployed)}')
    print(f'NOAK sites before removing FOAK deployed sites {len(noak_results)}')
    noak_results = noak_results[~(noak_results['id'].isin(foak_deployed_sites))]
    print(f'NOAK sites after removing FOAK deployed sites {len(noak_results)}')
    return noak_results

def main():
    # FOAK
    foak_noPTC = load_results('FOAK',with_PTC=False,ITC=0)
    foak_noPTC.to_excel('./results/all_FOAK_noPTC_ITC_0.xlsx', index=False)
    foak_PTC = load_results('FOAK',with_PTC=True,ITC=0.3)
    foak_PTC.to_excel('./results/all_FOAK_PTC_ITC_0.3.xlsx', index=False)
    foak_noPTC['tag'] = 'noPTC_noITC'
    foak_PTC['tag'] = 'PTC_ITC'
    foak = pd.concat([foak_PTC, foak_noPTC], ignore_index=True)
    foak.to_excel('./results/all_FOAK.xlsx')


    # NOAK
    # No incentives
    try:
        noak_noPTC = load_results('NOAK_wo_inc',with_PTC=False,ITC=0)
        noak_noPTC = exclude_foak_sites(noak_results=noak_noPTC, foak_results=foak_noPTC, tag="No incentive")
        noak_noPTC.to_excel('./results/all_NOAK_wo_inc_ITC_0.xlsx', index=False)
        # With 45V and 48E
        noak_PTC = load_results('NOAK_with_inc',with_PTC=True,ITC=0.3)
        noak_PTC = exclude_foak_sites(noak_results=noak_PTC, foak_results=foak_PTC, tag="With 45V and 48E")
        noak_PTC.to_excel('./results/all_NOAK_with_inc_ITC_0.3.xlsx', index=False)
        # Concatenate results for viz in Tableau
        noak_noPTC['tag'] = 'noPTC_noITC'
        noak_PTC['tag'] = 'PTC_ITC'
        noak = pd.concat([noak_PTC, noak_noPTC], ignore_index=True)
        noak.to_excel('./results/all_NOAK.xlsx')
    except FileNotFoundError:
        print('NOAK deployment phase not run yet, skip prepping data for Tableau viz')


if __name__ == '__main__':
    main()
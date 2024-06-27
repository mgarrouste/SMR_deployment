import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import palette, letter_annotation, cashflows_color_map 
import warnings
import pp_industrial_hydrogen


color_map = {'Industrial Hydrogen':'blue', 'Process Heat':'red', 'Total':'Green', 'FOAK':'limegreen', 
             'NOAK':'forestgreen','FOAK\nNo-Cogeneration':'limegreen','FOAK\nCogeneration':'lightsteelblue', 
           'NOAK\nNo-Cogeneration':'forestgreen','NOAK\nCogeneration':'slategrey'}

def load_elec_results(anr_tag):
  elec_results_path = f'./results/price_taker_{anr_tag}_MidCase.xlsx'
  elec_df = pd.read_excel(elec_results_path)
  elec_df['Annual Net Revenues (M$/MWe/y)'] = elec_df['Annual Net Revenues ($/year/MWe)']/1e6
  # Only keep the best design for each state and year
  df = elec_df.loc[elec_df.groupby('state')['Annual Net Revenues (M$/MWe/y)'].transform(max) == elec_df['Annual Net Revenues (M$/MWe/y)']]
  df['Application'] = 'Electricity'
  df.rename(columns={'ANR type':'ANR'}, inplace=True)
  return df
  

def load_h2_results(anr_tag, cogen_tag):
  """"Loads all hydrogen results and returns results sorted by breakeven prices"""
  h2_results_path = f'./results/clean_results_anr_{anr_tag}_h2_wacc_0.077.xlsx'
  industries = ['refining','steel','ammonia']
  list_df = []
  for ind in industries:
    df = pd.read_excel(h2_results_path, sheet_name=ind, index_col='id')
    list_cols = ['state', 'latitude', 'longitude','H2 Dem. (kg/day)','Net Revenues with H2 PTC ($/year)',\
                 'Net Revenues ($/year)','Electricity revenues ($/y)','IRR w PTC', 'IRR wo PTC',\
                  'Net Annual Revenues with H2 PTC ($/MWe/y)', 'HTSE', 'Depl. ANR Cap. (MWe)', 'ANR type', \
             '# ANR modules', 'Breakeven price ($/MMBtu)', 'BE wo PTC ($/MMBtu)','Ann. avoided CO2 emissions (MMT-CO2/year)', 'Electricity revenues ($/y)', \
             'Net Revenues with H2 PTC with elec ($/year)']
    if anr_tag == 'FOAK':
      list_cols.append('Breakeven CAPEX ($/MWe)')
      list_cols.append('Breakeven CAPEX wo PTC ($/MWe)')
      list_cols.append('State price ($/MMBtu)')
    df = df[list_cols]
    df['Industry'] = ind 
    list_df.append(df)
  all_df = pd.concat(list_df)
  all_df['Application'] = 'Industrial Hydrogen'
  all_df['ANR'] = all_df['ANR type']
  if cogen_tag=='cogen': 
    all_df['Annual Net Revenues (M$/MWe/y)'] = all_df['Net Revenues with H2 PTC with elec ($/year)']/(1e6*all_df['Depl. ANR Cap. (MWe)'])
    all_df['Annual Net Revenues (M$/y)'] = all_df['Net Revenues with H2 PTC with elec ($/year)']/1e6
    #all_df['Annual Net Revenues wo PTC (M$/y)'] = all_df.apply(lambda x: (x['Net Revenues ($/year)']+x['Electricity revenues ($/y)'])/1e6, axis=1)
    # Net revenues includes costs and avoided ng costs
  else: 
    all_df['Annual Net Revenues (M$/MWe/y)'] = all_df['Net Annual Revenues with H2 PTC ($/MWe/y)']/(1e6)
    all_df['Annual Net Revenues (M$/y)'] = all_df['Net Revenues with H2 PTC ($/year)']
    all_df['Annual Net Revenues wo PTC (M$/y)'] = all_df['Net Revenues ($/year)']/1e6 # Net revenues includes costs and avoided ng costs
  all_df.sort_values(by='Breakeven price ($/MMBtu)', inplace=True)
  return all_df 


def load_heat_results(anr_tag, cogen_tag, with_PTC=True):
  """Loads direct process heat results and returns them sorted by breakeven prices"""
  heat_results_path = f'./results/process_heat/best_pathway_{anr_tag}_{cogen_tag}_PTC_{with_PTC}.xlsx'
  heat_df = pd.read_excel(heat_results_path, index_col='FACILITY_ID')
  heat_df['Annual Net Revenues (M$/MWe/y)']  = heat_df['Pathway Net Ann. Rev. (M$/y)']/heat_df['Depl. ANR Cap. (MWe)']
  heat_df['Annual Net Revenues (M$/y)'] = heat_df['Pathway Net Ann. Rev. (M$/y)']
  heat_df.sort_values(by=['Breakeven NG price ($/MMBtu)', 'Annual Net Revenues (M$/MWe/y)'], inplace=True)
  heat_df['Application'] = 'Process Heat'
  return heat_df



def compute_cumulative_avoided_emissions(df, emissions_label='Ann. avoided CO2 emissions (MMT-CO2/year)'):
  """Computes cumulative viable avoided emissions under new column 'Viable avoided emissions (MMt-CO2/y)'
  """
  df['Viable avoided emissions (MMt-CO2/y)'] = df[emissions_label].cumsum()
  return df


def combine_emissions(applications_results):
  app_list = []
  for app, app_results in applications_results.items():
    data = app_results['data']
    emissions_label = app_results['emissions_label']
    price_label = app_results['price_label']
    data['Avoided emissions (MMt-CO2/y)'] = data[emissions_label]
    data['NG price ($/MMBtu)'] = data[price_label]
    app_list.append(data)
  total_df = pd.concat(app_list)
  total_df.sort_values(by=['NG price ($/MMBtu)'], inplace=True)
  total_df = compute_cumulative_avoided_emissions(total_df, emissions_label='Avoided emissions (MMt-CO2/y)')
  return total_df


def plot_cumulative_avoided_emissions(applications_results, anr_tag, cogen_tag, fig=None):
  if fig: ax = fig.add_subplot()
  else: fig, ax = plt.subplots(figsize=(7,3))
  xmax = 50
  # Total avoided emissions on top subplot
  total_df = combine_emissions(applications_results)
  values = list(total_df['Viable avoided emissions (MMt-CO2/y)'])
  values += [values[-1]]
  edges = [-50]+list(total_df['NG price ($/MMBtu)'])+[xmax]
  ax.stairs(values, edges, label='Total', color=color_map['Total'], baseline=None)
  for app, app_results in applications_results.items():
    data = app_results['data']
    values = list(data['Viable avoided emissions (MMt-CO2/y)'])
    values += [values[-1]]
    edges = [-50]+list(data[app_results['price_label']])+[xmax]
    ax.stairs(values, edges, label=app, color=color_map[app], baseline=None)
  ax.set_xlim(-2,xmax)
  ax.xaxis.set_ticks(np.arange(0, xmax, 5))
  ax.yaxis.set_ticks(np.arange(0,250, 25))
  ax.set_xlabel('Breakeven NG price ($/MMBtu)')
  ax.set_ylabel('Viable avoided emissions\n'+r'$(MMt-{CO_2}/y)$')
  ax.grid(True)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
  lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
  fig.legend(lines, labels, loc='upper right', ncol=1)
  if not fig:
    fig.savefig(f'./results/all_applications_viable_avoided_emissions_{anr_tag}_{cogen_tag}.png', bbox_inches='tight')
  




def concat_results(results):
  df_list = []
  for app, app_data in results.items():
    temp_df = app_data['data']
    temp_df['Application'] = app 
    df_list.append(temp_df)
  df = pd.concat(df_list, ignore_index=True)
  return df


def plot_net_annual_revenues_all_app(df, anr_tag, cogen_tag):
  fig, ax = plt.subplots(figsize=(10,4))
  save_path = f'./results/ANR_application_comparison_{anr_tag}_{cogen_tag}.png'
  print(save_path)
  
  sns.stripplot(ax=ax, data=df, x='Annual Net Revenues (M$/MWe/y)', y='Application',\
                  palette=palette, hue='ANR', alpha=0.6)
  sns.boxplot(ax=ax, data=df, x='Annual Net Revenues (M$/MWe/y)', y='Application', color='black',\
                  fill=False, width=0.5)
  sns.despine()
  ax.set_ylabel('')
  ax.set_xlabel('Net Annual Revenues (M$/MWe/y)')
  ax.get_legend().set_visible(False)
  ax.xaxis.set_ticks(np.arange(-1, 1, 0.25))
  ax.xaxis.grid(True)
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels,  bbox_to_anchor=(.5,1.08),loc='upper center', ncol=3)
  fig.tight_layout()
  fig.savefig(save_path, bbox_inches='tight')


def compare_oak_net_annual_revenues(cogen_tag):
  save_path = f'./results/ANR_application_comparison_FOAK_vs_NOAK_{cogen_tag}.png'
  noak_df = pd.read_excel(f'./results/ANR_application_comparison_NOAK_{cogen_tag}.xlsx')
  foak_df = pd.read_excel(f'./results/ANR_application_comparison_FOAK_{cogen_tag}.xlsx')
  noak_df['Stage'] = 'NOAK'
  foak_df['Stage'] = 'FOAK'
  total_df = pd.concat([foak_df, noak_df], ignore_index=True)
  applications = total_df['Application'].unique()
  fig, ax = plt.subplots(len(applications), 1, sharex=True, figsize=(9,6))
  for c, app in enumerate(applications):
    sns.stripplot(ax=ax[c], data = total_df[total_df.Application == app], x='Annual Net Revenues (M$/MWe/y)', y = 'Stage',\
                   palette=palette, hue='ANR', alpha=0.5)
    sns.boxplot(ax=ax[c], data =total_df[total_df.Application == app], x='Annual Net Revenues (M$/MWe/y)', y='Stage', \
                color='black', fill=False, width=0.5)
    sns.despine()
    ax[c].xaxis.grid(True)
    ax[c].set_ylabel(app)
    ax[c].get_legend().set_visible(False)
  handles, labels = ax[1].get_legend_handles_labels()
  fig.legend(handles, labels,  bbox_to_anchor=(.5,1.08),loc='upper center', ncol=3)
  fig.tight_layout()
  fig.savefig(save_path, bbox_inches='tight')



def create_data_dict(oak, cogen_tag):
  h2_df = load_h2_results(oak, cogen_tag)
  h2_df = compute_cumulative_avoided_emissions(h2_df)
  heat_df = load_heat_results(oak, cogen_tag)
  heat_df = compute_cumulative_avoided_emissions(heat_df, emissions_label='Emissions_mmtco2/y')
  results = {'Industrial Hydrogen':{'data':h2_df, 
                                    'emissions_label':'Ann. avoided CO2 emissions (MMT-CO2/year)',
                                    'price_label':'Breakeven price ($/MMBtu)'}
            ,'Process Heat':{'data':heat_df, 
                                      'emissions_label':'Emissions_mmtco2/y',
                                      'price_label':'Breakeven NG price ($/MMBtu)'}}
  return results

def compare_oak_avoided_emissions(cogen_tag):
  foak_results = create_data_dict('FOAK', cogen_tag) 
  noak_results = create_data_dict('NOAK', cogen_tag)
  foak_results = combine_emissions(foak_results)
  noak_results = combine_emissions(noak_results)
  fig, ax = plt.subplots(figsize=(8,5))
  xmax = 70

  values_foak = list(foak_results['Viable avoided emissions (MMt-CO2/y)'])
  values_foak += [values_foak[-1]]
  edges_foak = [0]+list(foak_results['NG price ($/MMBtu)'])+[xmax]
  ax.stairs(values_foak, edges_foak, label='FOAK', color=color_map['FOAK'], baseline=None)
  values_noak = list(noak_results['Viable avoided emissions (MMt-CO2/y)'])
  values_noak += [values_noak[-1]]
  edges_noak = [0]+list(noak_results['NG price ($/MMBtu)'])+[xmax]
  ax.stairs(values_noak, edges_noak, label='NOAK', color=color_map['NOAK'], baseline=None)
  ax.set_xlim(-2,xmax)
  ax.xaxis.set_ticks(np.arange(0, xmax+10, 5))
  ax.yaxis.set_ticks(np.arange(0, 250, 25))
  ax.set_xlabel('Breakeven NG price ($/MMBtu)')  
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.grid(True)
  fig.text(0.04, 0.5, r'Viable avoided emissions $(MMt-{CO_2}/y)$', va='center', rotation='vertical')
  lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
  lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
  fig.legend(lines, labels, loc='upper right', ncol=1)
  fig.savefig(f'./results/all_applications_viable_avoided_emissions_FOAK_vs_NOAK_{cogen_tag}.png')


def compare_cogen_net_annual_revenues(fig, dfs):
  # Load and prep data
  
  combined = pd.concat(dfs, ignore_index=True)
  h2comb = combined[combined.Application == 'Industrial Hydrogen']
  heatcomb = combined[combined.Application == 'Process Heat']

  # Plot left H2, right Heat
  (h2fig, heatfig) = fig.subfigures(1,2)

  h2ax = h2fig.subplots()
  sns.stripplot(ax=h2ax, data=h2comb, x='Annual Net Revenues (M$/MWe/y)', y='Case', palette=palette, hue='ANR', alpha=.6)
  sns.boxplot(ax=h2ax, data=h2comb, x='Annual Net Revenues (M$/MWe/y)', y='Case', color='black', fill=False, width=0.5)
  sns.despine()
  h2ax.xaxis.grid(True)
  h2ax.get_legend().set_visible(False)
  h2ax.set_ylabel('')
  letter_annotation(h2ax, -.25, 1, 'I')

  heatax = heatfig.subplots()
  sns.stripplot(ax=heatax, data=heatcomb, x='Annual Net Revenues (M$/MWe/y)', y = 'Case', palette=palette, hue='ANR', alpha=.6)
  sns.boxplot(ax=heatax, data=heatcomb, x='Annual Net Revenues (M$/MWe/y)', y='Case', color='black', fill=False, width=0.5)
  sns.despine()
  heatax.xaxis.grid(True)
  heatax.get_legend().set_visible(False)
  heatax.set_ylabel('')
  letter_annotation(heatax, -.25, 1, 'II')

  #duplicate legend entries issue
  h3, l3 = h2ax.get_legend_handles_labels()
  h4, l4 = heatax.get_legend_handles_labels()
  by_label = dict(zip(l3+l4, h3+h4))
  fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,1.03),loc='upper center', ncol=5)


def combined_avoided_emissions_oak_cogen():
  save_path = f'./results/all_applications_oak_cogen_emissions.png'
  # Load data for net annual revenues
  noak_co = pd.read_excel(f'./results/ANR_application_comparison_NOAK_cogen.xlsx')
  noak_no = pd.read_excel(f'./results/ANR_application_comparison_NOAK_nocogen.xlsx')
  foak_no = pd.read_excel(f'./results/ANR_application_comparison_FOAK_nocogen.xlsx')
  foak_co = pd.read_excel(f'./results/ANR_application_comparison_FOAK_cogen.xlsx')
  noak_co['Case'] = 'NOAK\nCogeneration'
  noak_no['Case'] = 'NOAK\nNo-Cogeneration'
  foak_co['Case'] = 'FOAK\nCogeneration'
  foak_no['Case'] = 'FOAK\nNo-Cogeneration'
  dfs = [foak_no, foak_co, noak_no, noak_co]

  for key,df in {'FOAK_no':foak_no, 'FOAK_co':foak_co, 'NOAK_no':noak_no, 'NOAK_co':noak_co }.items():
    file = f'./results/all_applications_{key}_rev_stats.xlsx'
    for app in df.Application.unique():
      app_df = df[df.Application == app]
      pp_industrial_hydrogen.print_stats(app, excel_file=file, df=app_df, column_name='Annual Net Revenues (M$/MWe/y)')

  # Load data for cumulative emissions plot
  foak_no_em = combine_emissions(create_data_dict('FOAK', 'nocogen'))
  noak_no_em = combine_emissions(create_data_dict('NOAK', 'nocogen'))
  foak_co_em = combine_emissions(create_data_dict('FOAK', 'cogen'))
  noak_co_em = combine_emissions(create_data_dict('NOAK', 'cogen'))
  emdfs = {'FOAK\nNo-Cogeneration':foak_no_em, 
           'FOAK\nCogeneration':foak_co_em, 
           'NOAK\nNo-Cogeneration':noak_no_em,
            'NOAK\nCogeneration':noak_co_em}

  # figure
  fig = plt.figure(figsize=(10,10))
  (topfig, emfig) = fig.subfigures(2,1, height_ratios=[1.5,1])
  compare_cogen_net_annual_revenues(topfig, dfs=dfs)

  # emissions
  emax = emfig.subplots()
  xmax = 50
  for label, results in emdfs.items():
    values = list(results['Viable avoided emissions (MMt-CO2/y)'])
    values += [values[-1]]
    edges = [-50]+list(results['NG price ($/MMBtu)'])+[xmax]
    emax.stairs(values, edges, label=label, color=color_map[label], baseline=None)
  emax.set_xlim(-2,xmax)
  emax.xaxis.set_ticks(np.arange(0, xmax+10, 5))
  emax.yaxis.set_ticks(np.arange(0, 250, 25))
  emax.set_xlabel('Breakeven NG price ($/MMBtu)')  
  emax.spines['top'].set_visible(False)
  emax.spines['right'].set_visible(False)
  emax.grid(True)
  letter_annotation(emax, -.25, 1, 'III')
  emax.set_ylabel('Viable avoided emissions\n'r'$(MMt-{CO_2}/y)$')
  lines_labels = [emax.get_legend_handles_labels() for ax in emfig.axes]
  lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
  emfig.legend(lines, labels, loc='upper right', ncol=1)
  fig.savefig(save_path, bbox_inches='tight')
  plt.close()




def heat_abatement_plot(fig, df, anr_tag, cogen_tag):
  df['Cost ANR ($/y)'] = (df['CAPEX ($/y)']+df['O&M ($/y)']+df['Conversion']-df['Avoided NG Cost ($/y)'])
  df['Abatement cost ($/tCO2)'] = df['Cost ANR ($/y)']/(df['Emissions_mmtco2/y']*1e6)
  df['Abatement potential (tCO2/y-MWe)'] = 1e6*df['Emissions_mmtco2/y']/df['Depl. ANR Cap. (MWe)']
  for ind in df['Pathway'].unique():
    pp_industrial_hydrogen.print_stats(ind, f'./results/process_heat/heat_abatement_cost_stats_{anr_tag}_cogen_{cogen_tag}.xlsx', \
                    df[df['Pathway']==ind], column_name='Abatement cost ($/tCO2)')
    pp_industrial_hydrogen.print_stats(ind, f'./results/process_heat/heat_abatement_pot_stats_{anr_tag}_cogen_{cogen_tag}.xlsx', \
                    df[df['Pathway']==ind], column_name='Abatement potential (tCO2/y-MWe)')
  ax = fig.subplots(2,1)
  sns.boxplot(ax=ax[0], data=df, y='Pathway', x='Abatement cost ($/tCO2)',color='black',fill=False, width=.5)
  sns.stripplot(ax=ax[0], data=df, y='Pathway', x='Abatement cost ($/tCO2)', hue='ANR', palette = palette,alpha=.6)
  letter_annotation(ax[0], -.25, 1, 'II-a')
  ax[0].set_ylabel('')
  ax[0].set_xlim(-50,5000)
  ax[0].get_legend().set_visible(False)
  sns.boxplot(ax=ax[1], data=df, y='Pathway', x='Abatement potential (tCO2/y-MWe)',color='black', fill=False, width=.5)
  sns.stripplot(ax=ax[1], data=df, y='Pathway',x='Abatement potential (tCO2/y-MWe)',hue='ANR', palette=palette,alpha=.6)
  letter_annotation(ax[1], -.25, 1, 'II-b')
  ax[1].set_ylabel('')
  ax[1].get_legend().set_visible(False)
  ax[1].set_xlim(-10,5010)
  ax[1].xaxis.set_ticks(np.arange(0, 7000, 1000))
  lines, labels = ax[1].get_legend_handles_labels()
  fig.legend(lines, labels, loc='upper right', ncol=1)
  sns.despine()

def combined_avoided_emissions_abatement(applications_results, anr_tag, cogen_tag):
  save_path = f'./results/all_applications_emissions_abatement_{anr_tag}_{cogen_tag}.png'
  fig = plt.figure(figsize=(8, 8))
  (topfig, bottomfig) = fig.subfigures(2, 1, height_ratios=[2,1])
  (h2fig, heatfig) = topfig.subfigures(1,2)
  # Emissions
  plot_cumulative_avoided_emissions(applications_results, anr_tag, cogen_tag, fig = bottomfig)
  # Hydrogen
  import pp_industrial_hydrogen
  h2_data = pp_industrial_hydrogen.load_data(anr_tag)
  pp_industrial_hydrogen.plot_abatement_cost(h2_data, OAK=anr_tag, fig=h2fig)
  # Direct heat
  heat_data = pd.read_excel(f'./results/process_heat/best_pathway_{anr_tag}_{cogen_tag}.xlsx')
  heat_abatement_plot(fig = heatfig, df= heat_data, anr_tag=anr_tag, cogen_tag=cogen_tag)
  fig.savefig(save_path, bbox_inches='tight')
  plt.close()



def combined_heat_ff_plot(anr_tag, cogen_tag):
  """Plot heat results: 
  - TOP: left breakeven prices distribution (boxplot), right net annual revenues
  - BOTTOM: average cashflows
  """
  save_path = f'./results/combined_heat_ff_{anr_tag}_{cogen_tag}.png'
  fig = plt.figure(figsize=(8, 7))
  (topfig, botfig) = fig.subfigures(2,1, height_ratios=[1,1.25])
  (befig, revfig) = topfig.subfigures(1,2)
  
  # Breakeven prices distribution plot
  # Load results
  heat_df = pd.read_excel(f'./results/process_heat/best_pathway_{anr_tag}_{cogen_tag}.xlsx')
  beax = befig.subplots()
  sns.boxplot(ax=beax, data=heat_df, y='Pathway', x='Breakeven NG price ($/MMBtu)', color='black', fill=False, width=.5)
  sns.stripplot(ax=beax, data=heat_df, x='Breakeven NG price ($/MMBtu)', y='Pathway', hue='ANR', palette=palette, alpha=.6)
  beax.get_legend().set_visible(False)
  beax.set_xlim(0,250)
  beax.set_ylabel('')
  beax.xaxis.grid(True)
  sns.despine()
  letter_annotation(beax, -.25, 1, 'I')

  # NEt annual revenues plot
  revax = revfig.subplots()
  sns.boxplot(ax=revax, data=heat_df, y='Pathway', x='Pathway Net Ann. Rev. (M$/y/MWe)', color='black', fill=False, width=.5)
  sns.stripplot(ax=revax, data=heat_df, x='Pathway Net Ann. Rev. (M$/y/MWe)', y='Pathway', hue='ANR', palette=palette, alpha=.6)
  revax.get_legend().set_visible(False)
  revax.set_xlabel('Net Annual Revenue (M$/MWe/y)')
  revax.set_ylabel('')
  revax.xaxis.grid(True)
  sns.despine()
  letter_annotation(revax, -.25, 1, 'II')

  # Average cashflows on bottom figure
  OAK = anr_tag
  cashflows_df_anr_anrh2 = heat_df[heat_df.Pathway=='ANR+ANR-H2']
  cashflows_df_anr_anrh2['ANR CAPEX'] = -cashflows_df_anr_anrh2[f'Annual_CAPEX_{OAK}']/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['ANR for H2 CAPEX'] = -cashflows_df_anr_anrh2['Annual ANR CAPEX']/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['H2 CAPEX'] = -cashflows_df_anr_anrh2['Annual H2 CAPEX']/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['ANR O&M'] = -(cashflows_df_anr_anrh2[f'FOPEX_{OAK}']+cashflows_df_anr_anrh2[f'VOPEX_{OAK}'])/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['ANR for H2 O&M'] = -(cashflows_df_anr_anrh2['ANR VOM']+cashflows_df_anr_anrh2['ANR FOM'])/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['H2 O&M'] = -(cashflows_df_anr_anrh2['H2 VOM']+cashflows_df_anr_anrh2['H2 FOM'])/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['Conversion'] = -(cashflows_df_anr_anrh2['Conversion'])/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['Avoided Fossil Fuel Costs'] = cashflows_df_anr_anrh2['Avoided NG Cost ($/y)']/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anr_anrh2['H2 PTC'] = cashflows_df_anr_anrh2['H2 PTC']/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])
  if cogen_tag == 'cogen': cashflows_df_anr_anrh2['Electricity'] = cashflows_df_anr_anrh2['Electricity revenues ($/y)']/(1e6*cashflows_df_anr_anrh2['Depl. ANR Cap. (MWe)'])

  cashflows_df_anrh2 = heat_df[heat_df.Pathway=='ANR-H2']
  cashflows_df_anrh2['ANR CAPEX'] = 0
  cashflows_df_anrh2['ANR for H2 CAPEX'] = -cashflows_df_anrh2['Annual ANR CAPEX']/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anrh2['H2 CAPEX'] = -cashflows_df_anrh2['Annual H2 CAPEX']/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anrh2['ANR O&M'] = 0
  cashflows_df_anrh2['ANR for H2 O&M'] = -(cashflows_df_anrh2['ANR VOM']+cashflows_df_anrh2['ANR FOM'])/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anrh2['H2 O&M'] = -(cashflows_df_anrh2['H2 VOM']+cashflows_df_anrh2['H2 FOM'])/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anrh2['Conversion'] = -(cashflows_df_anrh2['Conversion'])/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anrh2['Avoided Fossil Fuel Costs'] = cashflows_df_anrh2['Avoided NG Cost ($/y)']/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  cashflows_df_anrh2['H2 PTC'] = cashflows_df_anrh2['H2 PTC']/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  if cogen_tag == 'cogen': cashflows_df_anrh2['Electricity'] = cashflows_df_anrh2['Electricity revenues ($/y)']/(1e6*cashflows_df_anrh2['Depl. ANR Cap. (MWe)'])
  if cogen_tag =='cogen':
    cashflows_df_anr_anrh2 = cashflows_df_anr_anrh2[['ANR', 'ANR CAPEX','ANR for H2 CAPEX','H2 CAPEX','ANR O&M','ANR for H2 O&M', 
                                                  'H2 O&M','Conversion','Avoided Fossil Fuel Costs','H2 PTC', 'Electricity']]
    cashflows_df_anrh2 = cashflows_df_anrh2[['ANR', 'ANR CAPEX','ANR for H2 CAPEX','H2 CAPEX','ANR O&M','ANR for H2 O&M', 
                                                  'H2 O&M','Conversion','Avoided Fossil Fuel Costs','H2 PTC', 'Electricity']]
  else:
    cashflows_df_anr_anrh2 = cashflows_df_anr_anrh2[['ANR', 'ANR CAPEX','ANR for H2 CAPEX','H2 CAPEX','ANR O&M','ANR for H2 O&M', 
                                                  'H2 O&M','Conversion','Avoided Fossil Fuel Costs','H2 PTC']]
    cashflows_df_anrh2 = cashflows_df_anrh2[['ANR', 'ANR CAPEX','ANR for H2 CAPEX','H2 CAPEX','ANR O&M','ANR for H2 O&M', 
                                                    'H2 O&M','Conversion','Avoided Fossil Fuel Costs','H2 PTC']]
  cashflows_df_anr_anrh2 = cashflows_df_anr_anrh2.groupby(['ANR']).mean()
  cashflows_df_anrh2 = cashflows_df_anrh2.groupby(['ANR']).mean()


  ax = botfig.subplots(1,2,sharey=True)
  axup = ax[0]
  cashflows_df_anr_anrh2.plot(ax = axup, kind ='bar', stacked=True, color=cashflows_color_map, width=0.4)
  axup.set_ylabel('Average Normalized\nCashflows (M$/MWe/y)')
  axup.set_xlabel('')
  axup.yaxis.grid(True)
  axup.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
  axup.set_xticks(axup.get_xticks(), axup.get_xticklabels(), rotation=0, ha='center')
  axup.set_ylim(-1.02, 0.52)
  axup.yaxis.set_ticks(np.arange(-1.75, 1, 0.25))
  axup.get_legend().set_visible(False)
  letter_annotation(axup, -.25, 1.04, 'III-a: ANR+ANR-H2')

  lax = ax[1]
  cashflows_df_anrh2.plot(ax = lax, kind ='bar', stacked=True, color=cashflows_color_map, width=0.25)
  lax.set_ylabel('Average Normalized\nCashflows (M$/MWe/y)')
  lax.set_xlabel('')
  lax.yaxis.grid(True)
  lax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
  lax.set_xticks(lax.get_xticks(), lax.get_xticklabels(), rotation=0, ha='center')
  lax.set_ylim(-1.02, 0.52)
  lax.yaxis.set_ticks(np.arange(-1.75, 1, 0.25))
  lax.get_legend().set_visible(False)
  letter_annotation(lax, -.25, 1.04, 'III-b: ANR-H2')


  #Common legend for whole figure
  h3, l3 = revax.get_legend_handles_labels()
  h4, l4 = beax.get_legend_handles_labels()
  
 
  h, l = lax.get_legend_handles_labels()
  by_label = dict(zip(l3+l4+l, h3+h4+h))
  fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,-.01),loc='upper center', ncol=4)


  fig.savefig(save_path, bbox_inches='tight')
  plt.close()


def combined_h2_ff_plot(anr_tag, cogen_tag):
  """Plot heat results: 
  - TOP: left breakeven prices distribution (boxplot), right net annual revenues
  - BOTTOM: average cashflows
  """
  save_path = f'./results/combined_h2_ff_{anr_tag}_{cogen_tag}.png'
  fig = plt.figure(figsize=(8, 7))
  (topfig, botfig) = fig.subfigures(2,1, height_ratios=[1,1.25])
  (befig, revfig) = topfig.subfigures(1,2)

  h2_data = pp_industrial_hydrogen.load_data(OAK=anr_tag)
  # Breakeven prices
  beax = befig.subplots()
  sns.boxplot(ax=beax, data=h2_data, y='Industry', x='Breakeven price ($/MMBtu)', color='black', fill=False, width=.5)
  sns.stripplot(ax=beax, data=h2_data, y='Industry', x='Breakeven price ($/MMBtu)', hue='ANR type', palette=palette, alpha=.6)
  beax.get_legend().set_visible(False)
  #beax.set_xlim(0,250)
  beax.set_ylabel('')
  beax.xaxis.grid(True)
  sns.despine()
  letter_annotation(beax, -.25, 1, 'I')

  # Net annual revenues
  revax = revfig.subplots()
  h2_data = pp_industrial_hydrogen.compute_normalized_net_revenues(h2_data, OAK=anr_tag)
  if cogen_tag=='cogen': x = 'Net Annual Revenues with H2 PTC with elec (M$/MWe/y)'
  elif cogen_tag == 'nocogen': x = 'Net Annual Revenues with H2 PTC (M$/MWe/y)'
  sns.boxplot(ax=revax, data=h2_data, y='Industry', x=x, color='black',fill=False, width=.5)
  sns.stripplot(ax=revax, data=h2_data, y='Industry', x=x, hue='ANR type', palette=palette)
  revax.set_ylabel('')
  revax.set_xlabel('Net Annual Revenues (M$/MWe/y)')
  revax.get_legend().set_visible(False)
  revax.set_xlim(-1.3, 0.8)
  revax.xaxis.set_ticks(np.arange(-1.25, 1, 0.25))
  sns.despine()
  revax.xaxis.grid(True)
  letter_annotation(revax, -.25, 1.04, 'II')

  #Average cashflows
  # Cashflows in M$/MWe/y
  df = h2_data.copy()
  df['ANR CAPEX'] = -df['ANR CAPEX ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 CAPEX'] = -df['H2 CAPEX ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['ANR O&M'] = -df['ANR O&M ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 O&M'] = -df['H2 O&M ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['Conversion'] = -df['Conversion costs ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['Avoided Fossil Fuel Costs'] = df['Avoided NG costs ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 PTC'] = df['H2 PTC Revenues ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  if cogen_tag:
    df['Electricity'] = df['Electricity revenues ($/y)']/(1e6*df['Depl. ANR Cap. (MWe)'])
    design_df = df[['Industry','ANR type','ANR CAPEX', 'H2 CAPEX', 'ANR O&M', 'H2 O&M', 'Conversion', 'Avoided Fossil Fuel Costs', \
                    'H2 PTC', 'Electricity']]
  else:
    design_df = df[['Industry','ANR type','ANR CAPEX', 'H2 CAPEX', 'ANR O&M', 'H2 O&M', 'Conversion', 'Avoided Fossil Fuel Costs', 'H2 PTC']]
  am_df = design_df[design_df.Industry == 'Ammonia']
  am_df = am_df.drop(columns=['Industry'])
  am_df = am_df.groupby('ANR type').mean()
  ref_df = design_df[design_df.Industry == 'Refining']
  ref_df = ref_df.drop(columns=['Industry'])
  ref_df = ref_df.groupby(['ANR type']).mean()
  st_df = design_df[design_df.Industry == 'Steel']
  st_df = st_df.drop(columns=['Industry'])
  st_df = st_df.groupby(['ANR type']).mean()
  cax = botfig.subplots(1,3, sharey=True)
  am_df.plot(ax=cax[0], kind='bar', stacked=True, color=cashflows_color_map)
  ref_df.plot(ax=cax[1], kind='bar', stacked=True, color=cashflows_color_map)
  st_df.plot(ax=cax[2], kind='bar', stacked=True, color=cashflows_color_map)
  cax[0].set_ylabel('Average Normalized Cashflows (M$/MWe/y)')
  cax[0].set_xlabel('')
  cax[0].set_ylim(-2.75, 1.8)
  cax[0].yaxis.set_ticks(np.arange(-2.75, 2, 0.25))
  cax[0].set_xticks(cax[0].get_xticks(), cax[0].get_xticklabels(), rotation=0, ha='center')
  cax[0].get_legend().set_visible(False)
  cax[0].yaxis.grid(True)
  letter_annotation(cax[0], -.2, 1.04, 'III-a: Ammonia')
  cax[1].set_xlabel('')
  cax[1].set_ylim(-2.75, 1.8)
  cax[1].yaxis.set_ticks(np.arange(-2.75, 2, 0.25))
  cax[1].set_xticks(cax[1].get_xticks(), cax[1].get_xticklabels(), rotation=0, ha='center')
  cax[1].get_legend().set_visible(False)
  cax[1].yaxis.grid(True)
  letter_annotation(cax[1], -.1, 1.04, 'b: Refining')
  cax[2].set_xlabel('')
  cax[2].set_ylim(-2.75, 1.8)
  cax[2].yaxis.set_ticks(np.arange(-2.75, 2, 0.25))
  cax[2].set_xticks(cax[2].get_xticks(), cax[2].get_xticklabels(), rotation=0, ha='center')
  cax[2].get_legend().set_visible(False)
  cax[2].yaxis.grid(True)
  letter_annotation(cax[2], -.1, 1.04, 'c: Steel')
  
  #Common legend for whole figure
  h3, l3 = revax.get_legend_handles_labels()
  h4, l4 = beax.get_legend_handles_labels()
  
 
  h00, l00 = cax[0].get_legend_handles_labels()
  h01, l01 = cax[1].get_legend_handles_labels()
  h02, l02 = cax[2].get_legend_handles_labels()
  by_label = dict(zip(l3+l4+l00+l01+l02, h3+h4+h00+h01+h02))
  fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,-.01),loc='upper center', ncol=4)

  fig.savefig(save_path, bbox_inches='tight')



def run_case(oak, cogen):
  if oak: anr_tag = 'NOAK'
  else: anr_tag = 'FOAK'
  if cogen: cogen_tag = 'cogen'
  else: cogen_tag = 'nocogen'
  h2_df = load_h2_results(anr_tag, cogen)
  h2_df = compute_cumulative_avoided_emissions(h2_df)
  heat_df = load_heat_results(anr_tag, cogen_tag)
  heat_df = compute_cumulative_avoided_emissions(heat_df, emissions_label='Emissions_mmtco2/y')
  applications_results = {'Industrial Hydrogen':{'data':h2_df, 
                                                 'emissions_label':'Ann. avoided CO2 emissions (MMT-CO2/year)',
                                                 'price_label':'Breakeven price ($/MMBtu)'}
                          ,'Process Heat':{'data':heat_df, 
                                                   'emissions_label':'Emissions_mmtco2/y',
                                                   'price_label':'Breakeven NG price ($/MMBtu)'}}
  plot_cumulative_avoided_emissions(applications_results, anr_tag, cogen_tag)
  combined_avoided_emissions_abatement(applications_results, anr_tag, cogen_tag)
  elec_df = load_elec_results(anr_tag)
  applications_results['Electricity'] = {'data':elec_df, 
                                         'emissions_label':None, 
                                         'price_label':None}
  results = concat_results(applications_results)
  results_stats = results[['Application', 'Annual Net Revenues (M$/MWe/y)']].describe([.1,.25, .5, .75,.9])
  excel_file = f'./results/ANR_application_comparison_{anr_tag}_{cogen_tag}.xlsx'
  try:
    with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
      with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results.to_excel(writer, sheet_name='data', index=False)
        results_stats.to_excel(writer, sheet_name='stats')
  except FileNotFoundError:
    # If the file doesn't exist, create a new one and write the DataFrame to it
    results.to_excel(excel_file, sheet_name='data', index=False)
    with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
      with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_stats.to_excel(writer, sheet_name='stats')
  plot_net_annual_revenues_all_app(results,anr_tag, cogen_tag)
  combined_heat_ff_plot(cogen_tag=cogen_tag, anr_tag=anr_tag)
  combined_h2_ff_plot(cogen_tag=cogen_tag, anr_tag=anr_tag)
  

def main():
  warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
  for cogen in [True, False]:
    
    if cogen: cogen_tag = 'cogen'
    else: cogen_tag = 'nocogen'

    for noak in [True, False]:
      run_case(noak, cogen)

    compare_oak_net_annual_revenues(cogen_tag)
    compare_oak_avoided_emissions(cogen_tag)
    
  combined_avoided_emissions_oak_cogen()


if __name__ == '__main__':
  main()
  

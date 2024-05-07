import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import palette


color_map = {'Industrial Hydrogen':'blue', 'Direct Process Heat':'red', 'Total':'Green', 'FOAK':'limegreen', 
             'NOAK':'forestgreen'}

def load_elec_results(anr_tag):
  elec_results_path = f'./results/price_taker_{anr_tag}_MidCase.xlsx'
  elec_df = pd.read_excel(elec_results_path, index_col=0)
  elec_df['Annual Net Revenues (M$/MWe/y)'] = elec_df['Annual Net Revenues ($/year/MWe)']/1e6
  # Only keep the best design for each state and year
  df = elec_df.loc[elec_df.groupby('state')['Annual Net Revenues (M$/MWe/y)'].transform(max) == elec_df['Annual Net Revenues (M$/MWe/y)']]
  df['Application'] = 'Electricity'
  df.rename(columns={'ANR type':'ANR'}, inplace=True)
  return df
  

def load_h2_results(anr_tag, cogen):
  """"Loads all hydrogen results and returns results sorted by breakeven prices"""
  h2_results_path = f'./results/clean_results_anr_{anr_tag}_h2_wacc_0.077.xlsx'
  industries = ['refining', 'steel', 'ammonia']
  list_df = []
  for ind in industries:
    df = pd.read_excel(h2_results_path, sheet_name=ind, index_col='id')
    df = df[['state', 'H2 Dem. (kg/day)', 'Net Annual Revenues with H2 PTC ($/MWe/y)', 'HTSE', 'Depl. ANR Cap. (MWe)', 'ANR type', \
             '# ANR modules', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 'Electricity revenues ($/y)', \
             'Net Revenues with H2 PTC with elec ($/year)']]
    df['Industry'] = ind 
    list_df.append(df)
  all_df = pd.concat(list_df)
  all_df['Application'] = 'Industrial Hydrogen'
  all_df['ANR'] = all_df['ANR type']
  if cogen: all_df['Annual Net Revenues (M$/MWe/y)'] = all_df['Net Revenues with H2 PTC with elec ($/year)']/(1e6*all_df['Depl. ANR Cap. (MWe)'])
  else: all_df['Annual Net Revenues (M$/MWe/y)'] = all_df['Net Annual Revenues with H2 PTC ($/MWe/y)']/(1e6)
  all_df.sort_values(by='Breakeven price ($/MMBtu)', inplace=True)
  return all_df 


def load_heat_results(anr_tag, cogen_tag):
  """Loads direct process heat results and returns them sorted by breakeven prices"""
  heat_results_path = f'./results/process_heat/best_pathway_{anr_tag}_{cogen_tag}.xlsx'
  heat_df = pd.read_excel(heat_results_path, index_col='FACILITY_ID')
  heat_df['Annual Net Revenues (M$/MWe/y)']  = heat_df['Pathway Net Ann. Rev. (M$/y)']/heat_df['Depl. ANR Cap. (MWe)']
  heat_df.sort_values(by=['Breakeven NG price ($/MMBtu)', 'Annual Net Revenues (M$/MWe/y)'], inplace=True)
  heat_df['Application'] = 'Direct Process Heat'
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


def plot_cumulative_avoided_emissions(applications_results, anr_tag, cogen_tag):
  fig, ax = plt.subplots(2, 1, figsize=(8,5), sharex=True)
  xmax = 50
  # Total avoided emissions on top subplot
  total_df = combine_emissions(applications_results)
  values = list(total_df['Viable avoided emissions (MMt-CO2/y)'])
  values += [values[-1]]
  edges = [0]+list(total_df['NG price ($/MMBtu)'])+[xmax]
  ax[0].stairs(values, edges, label='Total', color=color_map['Total'], baseline=None)
  ax[0].set_xlim(-2,xmax)
  ax[0].xaxis.set_ticks(np.arange(0, xmax+10, 5))
  ax[0].yaxis.set_ticks(np.arange(0, 250, 25))
  ax[0].set_xlabel('')
  ax[0].set_ylabel('')
  ax[0].grid(True)
  ax[0].spines['top'].set_visible(False)
  ax[0].spines['right'].set_visible(False)
  for app, app_results in applications_results.items():
    data = app_results['data']
    values = list(data['Viable avoided emissions (MMt-CO2/y)'])
    values += [values[-1]]
    edges = [0]+list(data[app_results['price_label']])+[xmax]
    ax[1].stairs(values, edges, label=app, color=color_map[app], baseline=None)
  ax[1].set_xlim(-2,xmax)
  ax[1].xaxis.set_ticks(np.arange(0, xmax+10, 5))
  ax[1].yaxis.set_ticks(np.arange(0, 150, 25))
  ax[1].set_xlabel('Breakeven NG price ($/MMBtu)')
  ax[1].set_ylabel('')
  ax[1].grid(True)
  ax[1].spines['top'].set_visible(False)
  ax[1].spines['right'].set_visible(False)
  
  fig.text(0.04, 0.5, r'Viable avoided emissions $(MMt-{CO_2}/y)$', va='center', rotation='vertical')
  lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
  lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
  fig.legend(lines, labels, loc='upper right', ncol=1)
  fig.savefig(f'./results/all_applications_viable_avoided_emissions_{anr_tag}_{cogen_tag}.png')




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

def compare_oak_avoided_emissions(cogen_tag):
  oak = 'FOAK'
  h2_df = load_h2_results(oak, cogen=False)
  h2_df = compute_cumulative_avoided_emissions(h2_df)
  heat_df = load_heat_results(oak, cogen_tag)
  heat_df = compute_cumulative_avoided_emissions(heat_df, emissions_label='Emissions_mmtco2/y')
  foak_results = {'Industrial Hydrogen':{'data':h2_df, 
                                                 'emissions_label':'Ann. avoided CO2 emissions (MMT-CO2/year)',
                                                 'price_label':'Breakeven price ($/MMBtu)'}
                          ,'Direct Process Heat':{'data':heat_df, 
                                                   'emissions_label':'Emissions_mmtco2/y',
                                                   'price_label':'Breakeven NG price ($/MMBtu)'}}
  oak = 'NOAK'
  h2_df = load_h2_results(oak, cogen=False)
  h2_df = compute_cumulative_avoided_emissions(h2_df)
  heat_df = load_heat_results(oak, cogen_tag)
  heat_df = compute_cumulative_avoided_emissions(heat_df, emissions_label='Emissions_mmtco2/y')
  noak_results = {'Industrial Hydrogen':{'data':h2_df, 
                                                 'emissions_label':'Ann. avoided CO2 emissions (MMT-CO2/year)',
                                                 'price_label':'Breakeven price ($/MMBtu)'}
                          ,'Direct Process Heat':{'data':heat_df, 
                                                   'emissions_label':'Emissions_mmtco2/y',
                                                   'price_label':'Breakeven NG price ($/MMBtu)'}}
  foak_results = combine_emissions(foak_results)
  noak_results = combine_emissions(noak_results)
  fig, ax = plt.subplots(figsize=(8,5))
  xmax = 70
  # Total avoided emissions on top subplot
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


def compare_cogen_net_annual_revenues():
  save_path = f'./results/ANR_application_comparison_cogen_vs_nocogen.png'
  # Load and prep data
  noak_co = pd.read_excel(f'./results/ANR_application_comparison_NOAK_cogen.xlsx')
  noak_no = pd.read_excel(f'./results/ANR_application_comparison_NOAK_nocogen.xlsx')
  foak_no = pd.read_excel(f'./results/ANR_application_comparison_FOAK_nocogen.xlsx')
  foak_co = pd.read_excel(f'./results/ANR_application_comparison_FOAK_cogen.xlsx')
  noak_co['Co'] = 'Cogeneration'
  noak_no['Co'] = 'No Cogeneration'
  foak_co['Co'] = 'Cogeneration'
  foak_no['Co'] = 'No Cogeneration'
  foak = pd.concat([foak_co, foak_no], ignore_index=True)
  noak = pd.concat([noak_co, noak_no], ignore_index=True)
  applications = ['Industrial Hydrogen', 'Direct Process Heat']

  fig = plt.figure(figsize=(8,8))
  (topfig, botfig) = fig.subfigures(2,1)
  topfig.suptitle('FOAK')
  botfig.suptitle('NOAK')

  # FOAK on top
  topax = topfig.subplots(2,1,sharex=True)
  for c, app in enumerate(applications):
    sns.stripplot(ax=topax[c], data = foak[foak.Application == app], x='Annual Net Revenues (M$/MWe/y)', y = 'Co',\
                   palette=palette, hue='ANR', alpha=0.5)
    sns.boxplot(ax=topax[c], data =foak[foak.Application == app], x='Annual Net Revenues (M$/MWe/y)', y='Co', \
                color='black', fill=False, width=0.5)
    sns.despine()
    topax[c].xaxis.grid(True)
    topax[c].set_ylabel(app)
    topax[c].set_xlim(-1.1, 1)
    topax[c].get_legend().set_visible(False)
  h1, l1 = topax[1].get_legend_handles_labels()
  h2, l2 = topax[0].get_legend_handles_labels()
  #topfig.legend(h1+h1, l1+l2,  bbox_to_anchor=(.5,1.08),loc='upper center', ncol=3)

  #NOAK bottom
  botax = botfig.subplots(2,1,sharex=True)
  for c, app in enumerate(applications):
    sns.stripplot(ax=botax[c], data = noak[noak.Application == app], x='Annual Net Revenues (M$/MWe/y)', y = 'Co',\
                   palette=palette, hue='ANR', alpha=0.5)
    sns.boxplot(ax=botax[c], data =noak[noak.Application == app], x='Annual Net Revenues (M$/MWe/y)', y='Co', \
                color='black', fill=False, width=0.5)
    sns.despine()
    botax[c].xaxis.grid(True)
    botax[c].set_ylabel(app)
    botax[c].set_xlim(-1.1, 1)
    botax[c].get_legend().set_visible(False)
  h3, l3 = botax[1].get_legend_handles_labels()
  h4, l4 = botax[0].get_legend_handles_labels()
  by_label = dict(zip(l1+l2+l3+l4, h1+h2+h3+h4))
  botfig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,-.08),loc='upper center', ncol=3)
  fig.savefig(save_path, bbox_inches = 'tight')



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
                          ,'Direct Process Heat':{'data':heat_df, 
                                                   'emissions_label':'Emissions_mmtco2/y',
                                                   'price_label':'Breakeven NG price ($/MMBtu)'}}
  plot_cumulative_avoided_emissions(applications_results, anr_tag, cogen_tag)
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

def main():
  for cogen in [True, False]:
    
    if cogen: cogen_tag = 'cogen'
    else: cogen_tag = 'nocogen'

    for noak in [True, False]:
      run_case(noak, cogen)

    compare_oak_net_annual_revenues(cogen_tag)
    compare_oak_avoided_emissions(cogen_tag)
  compare_cogen_net_annual_revenues()

if __name__ == '__main__':
  main()
  

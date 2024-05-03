import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

anr_tag = 'FOAK'
cogen_tag = 'nocogen'

elec_results_path = f'./results/price_taker_{anr_tag}_MidCase.xlsx'

h2_results_path = f'./results/clean_results_anr_{anr_tag}_h2_wacc_0.077.xlsx'

heat_results_path = f'./results/direct_heat_maxv_results_{anr_tag}_{cogen_tag}.csv'

color_map = {'Industrial Hydrogen':'blue', 'Direct Process Heat':'red', 'Total':'Green'}

def load_elec_results(elec_results_path):
  elec_df = pd.read_excel(elec_results_path, index_col=0)
  elec_df['Net Annual Revenues (M$/MWe/y)'] = elec_df['Annual Net Revenues ($/year/MWe)']/1e6
  # Only keep the best design for each state and year
  df = elec_df.loc[elec_df.groupby('state')['Net Annual Revenues (M$/MWe/y)'].transform(max) == elec_df['Net Annual Revenues (M$/MWe/y)']]
  return df
  

def load_h2_results(h2_results_path):
  """"Loads all hydrogen results and returns results sorted by breakeven prices"""
  industries = ['process_heat', 'refining', 'steel', 'ammonia']
  list_df = []
  for ind in industries:
    df = pd.read_excel(h2_results_path, sheet_name=ind, index_col='id')
    df = df[['state', 'H2 Dem. (kg/day)', 'Net Revenues ($/year)', 'HTSE', 'Depl. ANR Cap. (MWe)', 'ANR type', \
             '# ANR modules', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)' ]]
    df['Industry'] = ind 
    list_df.append(df)
  all_df = pd.concat(list_df)
  all_df['Application'] = 'Industrial Hydrogen'
  all_df.sort_values(by='Breakeven price ($/MMBtu)', inplace=True)
  return all_df 


def load_heat_results(heat_results_path):
  heat_df = pd.read_csv(heat_results_path, index_col='id')
  # Sort by NG benchmark price and revenues = avoided NG cost -> Increaw
  heat_df[f'Net Annual Revenues {anr_tag} (M$/MWt/y)'] # TODO
  heat_df.sort_values(by=['NG price ($/MMBtu)', f'Net Annual Revenues {anr_tag} ($/y)'], inplace=True)
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


def plot_cumulative_avoided_emissions(applications_results):
  fig, ax = plt.subplots(2, 1, figsize=(6,6), sharex=True)
  xmax = 50
  # Total avoided emissions on top subplot
  total_df = combine_emissions(applications_results)
  values = list(total_df['Viable avoided emissions (MMt-CO2/y)'])
  values += [values[-1]]
  edges = [0]+list(total_df['NG price ($/MMBtu)'])+[xmax]
  ax[0].stairs(values, edges, label='Total', color=color_map['Total'], baseline=None)
  ax[0].set_xlim(-2,xmax)
  ax[0].xaxis.set_ticks(np.arange(0, xmax+10, 10))
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
  ax[1].xaxis.set_ticks(np.arange(0, xmax+10, 10))
  ax[1].set_xlabel('NG price ($/MMBtu)')
  ax[1].set_ylabel('')
  ax[1].grid(True)
  ax[1].spines['top'].set_visible(False)
  ax[1].spines['right'].set_visible(False)
  
  fig.text(0.04, 0.5, r'Viable avoided emissions $(MMt-{CO_2}/y)$', va='center', rotation='vertical')
  lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
  lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
  fig.legend(lines, labels, loc='upper right', ncol=1)
  fig.savefig('./results/viable_avoided_emissions.png')



def plot_net_annual_revenues_all_app(applications_results):
  fig, ax = plt.subplots(len(applications_results),1, sharex=True, figsize=(10,8))
  count = 0
  palette={'HTGR':'blue', 'iMSR':'orange', 'iPWR':'green', 'PBR-HTGR':'darkorchid', 'Microreactor':'darkgrey'}
  for app, app_data in applications_results.items():
    sns.stripplot(ax=ax[count], data=app_data['data'], x='Annual Net Revenues (M$/MWe/y)', y=app,\
                  palette=palette, hue='Reactor', marker='*', size=7, alpha=0.8)
    if count !=0:
      ax[count].get_legend().set_visible(False)
    ax[count].set_ylabel('')
    ax[count].grid(True)
    count +=1
  fig.tight_layout()
  plt.show()
  #fig.savefig('./results/ANR_application_comparison.png')

def main():
  h2_df = load_h2_results(h2_results_path=h2_results_path)
  h2_df = compute_cumulative_avoided_emissions(h2_df)
  heat_df = load_heat_results(heat_results_path=heat_results_path)
  heat_df = compute_cumulative_avoided_emissions(heat_df, emissions_label='Emissions')
  applications_results = {'Industrial Hydrogen':{'data':h2_df, 
                                                 'emissions_label':'Ann. avoided CO2 emissions (MMT-CO2/year)',
                                                 'price_label':'Breakeven price ($/MMBtu)'}
                          , 'Direct Process Heat':{'data':heat_df, 
                                                   'emissions_label':'Emissions',
                                                   'price_label':'NG price ($/MMBtu)'}}
  plot_cumulative_avoided_emissions(applications_results)
  elec_df = load_elec_results(elec_results_path)
  applications_results['Electricity'] = {'data':elec_df, 
                                         'emissions_label':None, 
                                         'price_label':None}
  plot_net_annual_revenues_all_app(applications_results)

if __name__ == '__main__':
  main()
  

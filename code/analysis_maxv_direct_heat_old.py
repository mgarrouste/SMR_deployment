import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os, glob
import utils

lifetime = 34 # years
foak = False

cambium_scenario = 'MidCase'
year = 2024

def get_data():
  """Loads Max Vanatta data for direct process heat generation from ANRS
  Goes through each sheet, corresponding to each benchmark price
  Concatenates all the data into one dataframe with existing columns and new 'NG price ($/MMBtu)' column
  """
  heat_dfs = []
  benchmark_prices = [6,8,10,12,14]
  for price in benchmark_prices:
    price_df = pd.read_excel('./input_data/maxv_heat_data.xlsx', sheet_name=str(price))
    price_df['NG price ($/MMBtu)'] = price
    heat_dfs.append(price_df)
  heat_df = pd.concat(heat_dfs)
  # Combine industries
  heat_df.replace({'Paperboard Mills':'Paper Mills', 'Paper Mills, except Newsprint': 'Paper Mills', 'Pulp Mills': 'Paper Mills',
                    'Petrochemicals': 'Petro- and organic chemicals', 'Other Basic Organic Chemicals': 'Petro- and organic chemicals'}, inplace=True)
  return heat_df


def compute_net_annual_revenues(heat_df, anr_tag):
  """
  Computes the net annual revenues for a given stage of deployment from O&M and capital costs in data
  Sorts entries by NG price and id
  Keeps only first occurence of facility with respect to increasing NG price so that avoided emissions are not accounted 
  for twice
  """
  heat_df[f'ANR Cost {anr_tag} ($/y)'] = heat_df['FOPEX']+heat_df['VOPEX']+heat_df[f'ACC_Cap_{anr_tag}']
  heat_df[f'ANR Cost {anr_tag} ($/MWt/y)'] = heat_df[f'ANR Cost {anr_tag} ($/y)']/heat_df['Installed_Cap']
  heat_df[f'Net Annual Revenues {anr_tag} ($/y)'] = heat_df['Fac_Ann_Rev']-heat_df[f'ANR Cost {anr_tag} ($/y)']
  # Keep only positive net annual revenues
  heat_df = heat_df[heat_df[f'Net Annual Revenues {anr_tag} ($/y)'] >0]
  # Remove facilities that appear for lower benchmark prices
  heat_df.sort_values(by=['NG price ($/MMBtu)', 'FACILITY_ID'], inplace=True)
  heat_df.drop_duplicates(subset=['FACILITY_ID'], keep='first', inplace=True, ignore_index=True)
  return heat_df




def compute_average_electricity_prices(cambium_scenario, year):
  folder = f'./input_data/cambium_{cambium_scenario.lower()}_state_hourly_electricity_prices'
  list_csv_files = glob.glob(folder+'/Cambium*.csv')
  state_prices = pd.DataFrame(columns=['average price ($/MWhe)', 'state'])
  state_prices.set_index('state', inplace=True)
  for file in list_csv_files:
    if str(year) in file:
      print(file)
      state = file.split('_')[-2]
      avg_price = pd.read_csv(file, skiprows=5)['energy_cost_enduse'].mean()
      print(avg_price)
      print(type(avg_price))
      state_prices.loc[state, 'average price ($/MWhe)'] = avg_price
  print(state_prices)
  state_prices.to_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx')


def compute_cogen(df):
  df['Remaining capacity (MWe)'] = df.apply(lambda x: x['Installed_Cap'] - x['Thermal MWh/hr'] \
                                            if x['Installed_Cap']-x['Thermal MWh/hr']>0 else 0, axis=1)
  try:
    elec_prices_df = pd.read_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx', index_col=0)
  except FileNotFoundError:
    compute_average_electricity_prices(cambium_scenario, year)
    elec_prices_df = pd.read_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx', index_col=0)
  df['Cogen revenues ($/y)'] = df.apply(lambda x: x['Remaining capacity (MWe)']*elec_prices_df.loc[x['STATE']]*8760, axis=1)
  return df
  


def save_data(df, anr_tag='', cogen_tag=''):
  df.rename(columns={'FACILITY_ID':'id', 'STATE':'state'}, inplace=True)
  df.to_csv(f'./results/direct_heat_maxv_results_{anr_tag}_{cogen_tag}.csv', index=False)


def plot_net_annual_revenues(df, anr_tag='FOAK', cogen_tag='nocogen'):
  save_path = f'./results/direct_heat_maxv_anr_net_ann_revenues_{anr_tag}_{cogen_tag}.png'
  print(f'Plot net annual revenues: {save_path}')
  fig, ax = plt.subplots(figsize=(8,6))
  df[f'Net Annual Revenues {anr_tag} (M$/MWt/y)'] = df[f'Net Annual Revenues {anr_tag} ($/y)']/(df['Installed_Cap']*1e6)
  sns.boxplot(ax=ax, data=df, y='Industry', x=f'Net Annual Revenues {anr_tag} (M$/MWt/y)', hue='NG price ($/MMBtu)', fill=False, width=.5)
  ax.set_ylabel('')
  sns.despine()
  ax.grid(True)
  h, l = ax.get_legend_handles_labels()
  ax.get_legend().set_visible(False)
  fig.legend(h, l, loc='outside right upper', title='NG price ($/MMBtu)')
  fig.tight_layout()
  fig.savefig(save_path)


def compute_net_annual_costs_ng_ccus(df):
  df['NG CCUS Cost ($/MWt/y)'] = (df['Fac_Ann_Rev'] +  df['Emissions']*1e6*utils.ccus_cost)/df['Thermal MWh/hr']
  return df


def plot_anr_vs_ng_ccus_costs(df, anr_tag='FOAK', cogen=False):
  save_path = f'./results/direct_heat_maxv_cost_comparison_anr_ng_ccus_{anr_tag}_{cogen}.png'
  print(f'Plot ANR v.s. NG with CCUS costs :{save_path}')
  # NG with CCUS cost on the x axis
  x = 'NG CCUS Cost (M$/MWt/y)'
  df[x] = df['NG CCUS Cost ($/MWt/y)']/1e6
  # ANR Cost on the y axis
  y = f'ANR Cost {anr_tag} (M$/MWt/y)'
  df[y] = df[f'ANR Cost {anr_tag} ($/MWt/y)']/1e6

  df.rename(columns={'Generator':'ANR design'}, inplace=True)
  fig, ax = plt.subplots(figsize=(8,6))
  med_x = np.arange(0,10, 0.1)
  # Style ANR design
  sns.scatterplot(ax=ax, data=df, y=y, x=x, hue= 'NG price ($/MMBtu)', style='ANR design', palette='flare')
  ax.plot(med_x, med_x, color='grey', linestyle='--', linewidth=0.8)
  ax.spines[['right', 'top']].set_visible(False)
  ax.set_ylim(-.1,2)
  ax.set_xlim(-0.1,7.1)
  ax.set_xlabel(f'{x}\nCCUS cost: {utils.ccus_cost} $/t-CO2')

  fig.tight_layout()
  fig.savefig(save_path)


def plot_anr_vs_ng_costs(df, anr_tag='FOAK', cogen=False):
  save_path = f'./results/direct_heat_maxv_cost_comparison_anr_ng_wo_ccus_{anr_tag}_{cogen}.png'
  print(f'Plot ANR v.s. NG without CCUS costs : {save_path}')
  # NG with CCUS cost on the x axis
  x = 'NG wo/ CCUS Cost (M$/MWt/y)'
  df[x] = df['Fac_Ann_Rev']/(df['Thermal MWh/hr']*1e6)
  # ANR Cost on the y axis
  y = f'ANR Cost {anr_tag} (M$/MWt/y)'
  df[y] = df[f'ANR Cost {anr_tag} ($/MWt/y)']/1e6

  df.rename(columns={'Generator':'ANR design'}, inplace=True)
  fig, ax = plt.subplots(figsize=(8,6))
  med_x = np.arange(0,7, 0.1)
  # Style ANR design
  sns.scatterplot(ax=ax, data=df, y=y, x=x, hue= 'NG price ($/MMBtu)', style='ANR design', palette='crest')
  ax.plot(med_x, med_x, color='grey', linestyle='--', linewidth=0.8)
  ax.spines[['right', 'top']].set_visible(False)
  ax.set_ylim(-.1,2)
  ax.set_xlim(-0.1,7.1)
  ax.set_xlabel(f'{x}')

  fig.tight_layout()
  fig.savefig(save_path)



def plot_deployment_comparison():
  noak_results = './results/direct_heat_maxv_results_NOAK_nocogen.csv'
  foak_results = './results/direct_heat_maxv_results_FOAK_nocogen.csv'
  assert os.path.isfile(noak_results), f'NOAK results not found: {noak_results}'
  assert os.path.isfile(foak_results), f'FOAK results not found: {foak_results}'
  noak_df = pd.read_csv(noak_results)
  foak_df = pd.read_csv(foak_results)
  noak_len, foak_len = len(noak_df), len(foak_df)
  total_df = pd.merge(left=noak_df, right=foak_df, on='id', how='left')
  # Check merging
  assert noak_len == len(total_df)
  # Differential net annual revenues in M$/MWt/y
  total_df['Net Annual Revenues FOAK ($/y)'] = total_df['Net Annual Revenues FOAK ($/y)'].fillna(0)
  total_df = total_df[~(total_df['Industry_x'] == 'Other_Not Found')]
  total_df['Delta Net Rev'] = (total_df['Net Annual Revenues NOAK ($/y)'] - total_df['Net Annual Revenues FOAK ($/y)'])/(total_df['Installed_Cap_x']*1e6)
  # Plot
  fig, ax = plt.subplots(figsize=(10,8), sharex=True)
  sns.boxplot(ax=ax, data=total_df, y='Industry_x', x='Delta Net Rev', hue='NG price ($/MMBtu)_x', fill=False, width=.5, palette='flare')
  ax.set_ylabel('')
  ax.set_xlim(-0.02, 0.085)
  ax.set_xlabel('Differential Net Annual Revenues (M$/MWt/y)')
  ax.axvline(x=0, color='grey', linestyle='--', linewidth=1)
  sns.despine()
  ax.grid(True)
  h, l = ax.get_legend_handles_labels()
  ax.get_legend().set_visible(False)
  fig.legend(h, l, loc='outside right upper', title='NG price ($/MMBtu)')
  fig.tight_layout()
  fig.savefig('./results/direct_heat_maxv_comparison_FOAK_NOAK_anr_net_ann_revenues.png')





def main():
  if foak: anr_tag = 'FOAK'
  else: anr_tag = 'NOAK'
  print(anr_tag)
  heat_df = get_data()
  heat_df = compute_net_annual_revenues(heat_df, anr_tag=anr_tag)
  heat_df = compute_net_annual_costs_ng_ccus(heat_df)
  #heat_df = compute_cogen(heat_df)
  
  save_data(heat_df, anr_tag=anr_tag, cogen_tag='nocogen')
 
  if anr_tag == 'NOAK':
    plot_deployment_comparison()
  else:
    plot_net_annual_revenues(heat_df, anr_tag=anr_tag, cogen_tag='nocogen')
    plot_anr_vs_ng_costs(heat_df, anr_tag=anr_tag)
    plot_anr_vs_ng_ccus_costs(heat_df, anr_tag=anr_tag)


if __name__ == '__main__':
  main()
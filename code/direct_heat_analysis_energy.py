import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import utils

lifetime = 34 # years
foak = True

def get_data(foak=True):
  heat_df = pd.read_csv('./input_data/max_vanatta_direct_heat_anrs/maxvanatta_energy_paper_data.csv')
  heat_df['NG price ($/MMBtu)'] = heat_df.apply(lambda x:float(x['Scenario'].split('_')[0]), axis=1)
  heat_df.drop(columns=['Scenario'], inplace=True)
  # Combine industries
  heat_df.replace({'Paperboard Mills':'Paper Mills', 'Paper Mills, except Newsprint': 'Paper Mills', 'Pulp Mills': 'Paper Mills',
                    'Petrochemicals': 'Petro- and organic chemicals', 'Other Basic Organic Chemicals': 'Petro- and organic chemicals'}, inplace=True)

  if foak:
    heat_df['Net Annual Revenues (M$/y)'] = heat_df['NPV_FOAK']/(1e6*34)
  else:
    heat_df['Net Annual Revenues (M$/y)'] = heat_df['NPV_NOAK']/(1e6*34)
  heat_df.sort_values(by='Net Annual Revenues (M$/y)', inplace=True)
  return heat_df


def compute_cogen(df):
  df['Remaining capacity (MWe)'] = df['NP_Capacity']
  pass


def save_data(df, cost_tag='', cogen_tag=''):
  df.to_csv(f'./results/direct_heat_maxv_results_energy_{cost_tag}_{cogen_tag}.csv', index=False)


def plot_net_annual_revenues(df, cost_tag='foak', cogen_tag='nocogen'):
  plt.figure(figsize=(7,5))
  ax = sns.boxplot(data=df, y='Industry', x='Net Annual Revenues (M$/y)', fill=False, width=.5)
  ax.set_ylabel('')
  sns.despine()
  ax.grid(True)
  plt.tight_layout()
  plt.savefig(f'./results/direct_heat_maxv_revenues_energy_{cost_tag}_{cogen_tag}.png')


def compute_ccus_npv(df):
  # Compute NPV_NG_CCUS from NG_Cost and cost of CCUS
  df['CCUS Cost ($/y)'] = df['Thermal MWh/hr']*8760*utils.heat_avg_carbon_intensity*utils.ccus_cost/utils.mmbtu_to_mwh
  df['NPV_NG_CCUS ($)'] = -lifetime*df['CCUS Cost ($/y)'] + df['NG_Cost (counterfactual)']
  df['NPV_NG_CCUS ($/MWt)'] = df['NPV_NG_CCUS ($)']/df['Thermal MWh/hr']
  df['NPV_NG_noCCUS ($/MWt)'] = df['NG_Cost (counterfactual)']/df['Thermal MWh/hr']
  return df


def plot_anr_vs_ng_npv(df, cost_tag='foak', cogen=False):
  npv = 'NPV_'+cost_tag.upper()+' (per MWt)'
  y = 'NPV_'+cost_tag.upper()+' (M$/MWt)'
  df[y] = df[npv]/1e6
  df['NPV NG (M$/MWt)'] = df['NPV_NG_noCCUS ($/MWt)']/1e6
  df['NPV NG with CCUS (M$/MWt)'] = df['NPV_NG_CCUS ($/MWt)']/1e6
  fig, ax = plt.subplots(2,1,figsize=(6,6))
  sns.scatterplot(ax=ax[0], data=df, y=y, x='NPV NG (M$/MWt)', color='red', style='GEN_Opt')
  med_x = np.arange(0,6, 0.1)
  ax[0].plot(med_x, med_x, color='grey', linestyle='--', linewidth=0.8)
  ax[0].spines[['right', 'top']].set_visible(False)
  sns.scatterplot(ax=ax[1], data=df, y=y, x='NPV NG with CCUS (M$/MWt)', color='green',style='GEN_Opt')
  ax[1].plot(med_x, med_x, color='grey', linestyle='--', linewidth=0.8)
  ax[1].spines[['right', 'top']].set_visible(False)
  plt.show()



def main():
  if foak: cost_tag = 'foak'
  else: cost_tag = 'noak'
  heat_df = get_data(foak)
  heat_df = compute_ccus_npv(heat_df)
  print(heat_df)
  print(heat_df.columns)
  save_data(heat_df, cost_tag=cost_tag, cogen_tag='nocogen')
  plot_net_annual_revenues(heat_df, cost_tag=cost_tag, cogen_tag='nocogen')
  plot_anr_vs_ng_npv(heat_df, cost_tag=cost_tag)



if __name__ == '__main__':
  main()
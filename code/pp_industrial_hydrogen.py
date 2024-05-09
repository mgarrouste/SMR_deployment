import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import cashflows_color_map, palette, letter_annotation
import seaborn as sns

OAK = 'FOAK'

cogen_tag = False
industries = {'ammonia':'Ammonia', 
              'refining':'Refining', 
              'steel':'Steel'}
anr_design_palette = {'HTGR':'blue', 
                      'iMSR':'orange', 
                      'iPWR':'green', 
                      'PBR-HTGR':'darkorchid', 
                      'Microreactor':'darkgrey'}

def load_data(OAK):
  list_df = []
  for ind, ind_label in industries.items():
    df = pd.read_excel(f'./results/clean_results_anr_{OAK}_h2_wacc_0.077.xlsx', sheet_name=ind)
    df['Industry'] = ind_label
    list_df.append(df)
  total_df = pd.concat(list_df, ignore_index=True)
  return total_df


def compute_normalized_net_revenues(df, OAK):
  anr_data = pd.read_excel('./ANRs.xlsx', sheet_name=OAK)
  anr_data = anr_data[['Reactor', 'Thermal Efficiency']]
  df = df.merge(anr_data, left_on='ANR type', right_on='Reactor')
  df['Net Annual Revenues (M$/MWe/y)'] = df['Net Annual Revenues ($/MWe/y)']/1e6
  df['Net Annual Revenues with H2 PTC (M$/MWe/y)'] = df['Net Annual Revenues with H2 PTC ($/MWe/y)']/1e6
  df['Net Annual Revenues with H2 PTC with elec (M$/MWe/y)'] = df['Net Revenues with H2 PTC with elec ($/year)']/(df['Depl. ANR Cap. (MWe)']*1e6)
  return df



def plot_net_annual_revenues(df):
  save_path = f'./results/industrial_hydrogen_anr_net_annual_revenues_{OAK}_cogen_{cogen_tag}.png'
  print(f'Plot net annual revenues: {save_path}')
  fig, ax = plt.subplots(figsize=(6,4))
  if cogen_tag: x = 'Net Annual Revenues with H2 PTC with elec (M$/MWe/y)'
  else: x = 'Net Annual Revenues with H2 PTC (M$/MWe/y)'
  sns.boxplot(ax=ax, data=df, y='Industry', x=x, color='black',fill=False, width=.5)
  sns.stripplot(ax=ax, data=df, y='Industry', x=x, hue='ANR type', palette=palette)
  ax.set_ylabel('')
  ax.set_xlabel('Net Annual Revenues (M$/MWe/y)')
  ax.get_legend().set_visible(False)
  ax.set_xlim(-0.8, 0.8)
  ax.xaxis.set_ticks(np.arange(-0.75, 1, 0.25))
  sns.despine()
  ax.xaxis.grid(True)
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels,  bbox_to_anchor=(.5,1.08),loc='upper center', ncol=3)
  fig.tight_layout()
  fig.savefig(save_path, bbox_inches='tight')



def plot_mean_cashflows(df):
  save_path = f'./results/industrial_hydrogen_anr_avg_cashflows_{OAK}_cogen_{cogen_tag}.png'
  print(f'Plot average cashflows: {save_path}')
  # Cashflows in M$/MWe/y
  df['ANR CAPEX'] = -df['ANR CAPEX ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 CAPEX'] = -df['H2 CAPEX ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['ANR O&M'] = -df['ANR O&M ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 O&M'] = -df['H2 O&M ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['Avoided Fossil Fuel Costs'] = df['Avoided NG costs ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 PTC'] = df['H2 PTC Revenues ($/year)']/(1e6*df['Depl. ANR Cap. (MWe)'])
  if cogen_tag:
    df['Electricity'] = df['Electricity revenues ($/y)']/(1e6*df['Depl. ANR Cap. (MWe)'])
    design_df = df[['Industry','ANR type','ANR CAPEX', 'H2 CAPEX', 'ANR O&M', 'H2 O&M', 'Avoided Fossil Fuel Costs', \
                    'H2 PTC', 'Electricity']]
  else:
    design_df = df[['Industry','ANR type','ANR CAPEX', 'H2 CAPEX', 'ANR O&M', 'H2 O&M', 'Avoided Fossil Fuel Costs', 'H2 PTC']]
  design_df = design_df.groupby([ 'ANR type','Industry']).mean()
  design_df.to_excel( f'./results/industrial_hydrogen_anr_avg_cashflows_{OAK}_cogen_{cogen_tag}.xlsx')
  fig, ax = plt.subplots(figsize = (8,6))
  design_df.plot(ax = ax, kind ='bar', stacked=True, color=cashflows_color_map)
  ax.set_ylabel('Average Normalized Cashflows (M$/MWe/y)')
  ax.set_xlabel('')
  ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
  ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=50, ha='right')
  ax.set_ylim(-1.3, 2)
  ax.yaxis.set_ticks(np.arange(-1.25, 2, 0.25))
  ax.get_legend().set_visible(False)
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels,  bbox_to_anchor=(.5,1.08),loc='upper center', ncol=3)
  fig.tight_layout()
  fig.savefig(save_path, bbox_inches='tight')
  return fig


def plot_abatement_cost(df, fig=None):
  save_path = f'./results/industrial_hydrogen_abatement_cost_{OAK}_cogen_{cogen_tag}.png'
  print(f'Plot average cashflows: {save_path}')
  df['Cost ANR ($/y)'] = df['ANR CAPEX ($/year)']+df['H2 CAPEX ($/year)']+df['ANR O&M ($/year)']+df['H2 O&M ($/year)']\
                        +df['Conversion costs ($/year)']-df['Avoided NG costs ($/year)']
  df['Abatement cost ($/tCO2)'] = df['Cost ANR ($/y)']/(df['Ann. avoided CO2 emissions (MMT-CO2/year)']*1e6)
  df['Abatement potential (tCO2/y-MWe)'] = 1e6*df['Ann. avoided CO2 emissions (MMT-CO2/year)']/df['Depl. ANR Cap. (MWe)']
  if fig: ax = fig.subplots(2,1)
  else: fig, ax = plt.subplots(2,1, figsize=(7,5))
  sns.boxplot(ax=ax[0], data=df, y='Industry', x='Abatement cost ($/tCO2)',color='black',fill=False, width=.5)
  sns.stripplot(ax=ax[0], data=df, y='Industry', x='Abatement cost ($/tCO2)', hue='ANR type', palette=palette,alpha=.6)
  letter_annotation(ax[0], -.25, 1, 'I-a')
  ax[0].set_ylabel('')
  ax[0].get_legend().set_visible(False)
  sns.boxplot(ax=ax[1], data=df, y='Industry', x='Abatement potential (tCO2/y-MWe)',color='black', fill=False, width=.5)
  sns.stripplot(ax=ax[1], data=df, y='Industry', x='Abatement potential (tCO2/y-MWe)', hue='ANR type', palette=palette, alpha=.6)
  letter_annotation(ax[1], -.25, 1, 'I-b')
  ax[1].set_ylabel('')
  ax[1].get_legend().set_visible(False)
  ax[1].set_xlim(-10,7010)
  sns.despine()
  if not fig:
    fig.savefig(save_path, bbox_inches='tight')


def main():
  total_df = load_data(OAK)
  total_df = compute_normalized_net_revenues(total_df, OAK)
  total_df.to_csv(f'./results/industrial_hydrogen_avg_cashflows_{OAK}_cogen_{cogen_tag}.csv')
  total_df[['Industry', 'Net Annual Revenues with H2 PTC (M$/MWe/y)']].describe(\
    percentiles=[.1,.25,.5,.75,.9]).to_csv(f'./results/industrial_hydrogen_avg_cashflows_stats_{OAK}_cogen_{cogen_tag}.csv')
  plot_net_annual_revenues(total_df)
  plot_mean_cashflows(total_df)
  plot_abatement_cost(total_df)


if __name__ == '__main__':
  main()
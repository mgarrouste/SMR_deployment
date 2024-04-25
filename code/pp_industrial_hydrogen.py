import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import seaborn as sns

OAK = 'FOAK'
BE= False
cogen_tag = False
industries = {'ammonia':'Ammonia', 
              'process_heat':'Hydrogen \nProcess Heat', 
              'refining':'Refining', 
              'steel':'Steel'}
anr_design_palette = {'HTGR':'blue', 
                      'iMSR':'orange', 
                      'iPWR':'green', 
                      'PBR-HTGR':'darkorchid', 
                      'Microreactor':'darkgrey'}

def load_data(OAK, BE):
  list_df = []
  for ind, ind_label in industries.items():
    df = pd.read_excel(f'./results/raw_results_anr_{OAK}_h2_wacc_0.077_BE_{BE}.xlsx', sheet_name=ind)
    df['Industry'] = ind_label
    list_df.append(df)
  total_df = pd.concat(list_df, ignore_index=True)
  return total_df


def compute_normalized_net_revenues(df, OAK):
  anr_data = pd.read_excel('./ANRs.xlsx', sheet_name=OAK)
  anr_data = anr_data[['Reactor', 'Thermal Efficiency']]
  df = df.merge(anr_data, left_on='ANR type', right_on='Reactor')
  df['Net Annual Revenues (M$/MWt/y)'] = df['Net Annual Revenues ($/MWe/y)']*df['Thermal Efficiency']/1e6
  df['Net Annual Revenues with H2 PTC (M$/MWt/y)'] = df['Net Annual Revenues with H2 PTC ($/MWe/y)']*df['Thermal Efficiency']/1e6
  return df


def compute_remaining_capacity(df):
  """Computes remaining unused capacity between demand from industrial process and installed ANR capacity"""
  # TODO


def plot_net_annual_revenues(df):
  save_path = f'./results/industrial_hydrogen_anr_net_annual_revenues_{OAK}_cogen_{cogen_tag}.png'
  print(f'Plot net annual revenues: {save_path}')
  fig, ax = plt.subplots(figsize=(6,4))
  df = df.replace({'Micro':'Microreactor'})
  sns.boxplot(ax=ax, data=df, y='Industry', x='Net Annual Revenues with H2 PTC (M$/MWt/y)', color='black',\
              fill=False, width=.5)
  sns.stripplot(ax=ax, data=df, y='Industry', x='Net Annual Revenues with H2 PTC (M$/MWt/y)', hue='ANR type', \
              palette = anr_design_palette)
  ax.set_ylabel('')
  ax.set_xlabel('Net Annual Revenues (M$/MWt/y)')
  ax.get_legend().set_visible(False)
  sns.despine()
  ax.xaxis.grid(True)
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels,  bbox_to_anchor=(.5,1.08),loc='upper center', ncol=3)
  fig.tight_layout()
  fig.savefig(save_path, bbox_inches='tight')



def plot_mean_cashflows(df):
  save_path = f'./results/industrial_hydrogen_anr_avg_cashflows_{OAK}_cogen_{cogen_tag}.png'
  print(f'Plot average cashflows: {save_path}')
  # Cashflows in M$/MWt/y
  df['ANR CAPEX'] = -df['ANR CAPEX ($/year)']*df['Thermal Efficiency']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 CAPEX'] = -df['H2 CAPEX ($/year)']*df['Thermal Efficiency']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['ANR O&M'] = -df['ANR O&M ($/year)']*df['Thermal Efficiency']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 O&M'] = -df['H2 O&M ($/year)']*df['Thermal Efficiency']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['Avoided Fossil Fuel Costs'] = df['Avoided NG costs ($/year)']*df['Thermal Efficiency']/(1e6*df['Depl. ANR Cap. (MWe)'])
  df['H2 PTC'] = df['H2 PTC Revenues ($/year)']*df['Thermal Efficiency']/(1e6*df['Depl. ANR Cap. (MWe)'])
  design_df = df[['Industry','ANR type','ANR CAPEX', 'H2 CAPEX', 'ANR O&M', 'H2 O&M', 'Avoided Fossil Fuel Costs', 'H2 PTC']]
  design_df = design_df.groupby([ 'ANR type','Industry']).mean()
  design_df.to_excel( f'./results/industrial_hydrogen_anr_avg_cashflows_{OAK}_cogen_{cogen_tag}.xlsx')
  color_map = {'ANR CAPEX': 'royalblue', 
               'H2 CAPEX': 'lightsteelblue', 
               'ANR O&M':'forestgreen', 
               'H2 O&M':'palegreen',
               'Avoided Fossil Fuel Costs':'darkorchid', 
               'H2 PTC':'plum'}
  fig, ax = plt.subplots(figsize = (8,6))
  design_df.plot(ax = ax, kind ='bar', stacked=True, color=color_map)
  ax.set_ylabel('Average Normalized Cashflows (M$/MWt/y)')
  ax.set_xlabel('')
  ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
  ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=50, ha='right')
  ax.set_ylim(-0.42, 0.62)
  ax.yaxis.set_ticks(np.arange(-0.4, 0.7, 0.1))
  ax.get_legend().set_visible(False)
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels,  bbox_to_anchor=(.5,1.08),loc='upper center', ncol=3)
  fig.tight_layout()
  fig.savefig(save_path, bbox_inches='tight')



def main():
  total_df = load_data(OAK, BE)
  total_df = compute_normalized_net_revenues(total_df, OAK)
  total_df.to_csv('./results/temp.csv')
  plot_net_annual_revenues(total_df)
  plot_mean_cashflows(total_df)


if __name__ == '__main__':
  main()
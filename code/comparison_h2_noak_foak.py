import pandas as pd 
import matplotlib.pyplot as plt
import os
import utils
import seaborn as sns


def load_h2_clean_results():
  noak_results = './results/clean_results_anr_NOAK_h2_wacc_0.077.xlsx'
  foak_results = './results/clean_results_anr_FOAK_h2_wacc_0.077.xlsx'
  assert os.path.isfile(foak_results), f'FOAK results not found: {foak_results}'
  assert os.path.isfile(foak_results), f'FOAK results not found: {foak_results}'
  industries = {'process_heat':'Process heat', 'ammonia':'Ammonia', 'steel': 'Steel', 'refining':'Refining'}
  noak_l = []
  foak_l = []
  for industry, label in industries.items():
    ind_noak_df = pd.read_excel(noak_results, sheet_name=industry)
    ind_noak_df['Industry'] = label
    noak_l.append(ind_noak_df)
    ind_foak_df = pd.read_excel(foak_results, sheet_name=industry)
    ind_foak_df['Industry'] = label
    foak_l.append(ind_foak_df)
  noak_df = pd.concat(noak_l, ignore_index=True)
  foak_df = pd.concat(foak_l, ignore_index=True)
  return noak_df, foak_df



def compute_net_annual_revenues(df):
  """Computes net annual revenues from 
  - 'Net Revenues ($/year)' column that corresponds to the ANR and H2 annualized costs,
  - 'Depl. ANR Cap. (MWe)', deployed electrical capacity in MWe
  - 'H2 Dem. (kg/day)', hydrogen demand
  - Adds revenues from hydrogen PTC
  """
  df['Net Annual Revenues (M$/MWe/y)'] = (df['Net Revenues ($/year)'] \
                                          + df['H2 Dem. (kg/day)']*utils.h2_ptc*365)/(1e6*df['Depl. ANR Cap. (MWe)'])
  return df



def plot_deployment_comparison(foak_df, noak_df):
  total_df = pd.merge(left=noak_df, right=foak_df, on=['id', 'Industry'], how='left',suffixes=['_NOAK', '_FOAK'])
  total_df['Delta Net Rev'] = total_df['Net Annual Revenues (M$/MWe/y)_NOAK'] - total_df['Net Annual Revenues (M$/MWe/y)_FOAK']
  # Plot
  fig, ax = plt.subplots(figsize=(5,5))
  colors = {'Ammonia':'blue', 'Refining':'limegreen', 'Process heat':'darkorange', 'Steel':'red'} 
  sns.boxplot(ax=ax, data=total_df, y='Industry', x='Delta Net Rev', hue='Industry', fill=False, width=.3, palette=colors)
  ax.set_ylabel('')
  ax.set_xlim(0, 0.25)
  ax.set_xlabel('Differential Net Annual Revenues (M$/MWe/y)')
  sns.despine()
  ax.grid(True)  
  fig.tight_layout()
  fig.savefig('./results/h2_comparison_FOAK_NOAK_anr_net_ann_revenues.png')

def main():
  noak_df, foak_df = load_h2_clean_results()
  noak_df, foak_df = compute_net_annual_revenues(noak_df), compute_net_annual_revenues(foak_df)
  plot_deployment_comparison(foak_df, noak_df)


if __name__ == '__main__':
  main()
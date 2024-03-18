import matplotlib.pyplot as plt 
import pandas as pd
import os
from itertools import product
import numpy as np
from matplotlib.lines import Line2D

INDUSTRIES = ['ammonia', 'process_heat', 'refining','steel']
years = [2024, 2030, 2040]
cambium_scenarios = ['HighRECost','LowRECostTCExpire','MidCaseTCExpire', 'MidCase', 'LowRECost',\
                    'HighNGPrice', 'LowNGPrice']


def get_average_revenues(year, industry, scenario):
  """Compute the average revenues for ANRs in ANR-H2 and electricity dedicated ANRs
  Args: 
    year (int): year
    industry (str): indsutrial sector
    scenario (str): cambium scenario
  Returns:
    avg_h2 (float): Average revenue for ANR in ANR-H2 in M$/year/MWe
    avg_elec (float): Average revenue for ANR producing electricity in M$/year/MWe
  """
  assert year in years
  assert industry in INDUSTRIES
  assert scenario in cambium_scenarios

  result_file = './results/electricity_prod_results_no_learning_'+scenario+'_'+str(year)+'.xlsx'
  try:
    res_df = pd.read_excel(result_file, sheet_name=industry, index_col=0)
    avg_elec = res_df['Electricity sales (M$/year/MWe)'].mean()
    avg_h2 = (res_df['H2 PTC revenues (M$/year/MWe)']+res_df['Avoided fossil fuel cost (M$/year/MWe)']).mean()
  except FileNotFoundError:
    print(f'No results for {year}, {industry}, {scenario}')
    avg_h2 = None
    avg_elec = None
  return avg_h2, avg_elec



def plot_sa_elec_revenues(sa_df):
  """Plot average revenues for anr-h2 vs dedicated electricity anr for a designated year
  Args:
    sa_df (DataFrame) : results of the SA runs, with index ('Year', 'Industry', 'Scenario')
  Returns: 
    None
  """

  #year_data = sa_df.loc[year]
  sa_df = sa_df.rename(index={'ammonia':'Ammonia', 
                              'process_heat':'HT Process Heat', 
                              'refining':'Refining', 
                              'steel':'Steel'})

  fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 10), sharex=True, sharey=True)

  industry_colors = {
      'Ammonia': 'blue',
      'HT Process Heat': 'orange',
      'Refining':'green',
      'Steel':'red'
  }

  scenarios_markers = {
    'HighRECost': 'x',
    'LowRECostTCExpire': 'o',
    'MidCaseTCExpire': 'v', 
    'MidCase': 's', 
    'LowRECost': 'D',
    'HighNGPrice': '^', 
    'LowNGPrice':'+'
  }

  for idx, year in enumerate([2024, 2030, 2040]):
    ax = axes[idx]
    year_data = sa_df.loc[year]
    legend_handles = []
    legend_labels = []

    # Add legend handles and labels for industries
    for industry, color in industry_colors.items():
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=f'{industry}'))
        legend_labels.append(industry)

    # Add legend handles and labels for scenarios
    for scenario, marker in scenarios_markers.items():
        if year != 2040 and not (scenario in ['LowRECostTCExpire', 'MidCaseTCExpire']):
          legend_handles.append(Line2D([0], [0], marker=marker, color='black', linestyle='None', label=f'{scenario}'))
          legend_labels.append(scenario)
        if year ==2040:
          legend_handles.append(Line2D([0], [0], marker=marker, color='black', linestyle='None', label=f'{scenario}'))
          legend_labels.append(scenario)


    for industry, industry_data in year_data.groupby(level='Industry'):
        for scenario, scenario_data in industry_data.groupby(level='Scenario'):
            ax.scatter(scenario_data['Avg electricity revenues (M$/year/MWe)'], scenario_data['Avg H2 revenues (M$/year/MWe)'], 
                      label=f'{industry}-{scenario}',
                      color = industry_colors[industry],
                      marker=scenarios_markers[scenario])
    # Plot dashed gray line for the median (y = x)
    ax.plot([0, 1.5], [0, 1.5], color='gray', linestyle='--', label='Median (y = x)')
    ax.set_xlabel('Average electricity revenues\n(M$/year/MWe)')
    ax.set_ylabel(R'Average $H_2$ revenues' '\n(M$/year/MWe)')
    ax.set_xlim(.0, 0.4)
    ax.set_ylim(.2, 1.5)
    ax.set_xticks(np.arange(0., 0.41, 0.05))
    ax.set_title(f'Year {year}')
  # Add custom legend
  fig.legend(handles=legend_handles, labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

  #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  # Adjust layout to prevent overlapping
  plt.tight_layout()
  plt.savefig(f'./results/electricity_prod_results_no_learning_total', bbox_inches='tight')  

  plt.close()



def main():
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  print(get_average_revenues(2024,'ammonia', 'MidCase'))


  df= pd.DataFrame(columns=['Year', 'Industry', 'Scenario', 'Avg H2 revenues (M$/year/MWe)', \
                            'Avg electricity revenues (M$/year/MWe)'])
  #df.loc[('at', 1),'Dwell']
  df.set_index(['Year', 'Industry', 'Scenario'], inplace=True)
  for year, industry, scenario in product(years, INDUSTRIES, cambium_scenarios):
    avg_h2, avg_elec = get_average_revenues(year, industry, scenario)
    df.loc[(year, industry, scenario), 'Avg H2 revenues (M$/year/MWe)'] = avg_h2
    df.loc[(year, industry, scenario), 'Avg electricity revenues (M$/year/MWe)'] = avg_elec
  df.dropna(inplace=True)
  df.to_excel('./results/electricity_prod_results_no_learning_total.xlsx')
  plot_sa_elec_revenues(df)

if __name__ == '__main__':
  main()


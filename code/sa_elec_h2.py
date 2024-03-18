import matplotlib.pyplot as plt 
import pandas as pd
import os
from itertools import product
import seaborn as sns
import numpy as np

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



def plot_sa_elec_revenues(sa_df, year):
  """Plot average revenues for anr-h2 vs dedicated electricity anr for a designated year
  Args:
    sa_df (DataFrame) : results of the SA runs, with index ('Year', 'Industry', 'Scenario')
  Returns: 
    None
  """

  year_data = sa_df.loc[year]

  fig, ax = plt.subplots()

  industry_colors = {
      'ammonia': 'blue',
      'process_heat': 'orange',
      'refining':'green',
      'steel':'red'
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
  if year == 2024:
    ax.set_xlim(.2, 0.4)
    ax.set_ylim(.2, 1.5)
    ax.set_xticks(np.arange(0.2, 0.41, 0.05))
  elif year == 2030:
    ax.set_xlim(.1, 0.3)
    ax.set_ylim(.1, 1.5)
    ax.set_xticks(np.arange(0.1, 0.31, 0.05))
  elif year == 2040:
    ax.set_xlim(.0, 0.3)
    ax.set_ylim(.0, 1.5)
    ax.set_xticks(np.arange(0, 0.31, 0.05))
  ax.set_title(f'Year {year}')
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  plt.savefig(f'./results/electricity_prod_results_no_learning_total_{year}', bbox_inches='tight')  

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
  for year in years:
    plot_sa_elec_revenues(df, year)

if __name__ == '__main__':
  main()


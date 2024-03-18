import matplotlib.pyplot as plt 
import pandas as pd
import os
from itertools import product

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
  print(df)
  df.dropna(inplace=True)
  df.to_excel('./results/electricity_prod_results_no_learning_total.xlsx')

if __name__ == '__main__':
  main()


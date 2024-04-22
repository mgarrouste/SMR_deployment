from pyomo.environ import *
import pandas as pd
import numpy as np
import os
import utils
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

WACC = utils.WACC
ITC_ANR = utils.ITC_ANR

anr_tag = 'NOAK' # 'FOAK'

cambium_scenario = 'MidCase'#'MidCaseTCExpire' # 'LowRECostTCExpire','MidCaseTCExpire', 'MidCase', 'LowRECost', 'HighRECost', 'HighNGPrice', 'LowNGPrice'

excel_file = f'./results/price_taker_{anr_tag}_{cambium_scenario}.xlsx' 

electricity_prices_partial_path = './input_data/cambium_'+cambium_scenario.lower()+'_state_hourly_electricity_prices/Cambium22_'+cambium_scenario+'_hourly_'


def get_electricity_prices(state, year):
  """Get the type and number of ANR for a site
  Args: 
    state (str): abbreviation of the state name
    year (int): Year
  Returns: 
    prices (DataFrame): with columns t, 0 to 8760, and price, electricity price in $/MWhe for year 
  """
  if state =='AK': state ='AR' # Arkansas abbreviation
  if state =='OA': state = 'IA' # Iowa abbreviation fix
  if state =='HI': state = 'CA' # Hawai as California: high prices

  elec_prices_path = electricity_prices_partial_path +state +'_'+str(year)+'.csv'
  electricity_prices = pd.read_csv(elec_prices_path, skiprows=5)
  electricity_prices = electricity_prices[['energy_cost_enduse']]
  # Convert to USD2020
  from utils import conversion_2021usd_to_2020usd
  electricity_prices['energy_cost_enduse'] = electricity_prices['energy_cost_enduse']*conversion_2021usd_to_2020usd
  # Index
  electricity_prices.set_index([pd.Index([t for t in range(8760)])])
  electricity_prices.reset_index(inplace=True)
  electricity_prices.rename(columns={'index':'t', 'energy_cost_enduse':'price'}, inplace=True)
  electricity_prices.set_index('t', inplace=True)

  return electricity_prices


def build_ED_electricity(state, ANRtype, ANR_data, year):
  """
  Performs economic dispatch of a type of ANR in a state given electricity prices
  Args: 
    state (str): abbreviation for state name
    ANRtype (str): design of ANR
    ANR_data (DataFrame): ANR techno-economic parameter data
    year (int): Year for electricity prices
  Returns: 

  """
  model = ConcreteModel(id)

  ### Sets ###
  
  model.t = Set(initialize = np.arange(8760), doc='Time') # hours in one year

  ### Variables ###
  model.vG = Var(model.t, within=NonNegativeReals)

  ### Parameters ###
  # Financial
  model.pWACC = Param(initialize = WACC)
  model.pITC_ANR = Param(initialize = ITC_ANR)
  # ANR
  model.pNbMod = Param(initialize = int(ANR_data.loc[ANRtype]['Max Modules']), \
                       doc='Number of modules deployed chosen as max modules')
  model.pANRCap =  Param(initialize = model.pNbMod*float(ANR_data.loc[ANRtype]['Power in MWe'])\
                         , doc='Total capacity deployed (MWe)')
  model.pANRCAPEX = Param(initialize = float(ANR_data.loc[ANRtype]['CAPEX $/MWe']))
  model.pANRVOM = Param(initialize = float(ANR_data.loc[ANRtype]['VOM in $/MWh-e']))
  model.pANRFOM = Param(initialize = float(ANR_data.loc[ANRtype]['FOPEX $/MWe-y']))
  model.pANRRampRate = Param(initialize = float(ANR_data.loc[ANRtype]['Ramp Rate (fraction of capacity/hr)']))
  model.pANRMSL = Param(initialize = model.pNbMod*float(ANR_data.loc[ANRtype]['MSL in MWe']))
  @model.Param()
  def pANRCRF(model):
    return model.pWACC / (1 - (1/(1+model.pWACC)**float(ANR_data.loc[ANRtype,'Life (y)'])))

  electricity_prices = get_electricity_prices(state=state, year=year)
  @model.Param(model.t)
  def pEPrice(model, t):
    return electricity_prices.loc[t,'price']
    
  ### Objective ###
  def annualized_revenues(model):
    return -(model.pANRCAPEX*model.pANRCRF*(1-model.pITC_ANR) + model.pANRFOM)*model.pANRCap \
               +sum((-model.pANRVOM+model.pEPrice[t])*model.vG[t] for t in model.t)
  model.NetRevenues = Objective(expr = annualized_revenues, sense= maximize)

  ### Constraints ###
  def max_capacity(model, t):
    return model.vG[t] <= model.pANRCap
  def msl(model, t):
    return model.vG[t] >= model.pANRMSL
  model.max_capacity = Constraint(model.t, rule=max_capacity, doc='Maximum capacity')
  model.msl = Constraint(model.t, rule=msl, doc='Minimum stable load')

  def RRamprateUp(model,t):
    if t == model.t.at(1): 
        return model.vG[t] == model.pANRMSL
    return (model.vG[model.t.prev(t)])-(model.vG[t]) >= -1*model.pANRRampRate*model.pANRCap
  model.RRamprateUp = Constraint(model.t, rule = RRamprateUp, doc = "The hourly change must not exceed the ramprate between time steps")

  def RRamprateDn(model,t):
    if t == model.t.at(1):
        return model.vG[t] == model.pANRMSL
        #return pe.Constraint.Feasible # V11C THis means that we can start iur reactor at any output.
    return (model.vG[model.t.prev(t)]-model.vG[t]) <= model.pANRRampRate*model.pANRCap
  model.RRamprateDn = Constraint(model.t, rule = RRamprateDn, doc = "The hourly change must not exceed the ramprate between time steps")

  return model


def solve_ED_electricity(state, ANRtype, ANR_data, year):
  print(f'Start price taker solve in state {state}, design {ANRtype}')
  model = build_ED_electricity(state, ANRtype, ANR_data, year)

  solver = SolverFactory('glpk')
  results = solver.solve(model, tee=False)
  if results.solver.termination_condition == TerminationCondition.optimal: 
    model.solutions.load_from(results)
  else:
    exit('Not solvable')
  
  results_dic = {}
  results_dic['Annual Net Revenues ($/year/MWe)'] = value(model.NetRevenues/model.pANRCap)
  results_dic['ANR type'] = ANRtype
  results_dic['state'] = state
  results_dic['year'] = year
  def compute_elec_sales(model):
    return sum(model.vG[t]*model.pEPrice[t] for t in model.t)/(1e6*model.pANRCap)
  results_dic['Electricity sales (M$/year/MWe)'] = value(compute_elec_sales(model))

  def compute_avg_elec_price(model):
    return sum(model.pEPrice[t] for t in model.t)/8760
  results_dic['Avg price ($/MWhe)'] = value(compute_avg_elec_price(model))
  return results_dic

def save_electricity_results(results_df, excel_file):
  """Save electricity results
  Args: 
    excel_file (str): path to excel file for electricity results
  Returns: 
    None 
  """
  try:
    with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
      with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_df.to_excel(writer)
  except FileNotFoundError:
    results_df.to_excel(excel_file)


def plot_results(excel_file, boxplot=True):
  """ Plot Net Annual Revenues for each ANR design"""
  fig, ax = plt.subplots(figsize=(6,3))
  df = pd.read_excel(excel_file, header=0, index_col=0)
  df['Annual Net Revenues (M$/year/MWe)'] = df['Annual Net Revenues ($/year/MWe)']/1e6
  print(df.columns)
  df.replace({'Micro':'Microreactor'}, inplace=True)
  palette={'HTGR':'blue', 'iMSR':'orange', 'iPWR':'green', 'PBR-HTGR':'darkorchid', 'Microreactor':'darkgrey'}
  if boxplot:
    sns.boxplot(ax=ax, data=df, x='Annual Net Revenues (M$/year/MWe)', y='ANR type', palette=palette, hue='ANR type')
  else:
    sns.stripplot(ax=ax, data=df, x='Annual Net Revenues (M$/year/MWe)', y='ANR type', palette=palette, \
                hue='ANR type', marker='*', size=7)
  ax.axvline(x=0, color='grey', linestyle='--', linewidth=1)
  ax.set_ylabel('')
  fig.tight_layout()
  fig.savefig(f'./results/electricity_price_taker_net_annual_revenues_{anr_tag}_{cambium_scenario}.png')


def main():
  print(f'ANR costs: {anr_tag}')
  ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name=anr_tag, index_col=0)
  # save path

  states = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', \
            'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', \
              'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WY']
  ANRtype_list = ['iPWR', 'HTGR', 'PBR-HTGR', 'iMSR', 'Micro']
  years = [2024, 2030, 2040]
  all_states_results_list = []
  for year in years:
    for state in states: 
      
      # Parallel solving
      with Pool(5) as pool:
        results = pool.starmap(solve_ED_electricity, [(state, ANRtype, ANR_data, year) for ANRtype in ANRtype_list])
      pool.close()

      state_elec_results_df = pd.DataFrame(results)
      all_states_results_list.append(state_elec_results_df)
  all_states_elec_results_df = pd.concat(all_states_results_list, ignore_index=True)
  save_electricity_results(all_states_elec_results_df, excel_file)



if __name__ == '__main__':
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--plot', required=False, help='Only plot results, does not run model')
  args = parser.parse_args()
  if args.plot:
    plot_results(excel_file)
  else:
    main()

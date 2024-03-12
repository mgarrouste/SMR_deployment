from pyomo.environ import *
import pandas as pd
import numpy as np
import os
from utils import load_data
import utils
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns

WACC = utils.WACC
ITC_ANR = utils.ITC_ANR
INDUSTRIES = ['ammonia', 'process_heat', 'refining','steel']
learning = False
year = 2024

electricity_prices_partial_path = './input_data/cambium_midcase_state_hourly_electricity_prices/Cambium22_MidCase_hourly_'

def load_industry_results(industry):
  """Load and returns the results of the deployment optimization for an industry
  Args:
    industry (str): Name of the industry: refining, ammonia, process_heat, steel
  Returns:
    DataFrame: Results of ANR-H2 deployment optimization for industry
  """
  assert industry in INDUSTRIES, f'{industry} not recognized ({INDUSTRIES})'
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  industry_df = pd.read_excel('./results/clean_results_anr_lr_0_h2_lr_0_wacc_0.077.xlsx', sheet_name=industry, index_col=0)
  return industry_df


def compute_avoided_ff_costs(industry_df, industry):
  """Compute the avoided fossil fuel costs for each industry
  Args: 
    industry_df (DataFrame): Results of optimization of ANR depl. 
    industry (str): Name of the industry
  Returns: 
    industry_df (DataFrame): Results with added column 'Avoided fossil fuel cost (bn$/year)'
  """
  if industry == 'ammonia': 
    industry_df['Avoided fossil fuel cost (M$/year/MWe)'] = utils.nh3_nrj_intensity*industry_df['Ammonia capacity (tNH3/year)']\
      *industry_df['Breakeven price ($/MMBtu)']/(1e6*industry_df['Depl. ANR Cap. (MWe)'])
  elif industry == 'process_heat':
    industry_df['Avoided fossil fuel cost (M$/year/MWe)'] = industry_df['Heat Dem. (MJ/year)']*industry_df['Breakeven price ($/MMBtu)']\
    /(1e6*industry_df['Depl. ANR Cap. (MWe)']*utils.mmbtu_to_mj)
  elif industry == 'refining': 
    industry_df['Avoided fossil fuel cost (M$/year/MWe)'] = industry_df['H2 Dem. (kg/day)']*365*utils.smr_nrj_intensity\
      *industry_df['Breakeven price ($/MMBtu)']/(1e6*industry_df['Depl. ANR Cap. (MWe)'])
  elif industry == 'steel':
    industry_df['Avoided fossil fuel cost (M$/year/MWe)'] = industry_df['Steel prod. (ton/year)']*utils.coal_to_steel_ratio_bau\
      *utils.coal_heat_content*industry_df['Breakeven price ($/MMBtu)']/(1e6*industry_df['Depl. ANR Cap. (MWe)'])
  return industry_df


def get_opt_anr(industry_df, id):
  """Get the type and number of ANR for a site
  Args: 
    industry_df (DataFrame): Results of optimization of ANR depl.
    id (str): site id
  Returns: 
    reactor (str): type of reactor deployed
    number (int): number of reactors deployed
  """
  reactor = industry_df.loc[id, 'ANR type']
  number = industry_df.loc[id, '# ANR modules']
  return reactor, number


def get_electricity_prices(industry_df, id, year):
  """Get the type and number of ANR for a site
  Args: 
    industry_df (DataFrame): Results of optimization of ANR depl.
    id (str): site id
    year (int): Year
  Returns: 
    prices (DataFrame): with columns t, 0 to 8760, and price, electricity price in $/MWhe for year 
  """
  state = industry_df.loc[id, 'state']
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


def build_ED_electricity(industry_df, id, ANR_data, year):
  """
  Performs economic dispatch of reactors at industrial site for type and capacity determined by opt industrial deployment
  Args: 
    industry_df (DataFrame): Results of optimization of ANR depl.
    id (str): site id
    ANR_data (DataFrame): ANR techno-economic parameter data
    year (int): Year for electricity prices
  Returns: 

  """
  model = ConcreteModel(id)

  ### Sets ###
  
  model.t = Set(initialize = np.arange(8760), doc='Time') # hours in one year
  ANRtype, ANRnb = get_opt_anr(industry_df, id)
  #model.N = Set(initialize = list(range(ANRnb)))

  ### Variables ###
  model.vG = Var(model.t, within=NonNegativeReals)

  ### Parameters ###
  # Financial
  model.pWACC = Param(initialize = WACC)
  model.pITC_ANR = Param(initialize = ITC_ANR)
  # ANR
  model.pANRCap =  Param(initialize = ANRnb*float(ANR_data.loc[ANRtype]['Power in MWe'])\
                         , doc='Total capacity deployed (MWe)')
  model.pANRCAPEX = Param(initialize = float(ANR_data.loc[ANRtype]['CAPEX $/MWe']))
  model.pANRVOM = Param(initialize = float(ANR_data.loc[ANRtype]['VOM in $/MWh-e']))
  model.pANRFOM = Param(initialize = float(ANR_data.loc[ANRtype]['FOPEX $/MWe-y']))
  model.pANRRampRate = Param(initialize = float(ANR_data.loc[ANRtype]['Ramp Rate (fraction of capacity/hr)']))
  model.pANRMSL = Param(initialize = ANRnb*float(ANR_data.loc[ANRtype]['MSL in MWe']))
  @model.Param()
  def pANRCRF(model):
    return model.pWACC / (1 - (1/(1+model.pWACC)**float(ANR_data.loc[ANRtype,'Life (y)'])))

  electricity_prices = get_electricity_prices(industry_df=industry_df, id=id, year=year)
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


def solve_ED_electricity(industry, industry_df, id, ANR_data, year):
  print(f'Plant {id}, {industry} : start solving')
  model = build_ED_electricity(industry_df, id, ANR_data, year)

  solver = SolverFactory('glpk')
  #solver.options['timelimit'] = 240
  #solver.options['mip pool relgap'] = 0.02
  #solver.options['mip tolerances absmipgap'] = 1e-4
  #solver.options['mip tolerances mipgap'] = 5e-3
  results = solver.solve(model, tee=False)

  if results.solver.termination_condition == TerminationCondition.optimal: 
    model.solutions.load_from(results)
    print(f'Plant {id}, {industry} : solved')
  else:
    exit('Not solvable')
  
  results_dic = {}
  results_dic['Industry'] = industry
  results_dic['id'] = id 
  def compute_elec_sales(model):
    return sum(model.vG[t]*model.pEPrice[t] for t in model.t)/1e9
  results_dic['Electricity sales (bn$/year)'] = value(compute_elec_sales(model))
  def compute_avg_elec_price(model):
    return sum(model.pEPrice[t] for t in model.t)/8760
  results_dic['Avg price ($/MWhe)'] = value(compute_avg_elec_price(model))
  return results_dic


def save_electricity_results(industry_df, excel_file, industry):
  """Save electricity results
  Args: 
    industry_df (DataFrame): result of one industry with dedicated electricity production results 
    excel_file (str): path to excel file for electricity results
    industry (str): industrial sector
  Returns: 
    None 
  """
  try:
  # Load the existing Excel file
    with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
        # Check if the sheet exists
        if industry in xls.sheet_names:
            # If the sheet exists, replace the data
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                industry_df.to_excel(writer, sheet_name=industry)
        else:
            # If the sheet doesn't exist, create a new sheet
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                industry_df.to_excel(writer, sheet_name=industry)
  except FileNotFoundError:
      # If the file doesn't exist, create a new one and write the DataFrame to it
      industry_df.to_excel(excel_file, sheet_name=industry)

def plot_electricity_vs_h2_revenues(excel_file, year):
  ammonia = pd.read_excel(excel_file, sheet_name='ammonia')
  heat = pd.read_excel(excel_file, sheet_name='process_heat')
  refining = pd.read_excel(excel_file, sheet_name='refining')
  steel = pd.read_excel(excel_file, sheet_name='steel')
  total_elec = pd.concat([ammonia, heat, refining, steel], ignore_index=True)
  
  ax = sns.scatterplot(data=total_elec, x='Electricity sales (bn$/year)', y='Avoided fossil fuel cost (M$/year/MWe)', size='Avg price ($/MWhe)',palette='bright', hue='Industry', style='ANR type')
  med_x = np.arange(0,4,0.05)
  ax.plot(med_x, med_x, 'k--', linewidth=0.5)
  ax.spines[['right', 'top']].set_visible(False)
  ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
  #ax.set_xlim(-.1,3.1)
  #ax.set_ylim(-.1,3.1)
  ax.set_xlabel('Electricity sales (bn$/year)')
  #plt.savefig('./results/steel_be_state_with_carbon_prices.png')

  # this is an inset axes over the main axes
  sub_ax = plt.axes([.4, .2, .3, .3]) 
  sns.scatterplot(ax = sub_ax, data=total_elec, x='Electricity sales (bn$/year)', y='Avoided fossil fuel cost (M$/year/MWe)', palette='bright', style='ANR type', hue='Industry')
  sub_ax.plot(med_x, med_x, 'k--', linewidth=0.5)
  sub_ax.set_xlim(-.01, 0.2)
  sub_ax.set_ylim(-.01, 0.2)
  sub_ax.get_legend().set_visible(False)
  sub_ax.set_xlabel('')
  sub_ax.set_ylabel('')
  fig = ax.get_figure()
  fig.set_size_inches((8,5))
  fig.tight_layout()
  if learning:
    elec_save_path = './results/avoided_fossil_vs_elec_sales_with_learning_'+str(year)+'.png'
  else:
    elec_save_path = './results/avoided_fossil_vs_elec_sales_without_learning_'+str(year)+'.png'
  plt.savefig(elec_save_path)


def test():
  excel_file = './results/electricity_prod_results_no_learning.xlsx'
  plot_electricity_vs_h2_revenues(excel_file)

def main():
  # TODO also run with CAPEX after reduction from learning from industrial deployment

  ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name='FOAK', index_col=0)
  
  excel_file = './results/electricity_prod_results_no_learning_'+str(year)+'.xlsx'

  industry_dfs = []
  for industry in INDUSTRIES: 
    print(f'Industrial sector: {industry}')
    # Load results and compute avoided fossil fuel costs
    industry_df = load_industry_results(industry=industry)
    industry_df = compute_avoided_ff_costs(industry_df=industry_df, industry=industry)

    # Optimization for dedicated electricity production 
    ids = list(industry_df.index)

    # Sequential solving
    #results ={}
    #for id in ids:
     # results[id] = solve_ED_electricity(industry, industry_df, id, ANR_data)
    # elec_industry_df = pd.DataFrame(results).transpose()
    
    # Parallel solving
    with Pool(5) as pool:
      results = pool.starmap(solve_ED_electricity, [(industry, industry_df, id, ANR_data, year) for id in ids])
    pool.close()

    elec_industry_df = pd.DataFrame(results)
    elec_industry_df.set_index('id', inplace=True)

    industry_df = industry_df.merge(elec_industry_df,left_index=True, right_index=True)
    industry_dfs.append(industry_df)

    #Save results
    save_electricity_results(industry_df, excel_file, industry)

  # Plot results
  plot_electricity_vs_h2_revenues(excel_file, year)
  
    
    



if __name__ == '__main__':
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  #test()
  main()
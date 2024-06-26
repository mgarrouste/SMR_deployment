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

cambium_scenario = 'MidCase'#'MidCaseTCExpire' # 'LowRECostTCExpire','MidCaseTCExpire', 'MidCase', 'LowRECost', 'HighRECost', 'HighNGPrice', 'LowNGPrice'

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

  def compute_be_capex(model):
    return (sum((model.pEPrice[t]-model.pANRVOM)*model.vG[t] for t in model.t) -\
            model.pANRFOM*model.pANRCap )/(model.pANRCap*model.pANRCRF*(1-model.pITC_ANR))
  results_dic['BE CAPEX ($/MWe)'] = value(compute_be_capex(model))

  def compute_capacity_factor(model):
    return sum(model.vG[t] for t in model.t)/(model.pANRCap*8760)
  results_dic['Capacity factor'] = value(compute_capacity_factor(model))

  def compute_lcoe(model):
    """Computes the breakeven price of electricity assuming a 95% capacity factor"""
    return ((model.pANRCAPEX*model.pANRCRF*(1-model.pITC_ANR) + model.pANRFOM)*model.pANRCap \
            + model.pANRVOM*model.pANRCap*0.95*8760)/(sum(model.vG[t] for t in model.t))
  
  results_dic['LCOE ($/MWhe)'] = value(compute_lcoe(model))
    
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


def plot_results(anr_tag, boxplot=False):
  """ Plot Net Annual Revenues for each ANR design"""
  
  excel_file = f'./results/price_taker_{anr_tag}_{cambium_scenario}.xlsx'
  df = pd.read_excel(excel_file, header=0)
  df['Annual Net Revenues (M$/year/MWe)'] = df['Annual Net Revenues ($/year/MWe)']/1e6
  stats = df[['ANR type', 'Annual Net Revenues (M$/year/MWe)']].describe(percentiles=[.1,.25,.5,.75,.9])
  save_stats = f'./results/price_taker_{anr_tag}_{cambium_scenario}_stats.xlsx'
  print('Statistics: {}'.format(save_stats))
  stats.to_excel(save_stats)
  if boxplot:
    fig, ax = plt.subplots(figsize=(6,2))
    sns.boxplot(ax=ax, data=df, x='Annual Net Revenues (M$/year/MWe)', color='black', fill=False, width=0.5)
    sns.stripplot(ax=ax, data=df, x='Annual Net Revenues (M$/year/MWe)', palette=utils.palette, hue='ANR type',alpha=.6)
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1)
    ax.set_ylabel('Electricity')
  else:
    fig, ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(ax=ax, data=df, y='Annual Net Revenues (M$/year/MWe)', x='Avg price ($/MWhe)', palette=utils.palette, \
                hue='ANR type')
    ax.set_ylabel('Net Annual Revenues\n M$/MWe/y')
    ax.set_xlabel('Average state-level electricity price\n($/MWhe)')
  ax.get_legend().set_visible(False)
  sns.despine()
  #duplicate legend entries issue
  h3, l3 = ax.get_legend_handles_labels()
  by_label = dict(zip(l3, h3))
  fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,1.1),loc='upper center', ncol=3)
  fig.tight_layout()
  save_plot = f'./results/electricity_price_taker_net_annual_revenues_{anr_tag}_{cambium_scenario}.png'
  print(f'Plot: {save_plot}')
  fig.savefig(save_plot, bbox_inches='tight')



def compute_cost_reduction():
  for anr_tag in ['FOAK', 'NOAK']:
    ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name=anr_tag)
    excel_file = f'./results/price_taker_{anr_tag}_{cambium_scenario}.xlsx'
    df = pd.read_excel(excel_file, index_col=0)
    df = df.merge(ANR_data[['Reactor', 'CAPEX $/MWe']], left_on='ANR type', right_on='Reactor', how='left')
    df['Cost red CAPEX BE'] = df.apply(lambda x: max(0,1-(x['BE CAPEX ($/MWe)']/x['CAPEX $/MWe'])), axis=1)
    df.to_excel(excel_file)

def main():
  states = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', \
            'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', \
              'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WY']
  ANRtype_list = ['iPWR', 'HTGR', 'PBR-HTGR', 'iMSR', 'Micro']
  years = [2024]#, 2030, 2040]
  all_states_results_list = []
  for anr_tag in ['FOAK', 'NOAK']:
    ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name=anr_tag, index_col=0)
    excel_file = f'./results/price_taker_{anr_tag}_{cambium_scenario}.xlsx'
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


def compare_deployment_stages():
  noak_results = f'./results/price_taker_NOAK_{cambium_scenario}.xlsx'
  foak_results = f'./results/price_taker_FOAK_{cambium_scenario}.xlsx'
  assert os.path.isfile(noak_results), f'NOAK results not found: {noak_results}'
  assert os.path.isfile(foak_results), f'FOAK results not found: {foak_results}'
  noak_df = pd.read_excel(noak_results, usecols="B:H")
  foak_df = pd.read_excel(foak_results)
  # concatenate results
  noak_df['Deployment'] = 'NOAK'
  foak_df['Deployment'] = 'FOAK'
  total_df = pd.concat([foak_df, noak_df], ignore_index=True)
  # Formatting
  total_df['Annual Net Revenues (M$/MWe/year)'] = total_df['Annual Net Revenues ($/year/MWe)']/1e6
  # Plot
  fig = plt.figure(figsize=(5,4))
  topfig, botfig = fig.subfigures(1,2, width_ratios=[1,1.7])
  topax = topfig.subplots()
  sns.boxplot(ax=topax, data=total_df, y='Annual Net Revenues (M$/MWe/year)', x='Deployment', color='black', fill=False, width=.5)
  sns.stripplot(ax=topax, data=total_df, y='Annual Net Revenues (M$/MWe/year)', x='Deployment', hue='ANR type', palette=utils.palette, alpha=.6)

  # Scatterplot CAPEX against FOM, ANR hue, marker FOAK or BE
  botax = botfig.subplots()
  foak_df['capex (M$/MWe)'] = foak_df['BE CAPEX ($/MWe)']/1e6
  foak_df['Type'] = 'Breakeven'

  anr_foak_capex_fom = pd.read_excel('./ANRs.xlsx', sheet_name='FOAK')[['Reactor', 'CAPEX $/MWe', 'FOPEX $/MWe-y']]
  
  anr_foak_fom = anr_foak_capex_fom[['Reactor', 'FOPEX $/MWe-y']]
  foak_df = foak_df.merge(anr_foak_fom, left_on='ANR type', right_on='Reactor')

  anr_foak_capex_fom['capex (M$/MWe)'] = anr_foak_capex_fom['CAPEX $/MWe']/1e6
  anr_foak_capex_fom['Type'] = 'FOAK'

  anr_foak_capex_fom['FOM (M$/MWe-y)'] = anr_foak_capex_fom['FOPEX $/MWe-y']/1e6
  anr_foak_capex_fom = anr_foak_capex_fom[['Reactor', 'capex (M$/MWe)', 'FOM (M$/MWe-y)', 'Type']]
  foak_df['FOM (M$/MWe-y)'] = foak_df['FOPEX $/MWe-y']/1e6
  foak_df = foak_df[['Reactor', 'capex (M$/MWe)', 'FOM (M$/MWe-y)', 'Type']]
  df = pd.concat([anr_foak_capex_fom, foak_df])
  sns.scatterplot(ax=botax, data=df, x='FOM (M$/MWe-y)', y='capex (M$/MWe)', style='Type',\
                   hue='Reactor', palette=utils.palette)
  botax.set_ylabel('CAPEX (M$/MWe)')

  sns.despine()

  topax.get_legend().set_visible(False)
  botax.get_legend().set_visible(False)
  h4, l4 = botax.get_legend_handles_labels()
  by_label = dict(zip(l4, h4))
  fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,1.1),loc='upper center', ncol=3)
  

  """
  sns.boxplot(ax=botax, data=foak_df, x='BE CAPEX (M$/MWe)', color='black', fill=False, width=.5)
  sns.stripplot(ax=botax, data=foak_df, x='BE CAPEX (M$/MWe)', hue='ANR type', palette=utils.palette, alpha=.6)
  topax.set_ylabel('')
  botax.set_ylabel('')
  topax.get_legend().set_visible(False)
  botax.get_legend().set_visible(False)
  sns.despine()
  #duplicate legend entries issue
  h3, l3 = topax.get_legend_handles_labels()
  h4, l4 = botax.get_legend_handles_labels()
  by_label = dict(zip(l3+l4, h3+h4))
  fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,1.05),loc='upper center', ncol=5)
  # Add vertical lines with values of FOAK and NOAK CAPEX for the 5 designs
  anr_foak_capex = pd.read_excel('./ANRs.xlsx', sheet_name='FOAK', index_col='Reactor')[['CAPEX $/MWe']].to_dict()['CAPEX $/MWe']
  for reactor, capex in anr_foak_capex.items():
    add_vertical_line(botax, capex/1e6, ymin=0, ymax=1, color=utils.palette[reactor])
  """

  fig.savefig(f'./results/electricity_comparison_NOAK_FOAK_net_annual_revenues_{cambium_scenario}.png', bbox_inches='tight')


def add_vertical_line(ax, x, ymin, ymax, color):
  ax.axvline(x, ymin, ymax, color=color, linestyle='-', linewidth=1)


if __name__ == '__main__':
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--plot', required=False, help='Only plot results, does not run model, indicate FOAK or NOAK for corresponding net revenues plot')
  parser.add_argument('-c', '--compare', required=False, help='Compare via a plot FOAK and NOAK results')
  parser.add_argument('-b', '--breakeven', required=False, help='Compute cost reduction needed for breakeven')
  args = parser.parse_args()
  if args.compare:
    compare_deployment_stages()
  elif args.plot:
    if args.plot == 'FOAK':
      plot_results(anr_tag='FOAK')
    elif args.plot == 'NOAK':
      plot_results(anr_tag='NOAK')
    else:
      print('Specify FOAK or NOAK to print corresponding results')
      exit()
  elif args.breakeven:
    compute_cost_reduction()
  else:
    main()

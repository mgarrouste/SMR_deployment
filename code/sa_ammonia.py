from pyomo.environ import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import queue, threading
from utils import load_data
import utils
import multiprocessing.pool

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

WACC = utils.WACC

MaxANRMod = 40
NG_PRICE = 6.4 #$/MMBtu
ELEC_PRICE = 30 #$/MWh-e

auxNucNH3CAPEX = 64511550 #$/year
auxNucNH3LT = 20 # years

ngNH3ConsRate = 30.82# MMBtu/tNH3
ngNH3ElecCons = 0.061 # MWh/tNH3

def get_ammonia_plant_demand(plant):
  ammonia_df = pd.read_excel('./h2_demand_ammonia_us_2022.xlsx', sheet_name='processed')
  plant_df = ammonia_df[ammonia_df['plant_id'] == plant]
  h2_demand_kg_per_day = float(plant_df['H2 Dem. (kg/year)'].iloc[0])/365
  elec_demand_MWe = float(plant_df['Electricity demand (MWe)'].iloc[0])
  ammonia_capacity = float(plant_df['Capacity (tNH3/year)'].iloc[0])
  return ammonia_capacity, h2_demand_kg_per_day, elec_demand_MWe

def build_ammonia_plant_deployment(plant, ANR_data, H2_data): 
  print(f'Ammonia plant {plant} : start solving')
  model = ConcreteModel(plant)

  ############### DATA ####################
  ### Hydrogen and electricity demand
  ammonia_capacity, h2_dem_kg_per_day, elec_dem_MWe = get_ammonia_plant_demand(plant)
  model.pNH3Cap = Param(initialize = ammonia_capacity)
  model.pH2Dem = Param(initialize = h2_dem_kg_per_day) # kg/day
  model.pElecDem = Param(initialize = elec_dem_MWe) #MW-e

  ############### SETS ####################
  #### Sets ####
  model.N = Set(initialize=list(range(MaxANRMod)))
  model.H = Set(initialize=list(set([elem[0] for elem in H2_data.index])))
  model.G = Set(initialize=ANR_data.index)


  ############### VARIABLES ###############
  model.vS = Var(model.G, within=Binary, doc='Chosen ANR type')
  model.vM = Var(model.N, model.G, within=Binary, doc='Indicator of built ANR module')
  model.vQ = Var(model.N, model.H, model.G, within=NonNegativeIntegers, doc='Nb of H2 module of type H for an ANR module of type g')

  ############### PARAMETERS ##############
  model.pWACC = Param(initialize = WACC)

  ### Nuc NH3 ###
  model.pAuxNH3CAPEX = Param(initialize = auxNucNH3CAPEX)

  ### H2 ###
  data = H2_data.reset_index(level='ANR')[['H2Cap (kgh2/h)']]
  data.drop_duplicates(inplace=True)
  model.pH2CapH2 = Param(model.H, initialize = data.to_dict()['H2Cap (kgh2/h)'])

  @model.Param(model.H, model.G)
  def pH2CapElec(model, h, g):
    return float(H2_data.loc[h,g]['H2Cap (MWe)'])

  # Electric and heat consumption
  @model.Param(model.H, model.G)
  def pH2ElecCons(model, h, g):
    return float(H2_data.loc[h,g]['H2ElecCons (MWhe/kgh2)'])

  @model.Param(model.H, model.G)
  def pH2HeatCons(model, h, g):
    return float(H2_data.loc[h,g]['H2HeatCons (MWht/kgh2)'])

  data = H2_data.reset_index(level='ANR')[['VOM ($/MWhe)']]
  data.drop_duplicates(inplace=True)
  model.pH2VOM = Param(model.H, initialize = data.to_dict()['VOM ($/MWhe)'])

  data = H2_data.reset_index(level='ANR')[['FOM ($/MWe-year)']]
  data.drop_duplicates(inplace=True)
  model.pH2FC = Param(model.H, initialize = data.to_dict()['FOM ($/MWe-year)'])

  data = H2_data.reset_index(level='ANR')[['CAPEX ($/MWe)']]
  data.drop_duplicates(inplace=True)
  model.pH2CAPEX = Param(model.H, initialize = data.to_dict()['CAPEX ($/MWe)'])

  @model.Param(model.H)
  def pH2CRF(model, h):
    data = H2_data.reset_index(level='ANR')[['Life (y)']]
    data = data.groupby(level=0).mean()
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**float(data.loc[h,'Life (y)']) ) )
    return crf

  @model.Param(model.H, model.G)
  def pH2CarbonInt(model, h, g):
    return float(H2_data.loc[h,g]['Carbon intensity (kgCO2eq/kgH2)'])

  ### ANR ###
  # Capacity of ANRs MWt
  @model.Param(model.G)
  def pANRCap(model, g):
    return float(ANR_data.loc[g]['Power in MWe'])

  @model.Param(model.G)
  def pANRVOM(model, g):
    return float(ANR_data.loc[g]['VOM in $/MWh-e'])

  @model.Param(model.G)
  def pANRFC(model, g):
    return float(ANR_data.loc[g]['FOPEX $/MWe-y'])

  @model.Param(model.G)
  def pANRCAPEX(model, g):
    return float(ANR_data.loc[g]['CAPEX $/MWe'])

  @model.Param(model.G)
  def pANRCRF(model, g):
    return model.pWACC / (1 - (1/(1+model.pWACC)**float(ANR_data.loc[g,'Life (y)'])))
    
  @model.Param(model.G)
  def pANRThEff(model, g):
    return float(ANR_data.loc[g]['Power in MWe']/ANR_data.loc[g]['Power in MWt'])



  ############### OBJECTIVE ##############

  def annualized_costs_anr_h2(model):
    costs =  sum(sum(model.pANRCap[g]*model.vM[n,g]*((model.pANRCAPEX[g]*model.pANRCRF[g]+model.pANRFC[g])+model.pANRVOM[g]*365*24) \
      + sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2CAPEX[h]*model.pH2CRF[h]+model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 
    return costs
  
  def annualized_nuc_NH3_costs(model):
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**auxNucNH3LT) ) 
    costs =auxNucNH3CAPEX*crf
    return costs

  def annualized_net_rev(model):
    return -annualized_nuc_NH3_costs(model)-annualized_costs_anr_h2(model)
  model.NetRevenues = Objective(expr=annualized_net_rev, sense=maximize)  


  ############### CONSTRAINTS ############

  # Meet hydrogen demand from ammonia plant
  model.meet_h2_dem_ammonia_plant = Constraint(
    expr = model.pH2Dem <= sum( sum( sum(model.vQ[n,h,g]*model.pH2CapH2[h]*24 for g in model.G) for h in model.H) for n in model.N)
  )

  # Only one type of ANR deployed 
  model.max_ANR_type = Constraint(expr = sum(model.vS[g] for g in model.G) <= 1)

  # Only build ANR modules of the chosen type
  def match_ANR_type(model, n, g): 
    return model.vM[n,g] <= model.vS[g]
  model.match_ANR_type = Constraint(model.N, model.G, rule=match_ANR_type)

  # Heat and electricity balance at the ANR module level
  def energy_balance_module(model, n, g): 
    return sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) <= model.pANRCap[g]*model.vM[n,g]
  model.energy_balance_module = Constraint(model.N, model.G, rule = energy_balance_module)

  # Energy balance at the steel plant level: include auxiliary electricity demand
  def energy_balance_plant(model, g): 
    return sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for n in model.N) + (model.pElecDem)*model.vS[g] \
            <= sum(model.pANRCap[g]*model.vM[n,g] for n in model.N)
  model.energy_balance_plant = Constraint(model.G, rule = energy_balance_plant)

  
  return model


def solve_ammonia_plant_deployment(ANR_data, H2_data, plant):
  model = build_ammonia_plant_deployment(plant, ANR_data, H2_data)
  ammonia_capacity, h2_dem_kg_per_day, elec_dem_MWh_per_day = get_ammonia_plant_demand(plant)
  # for carbon accounting
  def compute_annual_carbon_emissions(model):
    return sum(sum(sum(model.pH2CarbonInt[h,g]*model.vQ[n,h,g]*model.pH2CapH2[h]*24*365 for g in model.G) for h in model.H) for n in model.N)
  
  def compute_anr_capex(model):
    return sum(sum(model.pANRCap[g]*model.vM[n,g]*model.pANRCAPEX[g]*model.pANRCRF[g]for g in model.G) for n in model.N) 
  
  def compute_anr_om(model):
    return sum(sum(model.pANRCap[g]*model.vM[n,g]*(model.pANRFC[g]+model.pANRVOM[g]*365*24) for g in model.G) for n in model.N) 
  
  def compute_h2_capex(model):
    return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*model.pH2CAPEX[h]*model.pH2CRF[h] for h in model.H) for g in model.G) for n in model.N) 
  
  def compute_h2_om(model):
    return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 

  def get_crf(model):
    return sum(model.vS[g]*model.pANRCRF[g] for g in model.G)
  
  def compute_conv_costs(model):
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**auxNucNH3LT) ) 
    costs =auxNucNH3CAPEX*crf
    return costs
  
  def get_deployed_cap(model):
    return sum(sum (model.vM[n,g]*model.pANRCap[g] for g in model.G) for n in model.N)

  ############## SOLVE ###################
  solver = SolverFactory('cplex')
  solver.options['timelimit'] = 240
  solver.options['mip pool relgap'] = 0.02
  solver.options['mip tolerances absmipgap'] = 1e-4
  solver.options['mip tolerances mipgap'] = 5e-3
  results = solver.solve(model, tee = False)

  results_dic = {}
  results_dic['plant_id'] = plant
  results_dic['Ammonia capacity (tNH3/year)'] = ammonia_capacity
  results_dic['H2 Dem (kg/day)'] = value(model.pH2Dem)
  results_dic['Aux Elec Dem (MWe)'] = value(model.pElecDem)
  results_dic['Net Rev. ($/year)'] = value(model.NetRevenues)
  for h in model.H:
    results_dic[h] = 0
  if results.solver.termination_condition == TerminationCondition.optimal: 
    model.solutions.load_from(results)
    results_dic['Ann. CO2 emissions (kgCO2eq/year)'] = value(compute_annual_carbon_emissions(model))
    results_dic['ANR CAPEX ($/year)'] = value(compute_anr_capex(model))
    results_dic['H2 CAPEX ($/year)'] = value(compute_h2_capex(model))
    results_dic['ANR O&M ($/year)'] = value(compute_anr_om(model))
    results_dic['H2 O&M ($/year)'] = value(compute_h2_om(model))
    results_dic['Conversion costs ($/year)'] = value(compute_conv_costs(model))
    results_dic['ANR CRF'] = value(get_crf(model))
    results_dic['Depl. ANR Cap. (MWe)'] = value(get_deployed_cap(model))
    
    for g in model.G: 
      if value(model.vS[g]) >=1: 
        results_dic['ANR type'] = g
        total_nb_modules = int(np.sum([value(model.vM[n,g]) for n in model.N]))
        results_dic['# ANR modules'] = total_nb_modules
        for n in model.N:
          if value(model.vM[n,g]) >=1:
            for h in model.H:
              if value(model.vQ[n,h,g]) > 0:
                results_dic[h] += value(model.vQ[n,h,g])
    results_dic['Breakeven NG price ($/MMBtu)'] = compute_ng_breakeven_price(results_dic)
    print(f'Ammonia plant {plant} solved')
    return results_dic
  else:
    print('Not feasible.')
    return None
  

def compute_ng_breakeven_price(results_ref):
  net_rev = results_ref['Net Rev. ($/year)']
  capacity = results_ref['Ammonia capacity (tNH3/year)']
  elec_costs = ELEC_PRICE*ngNH3ElecCons*capacity # $/year
  be_price = (-net_rev-elec_costs)/(ngNH3ConsRate*capacity)
  return be_price

def compute_capex_breakeven(results_ref, be_ng_price_foak, ng_price):
  alpha = results_ref['Ammonia capacity (tNH3/year)'][0]*ngNH3ConsRate
  foak_anr_capex = results_ref['ANR CAPEX ($/year)'][0]
  anr_crf = results_ref['ANR CRF'][0]
  deployed_anr_cap = results_ref['Depl. ANR Cap. (MWe)'][0]
  be_capex = (alpha*(ng_price - be_ng_price_foak) + foak_anr_capex)/(anr_crf*deployed_anr_cap)
  return be_capex


def main(learning_rate_anr_capex = 0, learning_rate_h2_capex =0, wacc=WACC): 
  # Go the present directory
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)


  # Load steel data
  ammonia_df = pd.read_excel('h2_demand_ammonia_us_2022.xlsx', sheet_name='processed')
  plant_ids = list(ammonia_df['plant_id'])

  # Load ANR and H2 parameters
  ANR_data, H2_data = load_data(learning_rate_anr_capex, learning_rate_h2_capex)

  # Build results dataset one by one
  breakeven_df = pd.DataFrame(columns=['plant_id', 'H2 Dem (kg/day)', 'Aux Elec Dem (MWe)','Alkaline', 'HTSE', 'PEM', 'ANR type', '# ANR modules',\
                                        'Breakeven NG price ($/MMBtu)', 'Ann. CO2 emissions (kgCO2eq/year)',\
                                            'ANR CAPEX ($/year)', 'H2 CAPEX ($/year)', 'ANR O&M ($/year)','H2 O&M ($/year)', 'Conversion costs ($/year)'])
  not_feasible = []
  
  with multiprocessing.pool.ThreadPool(3) as pool:
    results = pool.starmap(solve_ammonia_plant_deployment, [(ANR_data, H2_data, plant) for plant in plant_ids])
  pool.close()

  breakeven_df = pd.DataFrame(results)

  
  if len(not_feasible)>=1: 
    print('\n\n NOT FEASIBLE: \n')
    for plant in not_feasible: 
      print('Plant : ', plant)
      print('Demand NH3 ton per year : ', get_ammonia_plant_demand(plant)[0])
      print('Demand H2 kg per day: ', get_ammonia_plant_demand(plant)[1])
      print('Demand electricity MW: ', get_ammonia_plant_demand(plant)[2]/24)

  # Median Breakeven price
  med_be = breakeven_df['Breakeven NG price ($/MMBtu)'].median()
  return med_be



def sa_morris():
  problem = {
      'num_vars': 3,
      'names': ["LR ANR CAPEX", "LR H2 CAPEX", 'WACC'],
      'bounds': [[0.03,0.10], [0.03,0.10], [0.05, 0.1]]
  }
  param_values = morris_sample.sample(problem,N=100, optimal_trajectories=2)
  lr_anr_capex_list = param_values.T[0]
  lr_h2_capex_list = param_values.T[1]
  wacc_list = param_values.T[2]
  # Issue top level tasks: optimal deployment for each tuple of the SA parameters
  with multiprocessing.pool.Pool(3) as pool:
    Y = pool.starmap(main, [(lr_anr_capex, lr_h2_capex, wacc) for lr_anr_capex, lr_h2_capex, wacc in zip(lr_anr_capex_list, lr_h2_capex_list, wacc_list )])
  Y = np.array(Y)
  morris_indices = morris_analyze.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=True, scaled=True)
  morris_indices.to_csv('./results/sa_ammonia_morris.csv')



def sa_sobol():
  # Set up the Sobol analysis
  problem = {
      'num_vars': 3,
      'names': ["LR ANR CAPEX", "LR H2 CAPEX", 'WACC'],
      'bounds': [[0.03,0.10], [0.03,0.10], [0.05, 0.1]]
  }

  param_values = sobol_sample.sample(problem,1)
  lr_anr_capex_list = param_values.T[0]
  lr_h2_capex_list = param_values.T[1]
  wacc_list = param_values.T[2]




  # Issue top level tasks: optimal deployment for each tuple of the SA parameters
  with multiprocessing.pool.Pool(3) as pool:
    Y = pool.starmap(main, [(lr_anr_capex, lr_h2_capex, wacc) for lr_anr_capex, lr_h2_capex, wacc in zip(lr_anr_capex_list, lr_h2_capex_list, wacc_list )])

    #results = pool.starmap(solve_ammonia_plant_deployment, [(ANR_data, H2_data, plant) for plant in plant_ids])

    #mean_be_ng = main(learning_rate_anr_capex = lr_anr_capex, learning_rate_h2_capex =lr_h2_capex, wacc=wacc)
    #Y[i] = mean_be_ng
  Y = np.array(Y)
  sobol_indices = sobol_analyze.analyze(problem, Y)
  total_Si, first_Si, second_Si = sobol_indices.to_df()

  si_df = total_Si.merge(first_Si, left_index=True, right_index=True)
  si_df.to_csv('./results/sa_ammonia_sobol.csv')
  print(si_df)


  fig, ax = plt.subplots()



  si_df[['ST', 'S1']].plot.bar(yerr = si_df[['ST_conf', 'S1_conf']].apply(np.abs), ax=ax, capsize=5, rot=0)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.show()


if __name__ == '__main__': 
  sa_morris()

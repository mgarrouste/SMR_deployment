from pyomo.environ import *
import pandas as pd
import numpy as np
import os
from v_2_1_refining import load_data, compute_breakeven_price

""" Version 0"""

MaxANRMod = 20
NAT_GAS_PRICE = 6.45 #$/MMBTU
CONV_MJ_TO_MMBTU = 1/1055.05585 #MMBTU/MJ
COAL_CONS_RATE = 0.663 #ton-coal/ton-steel for conventional BF/BOF plant

def get_steel_plant_demand(plant):
  steel_df = pd.read_excel('./h2_demand_bfbof_steel_us_2022.xlsx', sheet_name='processed')
  plant_df = steel_df[steel_df['Plant'] == plant]
  h2_demand_kg_per_day = float(plant_df['Hydrogen demand (kg/day)'].iloc[0])
  elec_demand_MWe = float(plant_df['Electricity demand (MWe)'].iloc[0])
  steel_cap_ton_per_annum = float(plant_df['Steel production capacity (ttpa)'].iloc[0]*1000)
  return steel_cap_ton_per_annum, h2_demand_kg_per_day, elec_demand_MWe

def build_steel_plant_deployment(plant, ANR_data, H2_data): 
  
  model = ConcreteModel(plant)

  ############### DATA ####################
  ### Hydrogen and electricity demand
  steel_cap_ton_per_annum, h2_dem_kg_per_day, elec_dem_MWe = get_steel_plant_demand(plant)
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
  model.pWACC = Param(initialize = 0.08)

  ### Steel ###
  # Carbon emissions from DRI process at 95% H2 concentration
  model.pDRICO2Intensity = 40 # kgCO2/ton-DRI

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

  print('Parameters established')


  ############### OBJECTIVE ##############

  def annualized_costs(model):
    costs =  sum(sum(model.pANRCap[g]*model.vM[n,g]*((model.pANRCAPEX[g]*model.pANRCRF[g]+model.pANRFC[g])+model.pANRVOM[g]*365*24) \
      + sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2CAPEX[h]*model.pH2CRF[h]+model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 
    return costs

  def annualized_net_rev(model):
    return -annualized_costs(model)
  model.NetRevenues = Objective(expr=annualized_net_rev, sense=maximize)  


  ############### CONSTRAINTS ############

  # Meet hydrogen demand from steel plant 
  model.meet_h2_dem_steel_plant = Constraint(
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
    return sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for n in model.N) + model.pElecDem*model.vS[g] \
            <= sum(model.pANRCap[g]*model.vM[n,g] for n in model.N)
  model.energy_balance_plant = Constraint(model.G, rule = energy_balance_plant)

  
  return model


def solve_steel_plant_deployment(model, plant):
  steel_cap_ton_per_annum, h2_dem_kg_per_day, elec_dem_MWh_per_day = get_steel_plant_demand(plant)
  # for carbon accounting
  def compute_annual_carbon_emissions(model):
    return sum(sum(sum(model.pH2CarbonInt[h,g]*model.vQ[n,h,g]*model.pH2CapH2[h]*24*365 for g in model.G) for h in model.H) for n in model.N)+\
          model.pDRICO2Intensity*steel_cap_ton_per_annum

  ############## SOLVE ###################
  solver = SolverFactory('cplex')
  solver.options['timelimit'] = 240
  solver.options['mip pool relgap'] = 0.02
  solver.options['mip tolerances absmipgap'] = 1e-4
  solver.options['mip tolerances mipgap'] = 5e-3
  results = solver.solve(model, tee = True)

  results_dic = {}
  results_dic['Plant'] = [plant]
  results_dic['Steel prod. (ton/year)'] = [steel_cap_ton_per_annum]
  results_dic['H2 Dem (kg/day)'] = [value(model.pH2Dem)]
  results_dic['Aux Elec Dem (MWe)'] = [value(model.pElecDem)]
  results_dic['Cost ($/year)'] = [value(model.NetRevenues)]
  results_dic['Ann. CO2 emissions (kgCO2eq/year)'] = [value(compute_annual_carbon_emissions(model))]
  for h in model.H:
    results_dic[h] = [0]
  if results.solver.termination_condition == TerminationCondition.optimal: 
    model.solutions.load_from(results)
    print('\n\n\n\n',' ------------ SOLUTION  -------------')
    for g in model.G: 
      if value(model.vS[g]) >=1: 
        print('Chosen type of advanced nuclear reactor is ',g)
        results_dic['ANR type'] = [g]
        total_nb_modules = int(np.sum([value(model.vM[n,g]) for n in model.N]))
        print(total_nb_modules, ' modules are needed.')
        results_dic['# ANR modules'] = [total_nb_modules]
        for n in model.N:
          if value(model.vM[n,g]) >=1:
            print('ANR Module # ',int(value(model.vM[n,g])), 'of type ', g)
            for h in model.H:
              print(int(value(model.vQ[n,h,g])), ' Hydrogen production modules of type:',h )
              if value(model.vQ[n,h,g]) > 0:
                results_dic[h][0] += value(model.vQ[n,h,g])
    return results_dic
  else:
    print('Not feasible.')
    return None

def compute_breakeven_price(results_ref):
  revenues = results_ref['Cost ($/year)'][0]
  breakeven_price = -revenues/(COAL_CONS_RATE* results_ref['Steel prod. (ton/year)'][0])
  return breakeven_price

def main(): 
  # Go the present directory
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)

  # Load steel data
  steel_df = pd.read_excel('h2_demand_bfbof_steel_us_2022.xlsx', sheet_name='processed')
  steel_ids = list(steel_df['Plant'])

  # Load ANR and H2 parameters
  ANR_data, H2_data = load_data()

  # Build results dataset one by one
  breakeven_df = pd.DataFrame(columns=['Plant', 'H2 Dem (kg/day)', 'Aux Elec Dem (MWe)','Alkaline', 'HTSE', 'PEM', 'ANR type', '# ANR modules',\
                                        'Breakeven coal price ($/ton)', 'Ann. CO2 emissions (kgCO2eq/year)'])
  not_feasible = []
  for plant in steel_ids:
    try: 
      model = build_steel_plant_deployment(plant, ANR_data, H2_data)
      result_plant = solve_steel_plant_deployment(model, plant)
      result_plant['Breakeven price coal ($/ton)'] = [compute_breakeven_price(result_plant)]
      breakeven_df = pd.concat([breakeven_df, pd.DataFrame.from_dict(data=result_plant)])
    except ValueError: 
      not_feasible.append(plant)
  

  # Sort results by h2 demand 
  breakeven_df.sort_values(by=['H2 Dem (kg/day)'], inplace=True)
  breakeven_df.to_csv('./results/breakeven_prices_steel.csv', header = True, index=False)

  if len(not_feasible)>=1: 
    print('\n\n NOT FEASIBLE: \n')
    for plant in not_feasible: 
      print('Plant : ', plant)
      print('Demand Steel ton per year : ', get_steel_plant_demand(plant)[0])
      print('Demand H2 kg per day: ', get_steel_plant_demand(plant)[1])
      print('Demand electricity MW: ', get_steel_plant_demand(plant)[2]/24)


def test():
  plant = 'U.S. Steel Granite City Works'
  print('Plant : ', plant)
  print('Demand Steel ton per year : ', get_steel_plant_demand(plant)[0])
  print('Demand H2 kg per day: ', get_steel_plant_demand(plant)[1])
  print('Demand electricity MW: ', get_steel_plant_demand(plant)[2]/24)


if __name__ == '__main__': 
  main()
  #test()









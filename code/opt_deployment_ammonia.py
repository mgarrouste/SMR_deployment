from pyomo.environ import *
import pandas as pd
import numpy as np
import os
from utils import load_data
import utils
from multiprocessing import Pool

"""
This script takes in the ammonia plant data from 2022 and optimizes the deployment of SMRs and electrolysis modules at each site to serve
their hydrogen demand
"""

WACC = utils.WACC
ITC_SMR = utils.ITC_SMR
ITC_H2 = utils.ITC_H2

MaxSMRMod = 40
ELEC_PRICE = 30 #$/MWh-e

auxNucNH3CAPEX = 64511550 #$
auxNucNH3LT = 20 # years

ngNH3ConsRate = 30.82# MMBtu/tNH3
ngNH3ElecCons = 0.061 # MWh/tNH3

def get_ammonia_plant_demand(plant):
  ammonia_df = pd.read_excel('./h2_demand_ammonia_us_2022.xlsx', sheet_name='processed')
  plant_df = ammonia_df[ammonia_df['id'] == plant]
  h2_demand_kg_per_day = float(plant_df['H2 Dem. (kg/year)'].iloc[0])/365
  elec_demand_MWe = float(plant_df['Electricity demand (MWe)'].iloc[0])
  ammonia_capacity = float(plant_df['Capacity (tNH3/year)'].iloc[0])
  state = plant_df['State'].iloc[0].strip()
  lat = plant_df['latitude'].iloc[0]
  lon = plant_df['longitude'].iloc[0]
  return ammonia_capacity, h2_demand_kg_per_day, elec_demand_MWe, state, lat, lon

def build_ammonia_plant_deployment(plant, SMR_data, H2_data): 
  print(f'Ammonia plant {plant} : start solving')
  model = ConcreteModel(plant)

  ############### DATA ####################
  ### Hydrogen and electricity demand
  ammonia_capacity, h2_dem_kg_per_day, elec_dem_MWe, state, lat, lon = get_ammonia_plant_demand(plant)
  model.pNH3Cap = Param(initialize = ammonia_capacity)
  model.pH2Dem = Param(initialize = h2_dem_kg_per_day) # kg/day
  model.pElecDem = Param(initialize = elec_dem_MWe) #MW-e
  model.pState = Param(initialize = state.strip(), within=Any)

  ############### SETS ####################
  #### Sets ####
  model.N = Set(initialize=list(range(MaxSMRMod)))
  model.H = Set(initialize=list(set([elem[0] for elem in H2_data.index])))
  model.G = Set(initialize=SMR_data.index)


  ############### VARIABLES ###############
  model.vS = Var(model.G, within=Binary, doc='Chosen SMR type')
  model.vM = Var(model.N, model.G, within=Binary, doc='Indicator of built SMR module')
  model.vQ = Var(model.N, model.H, model.G, within=NonNegativeIntegers, doc='Nb of H2 module of type H for an SMR module of type g')

  ############### PARAMETERS ##############
  # Financial
  model.pWACC = Param(initialize = WACC)
  model.pITC_H2 = Param(initialize = ITC_H2)
  model.pITC_SMR = Param(initialize = ITC_SMR)

  ### Nuc NH3 ###
  model.pAuxNH3CAPEX = Param(initialize = auxNucNH3CAPEX)

  ### H2 ###
  data = H2_data.reset_index(level='SMR')[['H2Cap (kgh2/h)']]
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

  data = H2_data.reset_index(level='SMR')[['VOM ($/MWhe)']]
  data.drop_duplicates(inplace=True)
  model.pH2VOM = Param(model.H, initialize = data.to_dict()['VOM ($/MWhe)'])

  data = H2_data.reset_index(level='SMR')[['FOM ($/MWe-year)']]
  data.drop_duplicates(inplace=True)
  model.pH2FC = Param(model.H, initialize = data.to_dict()['FOM ($/MWe-year)'])

  data = H2_data.reset_index(level='SMR')[['CAPEX ($/MWe)']]
  data.drop_duplicates(inplace=True)
  model.pH2CAPEX = Param(model.H, initialize = data.to_dict()['CAPEX ($/MWe)'])

  @model.Param(model.H)
  def pH2CRF(model, h):
    data = H2_data.reset_index(level='SMR')[['Life (y)']]
    data = data.groupby(level=0).mean()
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**float(data.loc[h,'Life (y)']) ) )
    return crf

  @model.Param(model.H, model.G)
  def pH2CarbonInt(model, h, g):
    return float(H2_data.loc[h,g]['Carbon intensity (kgCO2eq/kgH2)'])

  ### SMR ###
  # Capacity of SMRs MWt
  @model.Param(model.G)
  def pSMRCap(model, g):
    return float(SMR_data.loc[g]['Power in MWe'])

  @model.Param(model.G)
  def pSMRVOM(model, g):
    return float(SMR_data.loc[g]['VOM in $/MWh-e'])

  @model.Param(model.G)
  def pSMRFC(model, g):
    return float(SMR_data.loc[g]['FOPEX $/MWe-y'])

  @model.Param(model.G)
  def pSMRCAPEX(model, g):
    return float(SMR_data.loc[g]['CAPEX $/MWe'])

  @model.Param(model.G)
  def pSMRCRF(model, g):
    return model.pWACC / (1 - (1/(1+model.pWACC)**float(SMR_data.loc[g,'Life (y)'])))
    
  @model.Param(model.G)
  def pSMRThEff(model, g):
    return float(SMR_data.loc[g]['Power in MWe']/SMR_data.loc[g]['Power in MWt'])



  ############### OBJECTIVE ##############

  def annualized_costs_SMR_h2(model):
    costs =  sum(sum(model.pSMRCap[g]*model.vM[n,g]*((model.pSMRCAPEX[g]*(1-model.pITC_SMR)*model.pSMRCRF[g]+model.pSMRFC[g])+model.pSMRVOM[g]*365*24) \
      + sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2CAPEX[h]*(1-model.pITC_H2)*model.pH2CRF[h]+model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 
    return costs
  
  def annualized_nuc_NH3_costs(model):
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**auxNucNH3LT) ) 
    costs =auxNucNH3CAPEX*crf*(1-model.pITC_H2)
    return costs

  def annualized_net_rev(model):
    return -annualized_nuc_NH3_costs(model)-annualized_costs_SMR_h2(model)
  model.NetRevenues = Objective(expr=annualized_net_rev, sense=maximize)  


  ############### CONSTRAINTS ############

  # Meet hydrogen demand from ammonia plant
  model.meet_h2_dem_ammonia_plant = Constraint(
    expr = model.pH2Dem <= sum( sum( sum(model.vQ[n,h,g]*model.pH2CapH2[h]*24 for g in model.G) for h in model.H) for n in model.N)
  )

  # Only one type of SMR deployed 
  model.max_SMR_type = Constraint(expr = sum(model.vS[g] for g in model.G) <= 1)

  # Only build SMR modules of the chosen type
  def match_SMR_type(model, n, g): 
    return model.vM[n,g] <= model.vS[g]
  model.match_SMR_type = Constraint(model.N, model.G, rule=match_SMR_type)

  # Heat and electricity balance at the SMR module level
  def energy_balance_module(model, n, g): 
    return sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) <= model.pSMRCap[g]*model.vM[n,g]
  model.energy_balance_module = Constraint(model.N, model.G, rule = energy_balance_module)

  # Energy balance at the steel plant level: include auxiliary electricity demand
  def energy_balance_plant(model, g): 
    return sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for n in model.N) + (model.pElecDem)*model.vS[g] \
            <= sum(model.pSMRCap[g]*model.vM[n,g] for n in model.N)
  model.energy_balance_plant = Constraint(model.G, rule = energy_balance_plant)

  
  return model


def solve_ammonia_plant_deployment(SMR_data, H2_data, plant, print_results):
  model = build_ammonia_plant_deployment(plant, SMR_data, H2_data)
  ammonia_capacity, h2_dem_kg_per_day, elec_dem_MWh_per_day, state, lat, lon = get_ammonia_plant_demand(plant)
  # for carbon accounting
  def compute_annual_carbon_emissions(model):
    return sum(sum(sum(model.pH2CarbonInt[h,g]*model.vQ[n,h,g]*model.pH2CapH2[h]*24*365 for g in model.G) for h in model.H) for n in model.N)
  
  def compute_SMR_capex(model):
    return sum(sum(model.pSMRCap[g]*model.vM[n,g]*model.pSMRCAPEX[g]*(1-model.pITC_SMR)*model.pSMRCRF[g]for g in model.G) for n in model.N) 
  
  def compute_SMR_om(model):
    return sum(sum(model.pSMRCap[g]*model.vM[n,g]*(model.pSMRFC[g]+model.pSMRVOM[g]*365*24) for g in model.G) for n in model.N) 
  
  def compute_h2_capex(model):
    return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*model.pH2CAPEX[h]*(1-model.pITC_H2)*model.pH2CRF[h] for h in model.H) for g in model.G) for n in model.N) 
  
  def compute_h2_om(model):
    return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 

  def get_crf(model):
    return sum(model.vS[g]*model.pSMRCRF[g] for g in model.G)
  
  def get_SMR_capex(model):
    return sum(model.pSMRCAPEX[g]*model.vS[g] for g in model.G)
  
  def compute_conv_costs(model):
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**auxNucNH3LT) ) 
    costs =auxNucNH3CAPEX*crf*(1-model.pITC_H2)
    return costs
  
  def get_deployed_cap(model):
    return sum(sum (model.vM[n,g]*model.pSMRCap[g] for g in model.G) for n in model.N)
  
  def annualized_avoided_ng_costs(model):
    ng_price = utils.get_ng_price_aeo(model.pState)
    avoided_costs = utils.nh3_nrj_intensity*model.pNH3Cap*ng_price 
    return avoided_costs
  
  def get_eq_elec_dem_h2(model): return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for g in model.G) for n in model.N) 
  def compute_surplus_capacity(model):
    deployed_capacity = sum(sum (model.vM[n,g]*model.pSMRCap[g] for g in model.G) for n in model.N)
    aux_elec_demand = elec_dem_MWh_per_day/24
    h2_elec_demand  = sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for g in model.G) for n in model.N) 
    return deployed_capacity - aux_elec_demand - h2_elec_demand

  def compute_initial_investment(model):
    SMR_capex = sum(sum(model.pSMRCap[g]*model.vM[n,g]*model.pSMRCAPEX[g]*(1-model.pITC_SMR) for g in model.G) for n in model.N)
    h2_capex = sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2CAPEX[h]*(1-model.pITC_H2)) for h in model.H) for g in model.G) for n in model.N)
    conversion_costs = auxNucNH3CAPEX*(1-model.pITC_H2)
    Co = SMR_capex + h2_capex +conversion_costs 
    return Co

  ############## SOLVE ###################
  solver = SolverFactory('cplex_direct')
  solver.options['timelimit'] = 240
  solver.options['mip_pool_relgap'] = 0.02
  solver.options['mip_tolerances_absmipgap'] = 1e-4
  solver.options['mip_tolerances_mipgap'] = 5e-3
  results = solver.solve(model, tee = print_results)

  results_ref = {}
  results_ref['id'] = plant
  results_ref['state'] = value(model.pState)
  results_ref['State price ($/MMBtu)'] = utils.get_ng_price_aeo(results_ref['state'])
  results_ref['latitude'] = lat
  results_ref['longitude'] = lon
  results_ref['Ammonia capacity (tNH3/year)'] = ammonia_capacity
  results_ref['H2 Dem. (kg/day)'] = h2_dem_kg_per_day
  results_ref['SMR CAPEX ($/MWe)'] = value(get_SMR_capex(model))
  results_ref['Aux Elec Dem. (MWe)'] = elec_dem_MWh_per_day/24
  results_ref['Net Revenues ($/year)'] = value(model.NetRevenues)
  results_ref['H2 PTC Revenues ($/year)'] = h2_dem_kg_per_day*365*utils.h2_ptc
  results_ref['Net Revenues with H2 PTC ($/year)'] = results_ref['Net Revenues ($/year)']+results_ref['H2 PTC Revenues ($/year)']
  for h in model.H:
    results_ref[h] = 0
  if results.solver.termination_condition == TerminationCondition.optimal: 
    model.solutions.load_from(results)
    results_ref['Ann. CO2 emissions (kgCO2eq/year)'] = value(compute_annual_carbon_emissions(model))
    results_ref['Initial investment ($)'] = value(compute_initial_investment(model))
    results_ref['SMR CAPEX ($/year)'] = value(compute_SMR_capex(model))
    results_ref['SMR CRF'] = value(get_crf(model))
    results_ref['Depl. SMR Cap. (MWe)'] = value(get_deployed_cap(model))
    results_ref['Depl H2 Cap. (MWe)'] = value(get_eq_elec_dem_h2(model))
    results_ref['H2 CAPEX ($/year)'] = value(compute_h2_capex(model))
    results_ref['SMR O&M ($/year)'] = value(compute_SMR_om(model))
    results_ref['H2 O&M ($/year)'] = value(compute_h2_om(model))
    results_ref['Conversion costs ($/year)'] = value(compute_conv_costs(model))
    results_ref['Avoided NG costs ($/year)'] = value(annualized_avoided_ng_costs(model))
    results_ref['Breakeven price ($/MMBtu)'] = compute_ng_breakeven_price(results_ref) # Compute BE price before adding avoided ng costs!
    results_ref['BE wo PTC ($/MMBtu)'] = compute_ng_be_without_ptc(results_ref)
    results_ref['Net Revenues ($/year)'] +=results_ref['Avoided NG costs ($/year)']
    # Recalculate revenues with H2 PTC: add revenues from avoided NG costs
    results_ref['Net Revenues with H2 PTC ($/year)'] = results_ref['Net Revenues ($/year)']+results_ref['H2 PTC Revenues ($/year)']
    results_ref['Surplus SMR Cap. (MWe)'] = value(compute_surplus_capacity(model))
    results_ref['Net Annual Revenues ($/MWe/y)'] = (results_ref['Net Revenues ($/year)'])/results_ref['Depl. SMR Cap. (MWe)']
    results_ref['Net Annual Revenues with H2 PTC ($/MWe/y)'] = results_ref['Net Revenues with H2 PTC ($/year)']/results_ref['Depl. SMR Cap. (MWe)']
    for g in model.G: 
      if value(model.vS[g]) >=1: 
        results_ref['SMR type'] = g
        total_nb_modules = int(np.sum([value(model.vM[n,g]) for n in model.N]))
        results_ref['# SMR modules'] = total_nb_modules
        for n in model.N:
          if value(model.vM[n,g]) >=1:
            for h in model.H:
              if value(model.vQ[n,h,g]) > 0:
                results_ref[h] += value(model.vQ[n,h,g])
    print(f'Ammonia plant {plant} solved')
    return results_ref
  else:
    print(f'{plant} not feasible.')
    empty_dic = {}
    return empty_dic
  

def compute_ng_breakeven_price(results_ref):
  net_rev = results_ref['Net Revenues with H2 PTC ($/year)']
  capacity = results_ref['Ammonia capacity (tNH3/year)']
  elec_costs = ELEC_PRICE*ngNH3ElecCons*capacity # $/year
  be_price = (-net_rev-elec_costs)/(ngNH3ConsRate*capacity)
  return be_price

def compute_ng_be_without_ptc(results_ref):
  net_rev = results_ref['Net Revenues ($/year)']
  capacity = results_ref['Ammonia capacity (tNH3/year)']
  elec_costs = ELEC_PRICE*ngNH3ElecCons*capacity # $/year
  be_price = (-net_rev-elec_costs)/(ngNH3ConsRate*capacity)
  return be_price



def main(SMR_tag='FOAK', wacc=WACC, print_main_results=True, print_results=False): 
  # Go the present directory
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)

  # Load steel data
  ammonia_df = pd.read_excel('h2_demand_ammonia_us_2022.xlsx', sheet_name='processed')
  plant_ids = list(ammonia_df['id'])

  # Load SMR and H2 parameters
  SMR_data, H2_data = load_data(SMR_tag=SMR_tag)

  # Build results dataset one by one
  
  with Pool(10) as pool:
    results = pool.starmap(solve_ammonia_plant_deployment, [(SMR_data, H2_data, plant, print_results) for plant in plant_ids])
  pool.close()

  df = pd.DataFrame(results)

  excel_file = f'./results/raw_results_SMR_{SMR_tag}_h2_wacc_{str(wacc)}.xlsx'
  sheet_name = 'ammonia'
  if print_main_results:
    # Try to read the existing Excel file
    
    try:
    # Load the existing Excel file
      with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
          # Check if the sheet exists
          if sheet_name in xls.sheet_names:
              # If the sheet exists, replace the data
              with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                  df.to_excel(writer, sheet_name=sheet_name, index=False)
          else:
              # If the sheet doesn't exist, create a new sheet
              with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                  df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and write the DataFrame to it
        df.to_excel(excel_file, sheet_name=sheet_name, index=False)
    #df.sort_values(by=['Breakeven NG price ($/MMBtu)'], inplace=True)
    #csv_path = './results/ammonia_SMR_lr_'+str(learning_rate_SMR_capex)+'_h2_lr_'+str(learning_rate_h2_capex)+'_wacc_'+str(wacc)+'.csv'
    #df.to_csv(csv_path, header = True, index=False)

  # Median Breakeven price
  med_be = df['Breakeven price ($/MMBtu)'].median()
  return med_be


if __name__ == '__main__': 
  main(SMR_tag=utils.LEARNING)
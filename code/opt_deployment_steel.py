from pyomo.environ import *
import pandas as pd
import numpy as np
import os
import utils
from multiprocessing import Pool

""" Version 0"""

MaxANRMod = 20
NAT_GAS_PRICE = 6.45 #$/MMBTU
CONV_MJ_TO_MMBTU = 1/1055.05585 #MMBTU/MJ
COAL_CONS_RATE = 0.663 #ton-coal/ton-steel for conventional BF/BOF plant

iron_ore_cost = 100 #$/t_ironore
bfbof_iron_cons = 1.226 #t_ironore/t_steel
om_bfbof = 178.12 #$/t_steel

WACC = utils.WACC


def get_steel_plant_demand(plant):
  steel_df = pd.read_excel('./h2_demand_bfbof_steel_us_2022.xlsx', sheet_name='processed')
  plant_df = steel_df[steel_df['Plant'] == plant]
  h2_demand_kg_per_day = float(plant_df['Hydrogen demand (kg/day)'].iloc[0])
  elec_demand_MWe = float(plant_df['Electricity demand (MWe)'].iloc[0])
  steel_cap_ton_per_annum = float(plant_df['Steel production capacity (ttpa)'].iloc[0]*1000)
  return steel_cap_ton_per_annum, h2_demand_kg_per_day, elec_demand_MWe


def get_state(plant):
  steel_df = pd.read_excel('./h2_demand_bfbof_steel_us_2022.xlsx', sheet_name='processed')
  plant_df = steel_df[steel_df['Plant'] == plant]
  state = plant_df['STATE'].iloc[0]
  return state

def build_steel_plant_deployment(plant, ANR_data, H2_data): 
  print(f'Start {plant}')
  model = ConcreteModel(plant)

  ############### DATA ####################
  ### Hydrogen and electricity demand
  steel_cap_ton_per_annum, h2_dem_kg_per_day, elec_dem_MWe = get_steel_plant_demand(plant)
  model.pH2Dem = Param(initialize = h2_dem_kg_per_day) # kg/day
  model.pElecDem = Param(initialize = elec_dem_MWe) #MW-e
  state = get_state(plant)
  model.pState = Param(initialize = state, within=Any)

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
  # Financial 
  model.pWACC = Param(initialize = WACC)
  model.pITC_ANR = Param(initialize = utils.ITC_ANR)
  model.pITC_H2 = Param(initialize = utils.ITC_H2)

  ### Steel ###
  # Carbon emissions from DRI process at 95% H2 concentration
  model.pDRICO2Intensity = 40 # kgCO2/ton-DRI`````````````````
  model.pShaftFCAPEX = 250 # $/tDRI/year
  model.pEAFCAPEX = 160 # $/tsteel/year
  model.pEAFOM  = 24.89 # $/tsteel (EAF and casting)
  model.pIronOre = iron_ore_cost # $/tironore
  model.pRatioSteelDRI = 0.9311 # tsteel/tDRI
  model.pRatioIronOreDRI = 1.391 # tironore/tDRI


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
    costs =  sum(sum(model.pANRCap[g]*model.vM[n,g]*((model.pANRCAPEX[g]*(1-model.pITC_ANR)*model.pANRCRF[g]+model.pANRFC[g])+model.pANRVOM[g]*365*24) \
      + sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2CAPEX[h]*(1-model.pITC_H2)*model.pH2CRF[h]+model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 
    return costs
  
  def annualized_costs_dri_eaf(model):
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**20) )# assumes 20 years lifetime for shaft and eaf
    costs = steel_cap_ton_per_annum*(model.pEAFCAPEX*crf*(1-model.pITC_H2) + model.pShaftFCAPEX*(1-model.pITC_H2)*crf/model.pRatioSteelDRI + model.pEAFOM +\
            model.pIronOre*model.pRatioIronOreDRI/model.pRatioSteelDRI)
    return costs
  
  

  def annualized_net_rev(model):
    return -annualized_costs_anr_h2(model)-annualized_costs_dri_eaf(model)
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
    return sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for n in model.N) + (model.pElecDem)*model.vS[g] \
            <= sum(model.pANRCap[g]*model.vM[n,g] for n in model.N)
  model.energy_balance_plant = Constraint(model.G, rule = energy_balance_plant)

  
  return model


def solve_steel_plant_deployment(plant, ANR_data, H2_data):
  steel_cap_ton_per_annum, h2_dem_kg_per_day, elec_dem_MWh_per_day = get_steel_plant_demand(plant)

  model = build_steel_plant_deployment(plant, ANR_data, H2_data)
  # for carbon accounting
  def compute_annual_carbon_emissions(model):
    return sum(sum(sum(model.pH2CarbonInt[h,g]*model.vQ[n,h,g]*model.pH2CapH2[h]*24*365 for g in model.G) for h in model.H) for n in model.N)+\
          model.pDRICO2Intensity*steel_cap_ton_per_annum
  
  def compute_anr_capex(model):
    return sum(sum(model.pANRCap[g]*model.vM[n,g]*model.pANRCAPEX[g]*(1-model.pITC_ANR)*model.pANRCRF[g]for g in model.G) for n in model.N) 
  
  def compute_anr_om(model):
    return sum(sum(model.pANRCap[g]*model.vM[n,g]*(model.pANRFC[g]+model.pANRVOM[g]*365*24) for g in model.G) for n in model.N) 
  
  def compute_h2_capex(model):
    return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*model.pH2CAPEX[h]*(1-model.pITC_H2)*model.pH2CRF[h] for h in model.H) for g in model.G) for n in model.N) 
  
  def compute_h2_om(model):
    return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 

  def get_crf(model):
    return sum(model.vS[g]*model.pANRCRF[g] for g in model.G)
  
  def compute_conv_costs(model):
    crf = model.pWACC / (1 - (1/(1+model.pWACC)**20) )# assumes 20 years lifetime for shaft and eaf
    costs = steel_cap_ton_per_annum*(model.pEAFCAPEX*(1-model.pITC_H2)*crf + model.pShaftFCAPEX*(1-model.pITC_H2)*crf/model.pRatioSteelDRI )
    return costs
  
  def get_deployed_cap(model):
    return sum(sum (model.vM[n,g]*model.pANRCap[g] for g in model.G) for n in model.N)
  
  def annualized_avoided_ng_costs(model):
    ng_price = utils.get_ng_price_aeo(model.pState)
    avoided_costs = ng_price*steel_cap_ton_per_annum*utils.coal_to_steel_ratio_bau
    return avoided_costs
  
  def compute_surplus_capacity(model):
    deployed_capacity = sum(sum (model.vM[n,g]*model.pANRCap[g] for g in model.G) for n in model.N)
    aux_elec_demand = elec_dem_MWh_per_day/24
    h2_elec_demand  = sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for g in model.G) for n in model.N) 
    return deployed_capacity - aux_elec_demand - h2_elec_demand

  def get_eq_elec_dem_h2(model): return sum(sum(sum(model.pH2CapElec[h,g]*model.vQ[n,h,g] for h in model.H) for g in model.G) for n in model.N) 


  ############## SOLVE ###################
  solver = SolverFactory('cplex')
  solver.options['timelimit'] = 240
  solver.options['mip pool relgap'] = 0.02
  solver.options['mip tolerances absmipgap'] = 1e-4
  solver.options['mip tolerances mipgap'] = 5e-3
  results = solver.solve(model, tee = False)

  results_dic = {}
  results_dic['id'] = plant
  results_dic['state'] = value(model.pState)
  results_dic['Steel prod. (ton/year)'] = steel_cap_ton_per_annum
  results_dic['H2 Dem. (kg/day)'] = value(model.pH2Dem)
  results_dic['Aux Elec Dem. (MWe)'] = value(model.pElecDem)
  results_dic['Net Revenues ($/year)'] = value(model.NetRevenues)
  results_dic['H2 PTC Revenues ($/year)'] = value(model.pH2Dem)*365*utils.h2_ptc
  results_dic['Net Revenues with H2 PTC ($/year)'] = results_dic['Net Revenues ($/year)']+results_dic['H2 PTC Revenues ($/year)']
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
    results_dic['Avoided NG costs ($/year)'] = value(annualized_avoided_ng_costs(model))
    results_dic['ANR CRF'] = value(get_crf(model))
    results_dic['Depl. ANR Cap. (MWe)'] = value(get_deployed_cap(model))
    results_dic['Depl H2 Cap. (MWe)'] = value(get_eq_elec_dem_h2(model))
    results_dic['Surplus ANR Cap. (MWe)'] = value(compute_surplus_capacity(model))
    results_dic['Net Annual Revenues ($/MWe/y)'] = (results_dic['Avoided NG costs ($/year)']\
                                                    +results_dic['Net Revenues ($/year)'])/results_dic['Depl. ANR Cap. (MWe)']
    results_dic['Net Annual Revenues with H2 PTC ($/MWe/y)'] = results_dic['Net Revenues with H2 PTC ($/year)']/results_dic['Depl. ANR Cap. (MWe)']
    for g in model.G: 
      if value(model.vS[g]) >=1: 
        results_dic['ANR type'] = g
        total_nb_modules = int(np.sum([value(model.vM[n,g]) for n in model.N]))
        results_dic['# ANR modules'] = total_nb_modules
        for n in model.N:
          if value(model.vM[n,g]) >=1:
            for h in model.H:
              results_dic[h] += value(model.vQ[n,h,g])
    results_dic['Breakeven price ($/MMBtu)'] = compute_breakeven_price(results_dic)
    print(f'Solved {plant}')
    return results_dic
  else:
    print('Not feasible.')
    return None

def compute_breakeven_price(results_ref):
  revenues = results_ref['Net Revenues ($/year)']
  plant_cap = results_ref['Steel prod. (ton/year)']
  breakeven_price_per_ton = ( -revenues - iron_ore_cost*bfbof_iron_cons*plant_cap - om_bfbof*plant_cap)/(COAL_CONS_RATE*plant_cap)
  breakeven_price = breakeven_price_per_ton/utils.coal_heat_content
  return breakeven_price

def main(anr_tag='FOAK', wacc=WACC, print_main_results=True, print_results=False): 
  # Go the present directory
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)

  # Load steel data
  steel_df = pd.read_excel('h2_demand_bfbof_steel_us_2022.xlsx', sheet_name='processed')
  steel_ids = list(steel_df['Plant'])

  # Load ANR and H2 parameters
  ANR_data, H2_data = utils.load_data(anr_tag=anr_tag)

  # Build results dataset one by one

  with Pool(10) as pool:
    results = pool.starmap(solve_steel_plant_deployment, [(plant, ANR_data, H2_data) for plant in steel_ids])
  pool.close()

  df = pd.DataFrame(results)

  excel_file = f'./results/raw_results_anr_{anr_tag}_h2_wacc_{str(wacc)}.xlsx'
  sheet_name = 'steel'
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

  # Median Breakeven price
  med_be = df['Breakeven price ($/MMBtu)'].median()
  return med_be


if __name__ == '__main__': 
  main(anr_tag=utils.LEARNING)
  #test()









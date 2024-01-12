from pyomo.environ import *
import pandas as pd
import numpy as np
import csv, os

"""version 0.2 Relaxed the heat balance constraint to be <= instead of ==, now the problem is feasible
  version 0.3, restructure code to save results of refinery deployment to csv file, 
  version 0.4 Solving with CPLEX, cannot solve problems with non-convex constraints so removing the wasteHeat variable
  version 1.0 Add revenues to the optimization function from avoided natural gas purchases
  version 2.0 Computation of breakeven natural gas price for each refinery and optimal ANR-H2 configuration
  version 2.1 Compute annual CO2 emissions from carbon intensity of nuclear energy"""

MaxANRMod = 20
WACC = 0.08
#SCF_TO_KGH2 = 0.002408 #kgh2/scf
NAT_GAS_PRICE = 6.45 #$/MMBTU
CONV_MJ_TO_MMBTU = 1/1055.05585 #MMBTU/MJ
EFF_H2_SMR = 159.6 #MJ/kgH2
CONV_MWh_to_MJ = 3600 #MJ/MWh
# GF: Glass Furnace
GFCAPEX = 1340000 #$/MWth
GFLT = 12 # years
GFFOM = 0.03 # % of capex

def load_data():
  ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name='FOAK', index_col=0)
  H2_data = pd.read_excel('./h2_tech.xlsx', sheet_name='Summary', index_col=[0,1])
  return ANR_data, H2_data

def get_plant_demand(plant_id):
    ref_df = pd.read_excel('h2_demand_industry_heat.xlsx', sheet_name='max')
    select_df = ref_df[ref_df['FACILITY_ID']==plant_id]
    demand_kg_day = float(select_df['H2 demand (kg/year)']/365)
    return demand_kg_day

def get_heat_demand(plant_id):
  ref_df = pd.read_excel('h2_demand_industry_heat.xlsx', sheet_name='max')
  select_df = ref_df[ref_df['FACILITY_ID']==plant_id]
  yearly_heat_demand = float(select_df['Heat demand (MJ/year)'])
  return yearly_heat_demand

def solve_refinery_deployment(plant_id, ANR_data, H2_data):

  model = ConcreteModel(plant_id)

  #### Data ####
  demand_daily = get_plant_demand(plant_id)
  model.pH2Dem = Param(initialize=demand_daily) # kg/day
  yearly_heat_demand = get_heat_demand(plant_id)
  model.pHeatDem = Param(initialize = yearly_heat_demand) #MJ/year


  #### Sets ####
  model.N = Set(initialize=list(range(MaxANRMod)))
  model.H = Set(initialize=list(set([elem[0] for elem in H2_data.index])))
  model.G = Set(initialize=ANR_data.index)


  #### Variables ####
  model.vS = Var(model.G, within=Binary, doc='Chosen ANR type')
  model.vM = Var(model.N, model.G, within=Binary, doc='Indicator of built ANR module')
  model.vQ = Var(model.N, model.H, model.G, within=NonNegativeIntegers, doc='Nb of H2 module of type H for an ANR module of type g')

  #### Parameters ####
  model.pWACC = Param(initialize = WACC)

  ### Glass furnace fueled by hydrogen ###
  model.pGFCAPEX = Param(initialize = GFCAPEX)#$/MWth
  model.pGFLT = Param(initialize = GFLT) # years
  model.pGFFOM = Param(initialize = GFFOM) # % of capex

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


  #### Objective ####  

  def annualized_costs_anr_h2(model):
    costs =  sum(sum(model.pANRCap[g]*model.vM[n,g]*((model.pANRCAPEX[g]*model.pANRCRF[g]+model.pANRFC[g])+model.pANRVOM[g]*365*24) \
      + sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2CAPEX[h]*model.pH2CRF[h]+model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 
    return costs
  
  def annualized_costs_gf(model):
    #capital recovery factor
    gf_crf = model.pWACC / (1 - (1/(1+model.pWACC)**model.pGFLT) ) 
    costs = (model.pHeatDem/(CONV_MWh_to_MJ*365))*model.pGFCAPEX*gf_crf#*(1+model.pGFFOM)
    return costs

  def annualized_net_rev(model):
    return -annualized_costs_anr_h2(model) - annualized_costs_gf(model)
  model.NetRevenues = Objective(expr=annualized_net_rev, sense=maximize)  


  #### Constraints ####
  # Meet refinery demand
  model.meet_ref_demand = Constraint(
    expr = model.pH2Dem <= sum(sum(sum(model.vQ[n,h,g]*model.pH2CapH2[h]*24 for g in model.G) for h in model.H)for n in model.N)
  )

  # Only one type of ANR deployed 
  model.max_ANR_type = Constraint(expr = sum(model.vS[g] for g in model.G)<=1)

  # Only modules of the chosen ANR type can be built
  def match_ANR_type(model, n, g):
    return model.vM[n,g] <= model.vS[g]
  model.match_ANR_type = Constraint(model.N, model.G, rule=match_ANR_type)


  # Heat and electricity balance
  def heat_elec_balance(model, n, g):
    return sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]/model.pANRThEff[g] for h in model.H) <= (model.pANRCap[g]/model.pANRThEff[g])*model.vM[n,g]
  model.heat_elec_balance = Constraint(model.N, model.G, rule=heat_elec_balance)


  #### DATA ####
  def compute_annual_carbon_emissions(model):
    return sum(sum(sum(model.pH2CarbonInt[h,g]*model.vQ[n,h,g]*model.pH2CapH2[h]*24*365 for g in model.G) for h in model.H) for n in model.N)

  #### SOLVE with CPLEX ####
  opt = SolverFactory('cplex')

  results = opt.solve(model, tee = True)
  results_ref = {}
  results_ref['FACILITY_ID'] = [plant_id]
  results_ref['H2 Dem. (kg/day)'] = [value(model.pH2Dem)]
  results_ref['Heat Dem. (MJ/year)'] = [value(model.pHeatDem)]
  results_ref['Cost ($/year)'] = [value(model.NetRevenues)]
  results_ref['Ann. carbon emissions (kgCO2eq/year)'] = [value(compute_annual_carbon_emissions(model))]
  for h in model.H:
    results_ref[h] = [0]
  if results.solver.termination_condition == TerminationCondition.optimal: 
    model.solutions.load_from(results)
    print('\n\n\n\n',' ------------ SOLUTION  -------------')
    for g in model.G: 
      if value(model.vS[g]) >=1: 
        print('Chosen type of advanced nuclear reactor is ',g)
        results_ref['ANR type'] = [g]
        total_nb_modules = int(np.sum([value(model.vM[n,g]) for n in model.N]))
        print(total_nb_modules, ' modules are needed.')
        results_ref['# ANR modules'] = [total_nb_modules]
        for n in model.N:
          if value(model.vM[n,g]) >=1:
            print('ANR Module # ',int(value(model.vM[n,g])), 'of type ', g)
            for h in model.H:
              print(int(value(model.vQ[n,h,g])), ' Hydrogen production modules of type:',h )
              if value(model.vQ[n,h,g]) > 0:
                results_ref[h][0] += value(model.vQ[n,h,g])
    
    return results_ref
  else:
    print('Not feasible.')
    return None

def compute_breakeven_price(results_ref):
  anr_h2_rev = results_ref['Cost ($/year)'][0]
  heat_demand = results_ref['Heat Dem. (MJ/year)'][0]*CONV_MJ_TO_MMBTU
  breakeven_price = -anr_h2_rev/heat_demand
  return breakeven_price


def main():
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)
  ref_df = pd.read_excel('h2_demand_industry_heat.xlsx', sheet_name='max')
  ref_ids = list(ref_df['FACILITY_ID'])
  ANR_data, H2_data = load_data()

  results_df = pd.DataFrame(columns=['FACILITY_ID', 'H2 Dem. (kg/day)', 'Heat Dem. (MJ/year)','Alkaline', 'HTSE', 'PEM', 'ANR type', '# ANR modules', 'Cost ($/year)','Ann. carbon emissions (kgCO2eq/year)', 'Breakeven NG price ($/MMBtu)'])
  not_feasible = []
  for ref_id in ref_ids:
    try:
      result_ref = solve_refinery_deployment(ref_id, ANR_data, H2_data)
      result_ref['Breakeven NG price ($/MMBtu)'] = [compute_breakeven_price(result_ref)]
      results_df = pd.concat([results_df, pd.DataFrame.from_dict(data=result_ref)])
    except ValueError:
      not_feasible.append(ref_id)

  results_df.sort_values(by=['H2 Dem. (kg/day)'], inplace=True)
  results_df.to_csv('./results/results_heat_process_deployment.csv', header=True, index=False)

  if len(not_feasible) >= 1:
    print('\n\n\n\n\n Not feasible: ')
    for ref in not_feasible:
      print('Refinery :', ref)
      print('Demand :', get_plant_demand(ref), ' kg/day')



if __name__ == '__main__':
  main()
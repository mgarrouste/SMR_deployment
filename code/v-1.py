from pyomo.environ import *
import pandas as pd
import numpy as np


MaxANRMod = 2
MaxH2Mod = 10
Refinery_id_example = 'HU_TUS'
SCF_TO_KGH2 = 0.002408 #kgh2/scf


def load_data():
  ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name='FOAK', index_col=0)
  H2_data = pd.read_excel('./h2_tech.xlsx', sheet_name='Summary', index_col=[0,1])
  return ANR_data, H2_data

def get_refinery_demand(refinery_id = Refinery_id_example):
  ref_df = pd.read_excel('h2_demand_refineries.xlsx', sheet_name='processed')
  select_df = ref_df[ref_df['refinery_id']==Refinery_id_example]
  demand_sfc = float(select_df['demand_2022'])
  demand_kg_day = demand_sfc*SCF_TO_KGH2
  demand_kg_year = demand_kg_day*365
  return demand_kg_day



model = ConcreteModel('deployment at one refinery')


model.pRefDem = Param(initialize=get_refinery_demand) # kg/day

ANR_data, H2_data = load_data()
model.N = Set(initialize=list(range(MaxANRMod)))
model.H = Set(initialize=list(set([elem[0] for elem in H2_data.index])))


model.vM = Var(model.N, within=Binary, doc='Indicator of built ANR module')
model.vQ = Var(model.N, model.H, within=NonNegativeIntegers, doc='Nb of H2 module of type H for an ANR module')
model.wasteHeat = Var(model.N, within=PositiveReals, doc='Remaining heat no fed to h2 processes per ANR module')

g = 'iPWR'
#h = 'HTSE'


#### Parameters ####
model.pWACC = Param(initialize = 0.08)

### H2 ###

data = H2_data.reset_index(level='ANR')[['H2Cap (kgh2/h)']]
data.drop_duplicates(inplace=True)
model.pH2CapH2 = Param(model.H, initialize = data.to_dict()['H2Cap (kgh2/h)'])

@model.Param(model.H)
def pH2CapElec(model, h):
  return float(H2_data.loc[h,g]['H2Cap (MWe)'])

# Electric and heat consumption
@model.Param(model.H)
def pH2ElecCons(model, h):
  return float(H2_data.loc[h,g]['H2ElecCons (MWhe/kgh2)'])

@model.Param(model.H)
def pH2HeatCons(model, h):
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

### ANR ###
model.pANRCap = Param(initialize = float(ANR_data.loc[g]['Power in MWe']))
model.pANRVOM = Param(initialize = float(ANR_data.loc[g]['VOM in $/MWh-e']))
model.pANRFC = Param(initialize = float(ANR_data.loc[g]['FOPEX $/MWe-y']))
model.pANRCAPEX = Param(initialize = float(ANR_data.loc[g]['CAPEX $/MWe']))
model.pANRCRF = Param(initialize = float(model.pWACC / (1 - (1/(1+model.pWACC)**ANR_data.loc[g,'Life (y)']))))
model.pANRThEff = Param(initialize = float(ANR_data.loc[g]['Power in MWe']/ANR_data.loc[g]['Power in MWt']))



#### Objective ####
def annualized_cost(model):
    return sum(model.pANRCap*model.vM[n]*((model.pANRCAPEX*model.pANRCRF+model.pANRFC)+model.pANRVOM*365*24) for n in model.N)\
      + sum(sum(model.pH2CapElec*model.vQ[n,h]*(model.pH2CAPEX*model.pH2CRF+model.pH2FC+model.pH2VOM*365*24) for h in model.H)for n in model.N) 
#model.Cost = Objective(rule=annualized_cost, sense=minimize)  
model.value = Objective(
  expr = sum(sum(model.pANRCap*model.vM[n]*model.pANRCAPEX + model.pH2CapElec[h]*model.vQ[n,h]*model.pH2CAPEX[h] for h in model.H) for n in model.N), 
  sense= minimize
)



#### Constraints ####
# Meet refinery demand
model.meet_ref_demand = Constraint(
  expr = model.pRefDem <= sum(sum(model.vQ[n,h]*model.pH2CapH2[h]*24 for h in model.H)for n in model.N)
)

# At least one ANR module of any type deployed
def min_ANR_mod(model):
  return 1 <= sum(model.vM[n] for n in model.N)
model.min_ANR_mod = Constraint(rule=min_ANR_mod)

# At least one H2 module of any type deployed
def min_H2_mod(model):
  return 1 <= sum(sum(model.vQ[n,h] for h in model.H)for n in model.N) 
model.min_H2_mod = Constraint(rule=min_H2_mod)

# Heat and electricity balance
def heat_elec_balance(model, n):
  return sum(model.pH2CapH2[h]*model.vQ[n,h]*(model.pH2ElecCons[h]/model.pANRThEff)\
    + model.pH2CapH2[h]*model.vQ[n,h]*model.pH2HeatCons[h] for h in model.H)\
      == (model.pANRCap/model.pANRThEff +model.wasteHeat[n])*model.vM[n]
model.heat_elec_balance = Constraint(model.N, rule=heat_elec_balance)


model.pprint()

opt = SolverFactory('mindtpy')

results = opt.solve(model, tee = True)
if results.solver.termination_condition == TerminationCondition.optimal: 
  model.solutions.load_from(results)
  print(' ------------ SOLUTION  -------------')
  for n in model.N:
    print('ANR Module # ',int(value(model.vM[n])))
    print('Waste heat : ',value(model.wasteHeat[n]), ' MWh')
    for h in model.H:
      print(int(value(model.vQ[n,h])), ' Hydrogen production modules of type:',h )
else:
  print('Not feasible?')
  model.display()


from pyomo.environ import *
import pandas as pd
import numpy as np


MaxANRMod = 20
Refinery_id_example = 'TE_KEN'
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

#### Data ####
model.pRefDem = Param(initialize=10000) # kg/day

ANR_data, H2_data = load_data()


#### Sets ####
model.N = Set(initialize=list(range(MaxANRMod)))
model.H = Set(initialize=list(set([elem[0] for elem in H2_data.index])))
model.G = Set(initialize=ANR_data.index)

print('\n', 'Sets established')

#### Variables ####
model.vS = Var(model.G, within=Binary, doc='Chosen ANR type')
model.vM = Var(model.N, model.G, within=Binary, doc='Indicator of built ANR module')
model.vQ = Var(model.N, model.H, model.G, within=NonNegativeIntegers, doc='Nb of H2 module of type H for an ANR module of type g')
model.wasteHeat = Var(model.N, model.G, within=PositiveReals, doc='Remaining heat no fed to h2 processes per ANR module')

print('\n', 'Variables established')

#### Parameters ####
model.pWACC = Param(initialize = 0.08)

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
def annualized_cost(model):
    return sum(sum(model.pANRCap[g]*model.vM[n,g]*((model.pANRCAPEX[g]*model.pANRCRF[g]+model.pANRFC[g])+model.pANRVOM[g]*365*24) \
      + sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]*(model.pH2CAPEX[h]*model.pH2CRF[h]+model.pH2FC[h]+model.pH2VOM[h]*365*24) for h in model.H) for g in model.G) for n in model.N) 
model.Cost = Objective(rule=annualized_cost, sense=minimize)  


#### Constraints ####
# Meet refinery demand
model.meet_ref_demand = Constraint(
  expr = model.pRefDem <= sum(sum(sum(model.vQ[n,h,g]*model.pH2CapH2[h]*24 for g in model.G) for h in model.H)for n in model.N)
)

# Only one type of ANR deployed 
def max_ANR_type(model):
  return 1 == sum(model.vS[g] for g in model.G)
model.max_ANR_type = Constraint(rule=max_ANR_type)

# Only modules of the chosen ANR type can be built
def match_ANR_type(model, n, g):
  return model.vM[n,g] <= model.vS[g]
model.match_ANR_type = Constraint(model.N, model.G, rule=match_ANR_type)


# Heat and electricity balance
def heat_elec_balance(model, n, g):
  return sum(model.pH2CapElec[h,g]*model.vQ[n,h,g]/model.pANRThEff[g] for h in model.H) == ((model.pANRCap[g]/model.pANRThEff[g]) - model.wasteHeat[n,g])*model.vM[n,g]
model.heat_elec_balance = Constraint(model.N, model.G, rule=heat_elec_balance)

# Waste heat
def waste_heat_max(model, g, n):
  return model.wasteHeat[n,g] <= model.pANRCap[g]
model.waste_heat_max = Constraint(model.G, model.N, rule=waste_heat_max)



opt = SolverFactory('mindtpy')

results = opt.solve(model, tee = True)
if results.solver.termination_condition == TerminationCondition.optimal: 
  model.solutions.load_from(results)
  print('\n',' ------------ SOLUTION  -------------')
  for g in model.G: 
    if value(model.vS[g]) >=1: 
      print('Chosen type of advanced nuclear reactor is ',g)
      total_nb_modules = int(np.sum([value(model.vM[n,g]) for n in model.N]))
      print(total_nb_modules, ' modules are needed.')
      for n in model.N:
        if value(model.vM[n,g]) >=1:
          print('ANR Module # ',int(value(model.vM[n,g])), 'of type ', g)
          print('Waste heat : ',value(model.wasteHeat[n,g]), ' MWh')
          for h in model.H:
            print(int(value(model.vQ[n,h,g])), ' Hydrogen production modules of type:',h )
        
else:
  print('Not feasible?')
  model.display()


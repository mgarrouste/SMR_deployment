from pyomo.environ import *
import pandas as pd

MaxANRMod = 12
MaxH2Mod = 1000
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
  return demand_kg_year


def build_model():
  model = ConcreteModel(name='Deployment at Refineries')



  #### Sets ####
  ANR_data, H2_data = load_data()
  # SMRs
  model.G = Set(initialize=ANR_data.index)
  model.N = Set(initialize=list(range(MaxANRMod)))
  # Hydrogen prod technologies
  model.H = Set(initialize=list(set([elem[0] for elem in H2_data.index])))
  model.O = Set(initialize=list(range(MaxH2Mod)))



  #### Variables ####
  model.c = Var(doc='Annualized costs')
  model.vS = Var(model.G, within=Binary, doc='Indicator of chosen ANR type')
  model.vM = Var(model.G, model.N, within=Binary, doc='Indicator of built ANR module')
  model.vP = Var(model.H, within=Binary, doc='Indicator of chosen H2 type')
  model.vQ = Var(model.H, model.O, within=Binary, doc='Indicator of built H2 module')



  #### Parameters ####
  # WACC 
  model.pWACC = 0.08
  # Demand in kg/year
  model.pRefDem = Param(initialize=get_refinery_demand)

  # H2 tech capacity in kg-h2/h and MWe
  @model.Param(model.H)
  def pH2CapH2(model, h):
    data = H2_data.reset_index(level='ANR')[['H2Cap (kgh2/h)']]
    data.drop_duplicates(inplace=True)
    return float(data.loc[h,'H2Cap (kgh2/h)'])

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

  # Costs for H2
  @model.Param(model.H)
  def pH2VOM(model, h):
    data = H2_data.reset_index(level='ANR')[['VOM ($/MWhe)']]
    data.drop_duplicates(inplace=True)
    return float(data.loc[h,'VOM ($/MWhe)'])

  @model.Param(model.H)
  def pH2FC(model, h):
    data = H2_data.reset_index(level='ANR')[['FOM ($/MWe-year)']]
    data.drop_duplicates(inplace=True)
    return float(data.loc[h,'FOM ($/MWe-year)'])
  
  @model.Param(model.H)
  def pH2CAPEX(model, h):
    data = H2_data.reset_index(level='ANR')[['CAPEX ($/MWe)']]
    data.drop_duplicates(inplace=True)
    return float(data.loc[h,'CAPEX ($/MWe)'])
  
  @model.Param(model.H)
  def pH2CRF(model, h):
    data = H2_data.reset_index(level='ANR')[['Life (y)']]
    data = data.groupby(level=0).mean()
    crf = model.pWACC / (1 - (1+model.pWACC)**float(data.loc[h,'Life (y)']))
    return crf
  
  # Capacity of ANRs MWt
  @model.Param(model.G)
  def pANRCap(model, g):
    return float(ANR_data.loc[g]['Power in MWe'])

  # Parameters for ANRs
  @model.Param(model.G)
  def pANRVOM(model, g):
    return float(ANR_data.loc[g]['VOM in $/MWh-e'])
  
  @model.Param(model.G)
  def pANRFC(model, g):
    return float(ANR_data.loc[g]['FC in $/MWh-e'])

  @model.Param(model.G)
  def pANRCAPEX(model, g):
    return float(ANR_data.loc[g]['CAPEX $/MWe'])
  
  @model.Param(model.G)
  def pANRCRF(model, g):
    return model.pWACC / (1 - (1+model.pWACC)**float(ANR_data.loc[g,'Life (y)']))
    
  @model.Param(model.G)
  def pANRThEff(model, g):
    return float(ANR_data.loc[g]['Power in MWe']/ANR_data.loc[g]['Power in MWt'])



  #### Objective ####
  def annualized_cost(model):
    return sum(model.pANRCap[g]*model.pANRThEff[g]*model.vM[g,n]*((model.pANRCAPEX[g]*model.pANRCRF[g]+model.pANRFC[g])+model.pANRVOM[g]) for n in model.N for g in model.G)\
      + sum(model.pH2CapElec[h,g]*model.vQ[h,o]*(model.pH2CAPEX[h]*model.pH2CRF[h]+model.pH2FC[h]+model.pH2VOM[h]) for o in model.O for h in model.H for g in model.G)
  model.Cost = Objective(rule=annualized_cost, sense=minimize)  


  #### Constraints ####
  # Meet refinery demand
  def meet_ref_demand(model):
    return model.pRefDem <= sum(model.vQ[h,o]*model.pH2CapH2[h]*365*24 for o in model.O for h in model.H)
  model.meet_ref_demand = Constraint(rule=meet_ref_demand)

  # At least one ANR module of any type deployed
  def min_ANR_mod(model):
    return 1 <= sum(model.vM[g,n] for n in model.N for g in model.G)
  model.min_ANR_mod = Constraint(rule=min_ANR_mod)

  # At least one H2 module of any type deployed
  def min_H2_mod(model):
    return 1 <= sum(model.vQ[h,o] for o in model.O for h in model.H)
  model.min_H2_mod = Constraint(rule=min_H2_mod)

  # Only one type of ANR deployed 
  def max_ANR_type(model):
    return 1 == sum(model.vS[g] for g in model.G)
  model.max_ANR_type = Constraint(rule=max_ANR_type)

  # Only modules of the chosen ANR type can be built
  def match_ANR_type(model, g, n):
    return model.vM[g,n] <= model.vS[g]
  model.match_ANR_type = Constraint(model.G, model.N, rule=match_ANR_type)

  # Heat and electricity balance
  def heat_elec_balance(model, g):
    return sum(model.pH2CapH2[h]*model.vQ[h,o]*(model.pH2ElecCons[h,g]/model.pANRThEff[g]) \
      + model.pH2CapH2[h]*model.vQ[h,o]*model.pH2HeatCons[h,g] for o in model.O for h in model.H)\
        <= sum(model.pANRCap[g]*model.vM[g,n] for n in model.N)
  model.heat_elec_balance = Constraint(model.G, rule=heat_elec_balance)

  return model


if __name__ == '__main__':
  build_model()
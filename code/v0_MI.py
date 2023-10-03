from pyomo.environ import *
import pandas as pd

MaxANRMod = 12
MaxH2Mod = 1000

def load_parameters(model):
  return model

def load_data():
  ANR_data = pd.read_excel('./ANRs.xlsx', index_col=0)
  H2_data = pd.read_excel('./h2_tech.xlsx', sheet_name='Summary', index_col=[0,1])
  return ANR_data, H2_data

def build_model():
  model = ConcreteModel(name='Deployment at Refineries')

  ## Sets ##
  ANR_data, H2_data = load_data()
  # SMRs
  model.G = Set(initialize=ANR_data.index)
  model.N = Set(initialize=list(range(MaxANRMod)))
  # Hydrogen prod technologies
  model.H = Set(initialize=list(set([elem[0] for elem in H2_data.index])))
  model.O = Set(initialize=list(range(MaxH2Mod)))

  ## Variables ##
  model.c = Var(doc='Annualized costs')
  model.vS = Var(model.G, within=Binary, doc='Indicator of chosen ANR type')
  model.vM = Var(model.G, model.N, within=Binary, doc='Indicator of built ANR module')
  model.vP = Var(model.H, within=Binary, doc='Indicator of chosen H2 type')
  model.vQ = Var(model.H, model.O, within=Binary, doc='Indicator of built H2 module')

  ## Parameters ##


  return model

if __name__ == '__main__':
  build_model()
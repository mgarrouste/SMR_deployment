from pyomo.environ import *
import pandas as pd

model = ConcreteModel()


# DATA
anrs_data = pd.read_excel('ANRs.xlsx', sheet_name='FOAK')
anrs_data.set_index('Reactor', inplace=True)

ref_data = pd.read_excel('h2_demand_refineries.xlsx', sheet_name='processed')
ref_data.set_index('refinery_id', inplace=True)

print(anrs_data)
print(ref_data)

example_refinery = 'CH_RIC'


import pandas as pd 
import numpy as np

N=1000
WACC = 0.077
ITC_ANR = 0.3
ITC_H2 = 0.3

def update_capex_costs(ANR_data, learning_rate_anr_capex, H2_data, learning_rate_h2_capex):
  ANR_data['CAPEX $/MWe'] = ANR_data.apply(lambda x: x['CAPEX $/MWe']*np.power(N, np.log2(1-learning_rate_anr_capex)), axis=1)
  H2_data['CAPEX ($/MWe)'] = H2_data.apply(lambda x: x['CAPEX ($/MWe)']*np.power(N, np.log2(1-learning_rate_h2_capex)), axis=1)
  return ANR_data, H2_data


def load_data(learning_rate_anr_capex, learning_rate_h2_capex):
  H2_data = pd.read_excel('./h2_tech.xlsx', sheet_name='Summary', index_col=[0,1])
  ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name='FOAK', index_col=0)
  ANR_data, H2_data = update_capex_costs(ANR_data, learning_rate_anr_capex, H2_data, learning_rate_h2_capex)
  return ANR_data, H2_data


if __name__ =='__main__':
  anr_d, h2_d = load_data(0.1, 0.1)
  print(anr_d[['CAPEX $/MWe']])
  print(h2_d[['CAPEX ($/MWe)']])
import pandas as pd 
import numpy as np

N=1000

#INflation
conversion_2021usd_to_2020usd = 0.99 #2020$/2021$ source: data.bls.gov

#cost of captial
WACC = 0.077
# Credits
ITC_ANR = 0.3 #%
ITC_H2 = 0.3 #%
h2_ptc = 3 #$/kgH2 clean
elec_ptc = 25 #$/MWhe
# ANR-H2
avg_elec_cons_h2 = 0.022 #MWhe/kgh2, calculated from avg thermal efficiency (micro, PBR-HTGR, iMSR) and coupling efficiency with HTSE
# Ammonia process modelling parameters
nh3_carbon_intensity = 2.30 # tcO2/tNH3
nh3_nrj_intensity = 30.82 #MMBtu/tNH3
nh3_h2_intensity = 507.71 # kgh2/tNH3
# Refining
smr_carbon_intensity = 11.888 #kgCO2/kgh2
smr_nrj_intensity = 0.1578 # MMBtu/kgh2
# Process heat
heat_avg_carbon_intensity = 0.002028 #tCO2/MMBtu
h2_hhv = 141.88 # MJ/kgh2
mmbtu_to_mj = 1055.06 #mj/MmBtu
# Steel
coal_to_steel_ratio_bau = 0.474 # tcoal/tsteel
co2_to_steel_ratio_bau = 1.990 #tco2/tsteel
h2_to_dri_ratio = 67.095 #kgh2/tdri
steel_to_dri_ratio = 0.9311 # tsteel/tdri
aux_elec_dri = 0.1025 # MWhe/tdri
eaf_elec = 0.461 # MWhe/tsteel
coal_heat_content = 28.97 #MMBtu/Mton
dri_co2_intensity = 40 # kgCO2/ton-DRI
shaft_CAPEX = 250 # $/tDRI/year
eaf_CAPEX = 160 # $/tsteel/year
eaf_OM  = 24.89 # $/tsteel (EAF and casting)
iron_ore_cost = 100 #$/t_ironore # $/tironore
ratio_ironore_DRI = 1.391 # tironore/tDRI
bfbof_iron_cons = 1.226 #t_ironore/t_steel
om_bfbof = 178.12 #$/t_steel

def update_capex_costs(ANR_data, learning_rate_anr_capex, H2_data, learning_rate_h2_capex, N=N):
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
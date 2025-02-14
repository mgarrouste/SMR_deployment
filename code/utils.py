import pandas as pd 
import numpy as np
import glob

N=1000
LEARNING = 'FOAK'

palette={'HTGR':'orange', 
         'iMSR':'blue', 
         'iPWR':'green', 
         'PBR-HTGR':'darkorchid', 
         'Micro':'darkgrey'}
cashflows_color_map = {'SMR CAPEX': 'navy', 
             'ANR for H2 CAPEX': 'royalblue',
               'H2 CAPEX': 'lightsteelblue', 
               'SMR O&M':'darkgreen', 
               'ANR for H2 O&M':'forestgreen', 
               'H2 O&M':'palegreen',
               'Conversion':'grey',
               'Avoided Fossil Fuel Costs':'darkorchid', 
               'H2 PTC':'red', 
               'Electricity (cogen)':'pink'}
#INflation
conversion_2021usd_to_2020usd = 0.99 #2020$/2021$ source: data.bls.gov
conversion_2022usd_to_2020usd = 1/1.09 #2020$/2022$
# CCUS costs
ccus_cost = 50#$/ton CO2

# Energy conversion
mmbtu_to_mwh = 0.293071 # MWh/MMBtu
mwh_to_mj = 3600 # MJ/MWh

# Natural gas
mcf_to_mmbtu = 1.038 #mmbtu/mcf

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
# GF: Glass Furnace
GFCAPEX = 1340000 #$/MWth
GFLT = 12 # years
# Steel
coal_to_steel_ratio_bau = 0.663 # tcoal/tsteel
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

def calculate_irr(Co, Celec, Ch2, Cff, lifetime=20, ptc=True, add_capex={}):
  """Calculates the IRR given: 
  Args: 
    Co (float): initial total investment, CAPEX SMR, H2, and conversion costs
    Celec (float): yearly revenues from electricity production (cogeneration)
    Ch2 (float): yearly revenues from the H2 PTC (stops after 10 years)
    Cff (float): yearly revenues from avoided fossil fuel costs
    add_capex (dict[int:float]): dictionary with additional investments necessary, with year as key and amount as value
  Returns: 
    irr (float): internal rate of return
  """
  import numpy_financial as npf
  if ptc: 
    list_cashflows = [-Co]+[Celec+Ch2+Cff]*10+[Celec+Cff]*(lifetime-10)
  else:
    list_cashflows = [-Co]+[Celec+Cff]*lifetime
  if len(add_capex)>0:
    for year, add_investment in add_capex.items():
      list_cashflows[year] += add_investment
  return round(npf.irr(list_cashflows), 2)


def get_met_coal_eia_aeo_price():
  price2024 = 5.679083 # 2022$/MMBtu
  price = price2024*conversion_2022usd_to_2020usd
  return price



def letter_annotation(ax, xoffset, yoffset, letter):
  ax.text(xoffset, yoffset, letter, transform=ax.transAxes,
         size=12, weight='bold')
 
def compute_average_electricity_prices(cambium_scenario, year):
  folder = f'./input_data/cambium_{cambium_scenario.lower()}_state_hourly_electricity_prices'
  list_csv_files = glob.glob(folder+'/Cambium*.csv')
  state_prices = pd.DataFrame(columns=['average price ($/MWhe)', 'state'])
  state_prices.set_index('state', inplace=True)
  for file in list_csv_files:
    if str(year) in file:
      state = file.split('_')[-2]
      avg_price = pd.read_csv(file, skiprows=5)['energy_cost_enduse'].mean()
      state_prices.loc[state, 'average price ($/MWhe)'] = avg_price
  state_prices.to_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx')


def compute_cogen(df, surplus_cap_col_name, state_col_name, cambium_scenario, year):
  try:
    elec_prices_df = pd.read_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx', index_col=0)
  except FileNotFoundError:
    compute_average_electricity_prices(cambium_scenario, year)
    elec_prices_df = pd.read_excel(f'./results/average_electricity_prices_{cambium_scenario}_{year}.xlsx', index_col=0)
  df['Electricity revenues ($/y)'] = df.apply(lambda x: x[surplus_cap_col_name]*elec_prices_df.loc[x[state_col_name]]*8760, axis=1)
  return df
  

def get_ng_price_current(state):
  ng_prices_map = pd.read_excel('./input_data/ng_prices_state_annual_us.xlsx', sheet_name='clean_data_2022', index_col='state')
  ng_price = float(ng_prices_map.loc[state, 'price ($/MMBtu)'])
  return ng_price


def get_ng_price_aeo(state):
  state_prices = pd.read_csv('./input_data/eia_aeo_industrial_sector_ng_prices_2024.csv', index_col='state')
  ng_price = state_prices.loc[state, 'price']
  return ng_price


def update_capex_costs(ANR_data, learning_rate_anr_capex, H2_data, learning_rate_h2_capex, N=N):
  ANR_data['CAPEX $/MWe'] = ANR_data.apply(lambda x: x['CAPEX $/MWe']*np.power(N, np.log2(1-learning_rate_anr_capex)), axis=1)
  H2_data['CAPEX ($/MWe)'] = H2_data.apply(lambda x: x['CAPEX ($/MWe)']*np.power(N, np.log2(1-learning_rate_h2_capex)), axis=1)
  return ANR_data, H2_data


def load_data(anr_tag='FOAK'):
  H2_data = pd.read_excel('./h2_tech.xlsx', sheet_name='Summary', index_col=[0,1])
  ANR_data = pd.read_excel('./ANRs.xlsx', sheet_name=anr_tag, index_col=0)
  #ANR_data, H2_data = update_capex_costs(ANR_data, learning_rate_anr_capex, H2_data, learning_rate_h2_capex)
  return ANR_data, H2_data


if __name__ =='__main__':
  print(get_ng_price_aeo('LA'))
  print(type(get_ng_price_aeo('LA')))
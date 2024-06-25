from ANR_application_comparison import load_h2_results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def load_data():
  # Load data SMR-H2
  foak = load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
  foak = foak.reset_index()
  foak_ptc = foak[['id','Industry','Breakeven price ($/MMBtu)']]
  foak_ptc['stage'] = 'FOAK'
  foak_noptc = foak[['id','Industry','BE wo PTC ($/MMBtu)']]
  foak_noptc['stage'] = 'FOAK-NoPTC'
  foak_noptc = foak_noptc.rename(columns={'BE wo PTC ($/MMBtu)':'Breakeven price ($/MMBtu)'})

  noak = load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
  noak = noak.reset_index()
  noak_ptc = noak[['id','Industry','Breakeven price ($/MMBtu)']]
  noak_ptc['stage'] = 'NOAK'
  noak_noptc = noak[['id','Industry','BE wo PTC ($/MMBtu)']]
  noak_noptc['stage'] = 'NOAK-NoPTC'
  noak_noptc = noak_noptc.rename(columns={'BE wo PTC ($/MMBtu)':'Breakeven price ($/MMBtu)'})
  total = pd.concat([foak_noptc, foak_ptc, noak_noptc, noak_ptc], ignore_index=True)
  amdf = total[total.Industry == 'ammonia']
  redf = total[total.Industry == 'refining']
  stdf = total[total.Industry == 'steel']

  # Load data RES
  res_ammonia = pd.read_csv('./results/res_be_ammonia.csv')
  res_ammonia['industry'] = 'ammonia'
  res_ref = pd.read_csv('./results/res_be_refining.csv')
  res_ref['industry'] = 'refining'
  res_steel = pd.read_csv('./results/res_be_steel.csv')
  res_steel['industry'] = 'steel'
  res = pd.concat([res_ammonia, res_ref, res_steel], ignore_index=True)
  res.replace({'SMR 89% CCUS':'NG 89% CCUS'}, inplace=True)
  print(res)
  return amdf, redf, stdf, res

def plot_comparison(amdf, redf, stdf, res):
  fig, ax = plt.subplots(3,1, sharex=True)
  xmin, xmax = -40, 130

  stagep = {'FOAK':'yellowgreen', 'NOAK':'forestgreen', 
            'FOAK-NoPTC':'gold', 'NOAK-NoPTC':'darkorange'}
  resp = {'Grid PEM':'blue', 'Wind':'cyan', 'Solar PV-E':'plum', 'NG 89% CCUS':'gray'}
  
  res_list = res.RES.unique()

  # Ammonia
  #sns.stripplot(ax=ax[0], data=amdf, x='Breakeven price ($/MMBtu)',palette=stagep, hue='stage', alpha=0.6)
  sns.boxplot(ax=ax[0], data=amdf, x='Breakeven price ($/MMBtu)',palette=stagep, hue='stage', fill=False, width=0.5)
  sns.despine()
  ax[0].set_ylabel('Ammonia')
  ax[0].get_legend().set_visible(False)
  res_am = res[res.industry=='ammonia']
  for r in res_list:
    be = res_am[res_am.RES == r]['Breakeven price ($/MMBtu)'].iloc[0]
    ax[0].axvline(be, color=resp[r], ls='--',label=r)

  # REfinig
  sns.boxplot(ax=ax[1], data=redf, x='Breakeven price ($/MMBtu)',palette=stagep, hue='stage', fill=False, width=0.5)
  sns.despine()
  ax[1].set_ylabel('Refining')
  ax[1].get_legend().set_visible(False)
  res_re = res[res.industry=='refining']
  for r in res_list:
    be = res_re[res_re.RES == r]['Breakeven price ($/MMBtu)'].iloc[0]
    ax[1].axvline(be, color=resp[r], ls='--',label=r)
  
  # steel
  sns.boxplot(ax=ax[2], data=stdf, x='Breakeven price ($/MMBtu)',palette=stagep, hue='stage', fill=False, width=0.5)
  sns.despine()
  ax[2].set_ylabel('Steel')
  ax[2].get_legend().set_visible(False)
  res_st = res[res.industry=='steel']
  for r in res_list:
    be = res_st[res_st.RES == r]['Breakeven price ($/MMBtu)'].iloc[0]
    ax[2].axvline(be, color=resp[r], ls='--',label=r)



  #Common legend for whole figure
  h3, l3 = ax[0].get_legend_handles_labels()
  #h4, l4 = beax.get_legend_handles_labels()
  by_label = dict(zip(l3, h3))
  fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.,.33), ncol=2)

  fig.tight_layout()
  fig.savefig('./results/comparison_res_h2.png')




def main():
  amdf, redf, stdf, res = load_data()
  plot_comparison(amdf, redf, stdf, res)



if __name__ == '__main__':
  warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

  main()


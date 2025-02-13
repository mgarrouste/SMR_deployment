import pandas as pd
from ANR_application_comparison import load_h2_results, load_heat_results
import warnings
import matplotlib.pyplot as plt
from utils import palette
import seaborn as sns

def load_data():
  # SMRs parameters
  # Current CAPEX values for SMRs
  smrs = pd.read_excel('./ANRs.xlsx', sheet_name='FOAK')[['Reactor','CAPEX $/MWe']]

  # Load data SMR-H2
  h2 = load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
  h2.rename(columns={'ANR type':'SMR'}, inplace=True)
  h2 = h2.reset_index()
  h2_ptc = h2[['id','Industry','Breakeven CAPEX ($/MWe)', 'SMR']]
  h2_ptc['scenario'] = 'With H2 PTC'
  h2_noptc = h2[['id','Industry','Breakeven CAPEX wo PTC ($/MWe)', 'SMR']]
  h2_noptc['scenario'] = 'Without H2 PTC'
  h2_noptc = h2_noptc.rename(columns={'Breakeven CAPEX wo PTC ($/MWe)':'Breakeven CAPEX ($/MWe)'})

  total = pd.concat([h2_ptc, h2_noptc], ignore_index=True)
  total = total.merge(smrs, left_on='SMR', right_on='Reactor')
  total['Margin'] = 100*(total['Breakeven CAPEX ($/MWe)']-total['CAPEX $/MWe'])/total['CAPEX $/MWe']
  total['Breakeven CAPEX ($/MWe)'] /=1e6
  amdf = total[total.Industry == 'ammonia']
  redf = total[total.Industry == 'refining']
  stdf = total[total.Industry == 'steel']
  print(total)

  # Load heat SMR-H2 results
  heat = load_heat_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=True)
  heat = heat.reset_index(names=['id'])
  heat = heat.merge(smrs, left_on='SMR', right_on='Reactor')
  
  heat_ptc = heat[['id', 'Breakeven CAPEX ($/MWe)', 'SMR', 'CAPEX $/MWe']]
  heat_noptc = heat[['id','Breakeven CAPEX wo PTC ($/MWe)', 'SMR','CAPEX $/MWe']]
  heat_ptc['scenario'] = 'With H2 PTC'
  heat_noptc['scenario'] = 'Without H2 PTC'
  heat_noptc = heat_noptc.rename(columns={'Breakeven CAPEX wo PTC ($/MWe)':'Breakeven CAPEX ($/MWe)'})
  heat_total = pd.concat([heat_ptc, heat_noptc], ignore_index=True)
  heat_total['Margin'] = 100*(heat_total['Breakeven CAPEX ($/MWe)']-heat_total['CAPEX $/MWe'])/heat_total['CAPEX $/MWe']
  heat_total['Breakeven CAPEX ($/MWe)'] /=1e6
  print(heat_total)
  return amdf, redf, stdf, heat_total


def plot_comparison_margins(amdf, redf, stdf, heat):
  fig, ax = plt.subplots(4,1, sharex=True, figsize=(9,4))

  scenariop = {'With H2 PTC':'green', 'Without H2 PTC':'crimson'}

  # Ammonia
  sns.boxplot(ax=ax[0], data=amdf, x='Margin',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[0].set_ylabel('Ammonia')
  ax[0].get_legend().set_visible(False)

  # refining
  sns.boxplot(ax=ax[1], data=redf, x='Margin',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[1].set_ylabel('Refining')
  ax[1].get_legend().set_visible(False)

  # Steel
  sns.boxplot(ax=ax[2], data=stdf, x='Margin',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[2].set_ylabel('Steel')
  ax[2].get_legend().set_visible(False)

  # Heat
  sns.boxplot(ax=ax[3], data=heat, x='Margin',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[3].set_ylabel('Process\nheat')
  ax[3].get_legend().set_visible(False)

  #ax[3].set_xlim(-50,450)
  #Common legend for whole figure
  h3, l3 = ax[0].get_legend_handles_labels()
  by_label = dict(zip(l3, h3))
  fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.9,.25), ncol=1)

  ax[3].set_xlabel('CAPEX profitability margin (%)')

  plt.subplots_adjust(wspace=.1,
                    hspace=0.1)

  fig.savefig('./results/capex_sa_be_analysis_margin.pdf')
  plt.show()


def plot_comparison(amdf, redf, stdf, heat):
  fig, ax = plt.subplots(4,1, sharex=True)

  scenariop = {'With H2 PTC':'green', 'Without H2 PTC':'crimson'}

  # Current CAPEX values for SMRs
  smrs = pd.read_excel('./ANRs.xlsx', index_col='Reactor', sheet_name='FOAK')[['CAPEX $/MWe']]

  # Ammonia
  sns.boxplot(ax=ax[0], data=amdf, x='Breakeven CAPEX ($/MWe)',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[0].set_ylabel('Ammonia')
  ax[0].get_legend().set_visible(False)
  for smr in smrs.index :
    ax[0].axvline(smrs.loc[smr, 'CAPEX $/MWe']/1e6, color=palette[smr], ls='--', label=smr)

  # refining
  sns.boxplot(ax=ax[1], data=redf, x='Breakeven CAPEX ($/MWe)',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[1].set_ylabel('Refining')
  ax[1].get_legend().set_visible(False)
  for smr in smrs.index :
    ax[1].axvline(smrs.loc[smr, 'CAPEX $/MWe']/1e6, color=palette[smr], ls='--')

  # Steel
  sns.boxplot(ax=ax[2], data=stdf, x='Breakeven CAPEX ($/MWe)',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[2].set_ylabel('Steel')
  ax[2].get_legend().set_visible(False)
  for smr in smrs.index :
    ax[2].axvline(smrs.loc[smr, 'CAPEX $/MWe']/1e6, color=palette[smr], ls='--')

  # Heat
  sns.boxplot(ax=ax[3], data=heat, x='Breakeven CAPEX ($/MWe)',palette=scenariop, hue='scenario', fill=False, width=0.5)
  sns.despine()
  ax[3].set_ylabel('Process\nheat')
  ax[3].get_legend().set_visible(False)
  for smr in smrs.index :
    ax[3].axvline(smrs.loc[smr, 'CAPEX $/MWe']/1e6, color=palette[smr], ls='--')

  #Common legend for whole figure
  h3, l3 = ax[0].get_legend_handles_labels()
  by_label = dict(zip(l3, h3))
  fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1,.5), ncol=1)

  ax[3].set_xlabel('Breakeven CAPEX (M$/MWe)')

  plt.subplots_adjust(wspace=.1,
                    hspace=0.1)

  fig.savefig('./results/capex_sa_be_analysis.png')
  plt.show()


def main():
  amdf, redf, stdf, heat = load_data()
  #plot_comparison(amdf, redf, stdf, heat)
  plot_comparison_margins(amdf, redf, stdf, heat)

if __name__ == '__main__':
  warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
  main()
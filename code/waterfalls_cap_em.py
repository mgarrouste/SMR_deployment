import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ANR_application_comparison
import matplotlib.pyplot as plt
import seaborn as sns
from utils import palette


def load_foak_positive():
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)' ]]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'ANR':'SMR'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
	heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'SMR',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
												'Annual Net Revenues (M$/y)', 'Application']]
	heat_data = heat_data.rename(columns={'Emissions_mmtco2/y':'Emissions','Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)'})
	heat_data['App'] = 'Process Heat'
	heat_data.reset_index(inplace=True, names='id')

	foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
	foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]
	return foak_positive


def load_noak_positive():
	# NOAK data
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)' ]]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'ANR':'SMR'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen')
	heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'SMR',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
												  'Application', 'Annual Net Revenues (M$/y)']]
	heat_data.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)',
													'Emissions_mmtco2/y':'Emissions'}, inplace=True)
	heat_data.reset_index(inplace=True, names='id')
	heat_data['App'] = 'Process Heat'

	noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]

	noak_positive.set_index('id', inplace=True)
	foak_positive = load_foak_positive()
	foak_positive.set_index('id', inplace=True)
	foak_to_drop = foak_positive.index.to_list()
	noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')
	noak_positive = noak_positive.reset_index()
	return noak_positive


def load_noak_noPTC():
	# NOAK data
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Net Revenues ($/year)','Electricity revenues ($/y)' ]]

	h2_data['Annual Net Revenues (M$/y)'] =h2_data.loc[:,['Net Revenues ($/year)','Electricity revenues ($/y)']].sum(axis=1)
	h2_data['Annual Net Revenues (M$/y)'] /=1e6
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'ANR':'SMR'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data = h2_data.drop(columns=['Net Revenues ($/year)','Electricity revenues ($/y)' ])
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=False)
	heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'SMR',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
													'Application', 'Annual Net Revenues (M$/y)']]
	heat_data.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)',
												'Emissions_mmtco2/y':'Emissions'}, inplace=True)
	heat_data.reset_index(inplace=True, names=['id'])
	heat_data['App'] = 'Process Heat'

	# Only profitable facilities
	noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]
	noak_positive.to_excel('./results/results_heat_h2_NOAK_noPTC.xlsx', index=False)
	noak_positive.set_index('id', inplace=True)
	foak_positive = load_foak_positive()
	foak_positive.set_index('id', inplace=True)
	foak_to_drop = foak_positive.index.to_list()
	noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')
	noak_positive = noak_positive.reset_index()
	return noak_positive

foak_positive = load_foak_positive()
noak_positive = load_noak_positive()
noak_noPTC = load_noak_noPTC()


def plot_bars(foak_positive, noak_positive, noak_noPTC):
	df = foak_positive[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df1 = df.reset_index()
	df1 = df1.replace('Industrial Hydrogen-', '', regex=True)
	df1['measure'] = 'relative'
	total_df1 = df1.sum()
	total_df1['App'] = 'Total'
	total_df1['measure'] = 'total'
	df1['tag'] = 'FOAK-cogen'
	total_df1['tag'] = 'FOAK-cogen'

	df = noak_positive[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df2 = df.reset_index()
	df2 = df2.replace('Industrial Hydrogen-', '', regex=True)
	df2['measure'] = 'relative'
	total_df2 = df2.sum()
	total_df2['App'] = 'Total'
	total_df2['measure'] = 'total'
	df2['tag'] = 'NOAK-cogen'
	total_df2['tag'] = 'NOAK-cogen'
	
	df = noak_noPTC[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df3 = df.reset_index()
	df3 = df3.replace('Industrial Hydrogen-', '', regex=True)
	# Differential from NOAK with to without PTC
	df2_forcalc = df2.copy()
	df2_forcalc.set_index('App', inplace=True)
	df3_forcalc = df3.copy()
	df3_forcalc.set_index('App', inplace=True)
	df3_forcalc.at['Ammonia', 'Capacity'] = 0
	df3_forcalc.at['Ammonia', 'Emissions'] = 0
	df3_forcalc.at['Refining', 'Capacity'] = 0
	df3_forcalc.at['Refining', 'Emissions'] = 0
	diffdf = df2_forcalc.copy()
	diffdf['Capacity'] = df3_forcalc["Capacity"] - df2_forcalc['Capacity']
	diffdf['Emissions'] = df3_forcalc["Emissions"] - df2_forcalc['Emissions']
	diffdf = diffdf.reset_index()
	diffdf['measure'] = 'relative'
	total_diffdf = diffdf.sum()
	total_diffdf['App'] = 'Total'
	total_diffdf['measure'] = 'total'
	diffdf['tag'] = 'NOAK-cogen-NoPTC'
	total_diffdf['tag'] = 'NOAK-cogen-NoPTC'


	# Now create a combined DataFrame from df1 and the adjusted df2
	combined_df = pd.concat([df1, pd.DataFrame([total_df1]), df2, pd.DataFrame([total_df2]), diffdf, pd.DataFrame([total_diffdf])], ignore_index=True)
	print(combined_df)

	fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
	combined_df['text_em'] = combined_df.apply(lambda x: int(x['Emissions']), axis=1)
	
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = combined_df['measure'],
		x = [combined_df['tag'],combined_df['App']],
		textposition = "outside",
		text = combined_df['text_em'],
		y = combined_df['Emissions'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "paleGreen"}},
		decreasing = {"marker":{"color": "firebrick"}},
		totals = {"marker":{"color": "limeGreen"}}
		),
		row=1, col=1
	)

	combined_df['text_cap'] = combined_df.apply(lambda x: int(x['Capacity']), axis=1)
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = combined_df['measure'],
		x = [combined_df['tag'],combined_df['App']],
		textposition = "outside",
		text = combined_df['text_cap'],
		y = combined_df['Capacity'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "lightBlue"}},
		decreasing = {"marker":{"color": "darkOrange"}},
		totals = {"marker":{"color": "royalBlue"}}
		),
		row=1, col=2
	)
	# Set y-axis titles
	fig.update_yaxes(title_text='Avoided emissions (MMtCO2/y)', row=1, col=1)
	fig.update_yaxes(title_text='ANR Capacity (GWe)', row=1, col=2)
	fig.update_xaxes(tickangle=53)
	# Set chart layout
	fig.update_layout(
		margin=dict(l=20, r=20, t=20, b=20),
		showlegend = False,
		width=1050,  # Set the width of the figure
		height=550,  # Set the height of the figure
	)
	fig.write_image('./results/waterfall_foak_noak_noaknoPTC_emissions_capacity.png', scale=4)




def abatement_cost_plot():
	save_path = './results/abatement_cost_cogen.png'
	
	fig, ax = plt.subplots(2,1, figsize=(7,4),sharex=True)
	xmin = -50
	xmax = 600

	epa_scc = 230
	rff_scc = 185

	# FOAK on the left
	anr_tag = 'FOAK'
	import pp_industrial_hydrogen
	h2_data = pp_industrial_hydrogen.load_data(anr_tag)
	# Select only profitable sites  !
	h2_data = h2_data[h2_data['Net Annual Revenues with H2 PTC ($/MWe/y)']>=0]
	h2_data['Cost ANR ($/y)'] = h2_data['ANR CAPEX ($/year)']+h2_data['H2 CAPEX ($/year)']+h2_data['ANR O&M ($/year)']+h2_data['H2 O&M ($/year)']\
												+h2_data['Conversion costs ($/year)']-h2_data['Avoided NG costs ($/year)']
	h2_data['Abatement cost ($/tCO2)'] = h2_data['Cost ANR ($/y)']/(h2_data['Ann. avoided CO2 emissions (MMT-CO2/year)']*1e6)
	print('FOAK h2')
	h2_data = h2_data[['ANR type', 'Abatement cost ($/tCO2)', 'Industry']]
	print(h2_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	h2_data = h2_data.rename(columns={'ANR type':'SMR'})
	# Direct heat
	heat_data = pd.read_excel(f'./results/process_heat/best_pathway_{anr_tag}_cogen_PTC_True.xlsx')
	heat_data = heat_data[heat_data['Pathway Net Ann. Rev. (M$/y)']>=0]
	heat_data['Cost ANR ($/y)'] = (heat_data['CAPEX ($/y)']+heat_data['O&M ($/y)']+heat_data['Conversion']-heat_data['Avoided NG Cost ($/y)'])
	heat_data['Abatement cost ($/tCO2)'] = heat_data['Cost ANR ($/y)']/(heat_data['Emissions_mmtco2/y']*1e6)
	heat_data = heat_data[['SMR', 'Abatement cost ($/tCO2)']]
	print('FOAK heat')
	print(heat_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	heat_data['Industry'] = 'Process Heat'

	foak = pd.concat([h2_data, heat_data], ignore_index=True)

	sns.boxplot(ax=ax[0], data=foak, y='Industry', x='Abatement cost ($/tCO2)', color='k', fill=False, width=.4)
	sns.stripplot(ax=ax[0], data=foak,y='Industry', x='Abatement cost ($/tCO2)', hue='SMR', palette=palette, marker='P',alpha=.7)
	ax[0].get_legend().set_visible(False)
	ax[0].set_xlim(xmin, xmax)
	ax[0].set_title('FOAK')
	ax[0].get_legend().set_visible(False)
	ax[0].set_ylabel('')
	ax[0].axvline(x=rff_scc, ls='--', color='orange')
	ax[0].axvline(x=epa_scc, ls='--', color='tomato')
	ax[0].text(x=rff_scc, y=-.4, s='RFF',color='orange')
	ax[0].text(x=epa_scc, y=-.4, s='EPA',color='tomato')
	#ax[0].set_yticks(ax[0].get_yticks(), ax[0].get_yticklabels(), rotation=-30, ha='left')

	# FOAK on the left
	anr_tag = 'NOAK'
	import pp_industrial_hydrogen
	h2_data = pp_industrial_hydrogen.load_data(anr_tag)
	# Select only profitable sites
	h2_data = h2_data[h2_data['Net Annual Revenues with H2 PTC ($/MWe/y)']>=0]
	h2_data['Cost ANR ($/y)'] = h2_data['ANR CAPEX ($/year)']+h2_data['H2 CAPEX ($/year)']+h2_data['ANR O&M ($/year)']+h2_data['H2 O&M ($/year)']\
												+h2_data['Conversion costs ($/year)']-h2_data['Avoided NG costs ($/year)']
	h2_data['Abatement cost ($/tCO2)'] = h2_data['Cost ANR ($/y)']/(h2_data['Ann. avoided CO2 emissions (MMT-CO2/year)']*1e6)
	h2_data = h2_data[['ANR type', 'Abatement cost ($/tCO2)', 'Industry']]
	print('NOAK h2')
	print(h2_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	h2_data = h2_data[['ANR type', 'Abatement cost ($/tCO2)', 'Industry']]
	h2_data = h2_data.rename(columns={'ANR type':'SMR'})
	# Direct heat
	heat_data = pd.read_excel(f'./results/process_heat/best_pathway_{anr_tag}_cogen_PTC_True.xlsx')
	heat_data = heat_data[heat_data['Pathway Net Ann. Rev. (M$/y)']>=0]
	heat_data['Cost ANR ($/y)'] = (heat_data['CAPEX ($/y)']+heat_data['O&M ($/y)']+heat_data['Conversion']-heat_data['Avoided NG Cost ($/y)'])
	heat_data['Abatement cost ($/tCO2)'] = heat_data['Cost ANR ($/y)']/(heat_data['Emissions_mmtco2/y']*1e6)
	heat_data = heat_data[['SMR', 'Abatement cost ($/tCO2)']]
	print('NOAK heat')
	print(heat_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	heat_data['Industry'] = 'Process Heat'

	noak = pd.concat([h2_data, heat_data], ignore_index=True)
	sns.boxplot(ax=ax[1], data=noak,y='Industry', x='Abatement cost ($/tCO2)', color='k', fill=False, width=.4)
	sns.stripplot(ax=ax[1], data=noak, y='Industry', x='Abatement cost ($/tCO2)', hue='SMR', palette=palette, marker='P',alpha=.7)
	ax[1].set_xlim(xmin,xmax)
	ax[1].set_title('NOAK')
	ax[1].get_legend().set_visible(False)
	ax[1].set_ylabel('')
	ax[1].axvline(x=rff_scc, ls='--', color='orange')
	ax[1].axvline(x=epa_scc, ls='--', color='tomato')
	#ax[1].set_yticks(ax[1].get_xticks(), ax[1].get_xticklabels(), rotation=-30, ha='left')

	h3, l3 = ax[0].get_legend_handles_labels()
	h4, l4 = ax[1].get_legend_handles_labels()
	by_label = dict(zip(l3+l4, h3+h4))
	fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(1.,.8),ncol=1)


	sns.despine()
	fig.savefig(save_path, bbox_inches='tight', dpi=500)

	



plot_bars(foak_positive, noak_positive, noak_noPTC)
abatement_cost_plot()
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ANR_application_comparison
import matplotlib.pyplot as plt
import seaborn as sns
from utils import palette

def load_foaknoPTC():
	# profitable FOAK without the H2 PTC
	heat = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=False)
	heat= heat[['STATE','latitude', 'longitude', 'NG price ($/MMBtu)', 'Emissions_mmtco2/y', 'SMR',
											 'Depl. ANR Cap. (MWe)', 'Industry','Annual Net Revenues (M$/y)', 'Application', 'IRR wo PTC']]
	heat = heat[heat['Annual Net Revenues (M$/y)']>0]
	heat.rename(columns={'Emissions_mmtco2/y':'Ann. avoided CO2 emissions (MMT-CO2/year)',
														'NG price ($/MMBtu)':'State price ($/MMBtu)', 'STATE':'state'}, inplace=True)
	heat['application'] = 'Process Heat'
	heat.reset_index(inplace=True, names=['id'])
	print('# process heat facilities profitable wo PTc :{}'.format(len(heat[heat['Annual Net Revenues (M$/y)']>0])))
	print(heat['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	print(heat['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	print(heat['SMR'].unique())


	h2 = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2 = h2.loc[:,~h2.columns.duplicated()]
	h2 = h2.reset_index()
	h2['Annual Net Revenues wo PTC (M$/y)'] = h2['Electricity revenues ($/y)']+h2['Net Revenues ($/year)']
	print('# process hydrogen facilities profitable wo PTc :{}'.format(len(h2[h2['Annual Net Revenues wo PTC (M$/y)']>0])))

	return heat

def load_foak_positive():
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)', 'state' ]]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'ANR':'SMR'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
	heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'SMR', 'STATE',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
												'Annual Net Revenues (M$/y)', 'Application']]
	heat_data = heat_data.rename(columns={'Emissions_mmtco2/y':'Emissions',
																			 'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)', 'STATE':'state'})
	heat_data['App'] = 'Process Heat'
	heat_data.reset_index(inplace=True, names='id')

	foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
	foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]

	foak_noPTC = load_foaknoPTC()
	foak_positive.set_index('id', inplace=True)
	foak_noPTC.set_index('id', inplace=True)
	foak_to_drop = foak_noPTC.index.to_list()
	foak_positive = foak_positive.drop(foak_to_drop, errors='ignore')
	foak_positive = foak_positive.reset_index()
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

	# Drop FOAK sites
	noak_positive.set_index('id', inplace=True)
	foak_positive = load_foak_positive()
	foak_positive.set_index('id', inplace=True)
	foak_to_drop = foak_positive.index.to_list()
	noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')

	# Drop FOAK no PTC sites
	foak_noPTC = load_foaknoPTC()
	foak_noPTC.set_index('id', inplace=True)
	foak_noPTC_todrop = foak_noPTC.index.to_list()
	noak_positive = noak_positive.drop(foak_noPTC_todrop, errors='ignore')

	# Drop NOAK no PTC sites
	noak_noPTC = load_noak_noPTC()
	noak_noPTC.set_index('id', inplace=True)
	noak_noPTC_todrop = noak_noPTC.index.to_list()
	noak_positive = noak_positive.drop(noak_noPTC_todrop, errors='ignore')

	noak_positive = noak_positive.reset_index()
	return noak_positive


def load_noak_noPTC():
	# NOAK data
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Net Revenues ($/year)','Electricity revenues ($/y)', 'IRR wo PTC']]

	h2_data['Annual Net Revenues (M$/y)'] =h2_data.loc[:,['Net Revenues ($/year)','Electricity revenues ($/y)']].sum(axis=1)
	h2_data['Annual Net Revenues (M$/y)'] /=1e6
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'ANR':'SMR', 'IRR wo PTC': 'IRR (%)'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data = h2_data.drop(columns=['Net Revenues ($/year)','Electricity revenues ($/y)' ])
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=False)
	heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'SMR',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
													'Application', 'Annual Net Revenues (M$/y)', 'IRR wo PTC']]
	heat_data.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)',
												'Emissions_mmtco2/y':'Emissions', 'IRR wo PTC': 'IRR (%)'}, inplace=True)
	heat_data.reset_index(inplace=True, names=['id'])
	heat_data['App'] = 'Process Heat'

	# Only profitable facilities
	noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]
	noak_positive['IRR (%)'] *=100
	noak_positive.to_excel('./results/results_heat_h2_NOAK_noPTC.xlsx', index=False)
	noak_positive.set_index('id', inplace=True)

	# Drop FOAK no PTC sites
	foak_noPTC = load_foaknoPTC()
	foak_noPTC.set_index('id', inplace=True)
	foak_noPTC_todrop = foak_noPTC.index.to_list()
	noak_positive = noak_positive.drop(foak_noPTC_todrop, errors='ignore')
	# Drop FOAK sites
	foak_positive = load_foak_positive()
	foak_positive.set_index('id', inplace=True)
	foak_to_drop = foak_positive.index.to_list()
	noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')
	noak_positive = noak_positive.reset_index()
	return noak_positive

foak_noPTC = load_foaknoPTC()
foak_positive = load_foak_positive()
noak_positive = load_noak_positive()
noak_noPTC = load_noak_noPTC()


def plot_bars(foak_noPTC, foak_positive, noak_positive, noak_noPTC):
	df = foak_noPTC[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df0 = df.reset_index()
	df0 = df0.replace('Industrial Hydrogen-', '', regex=True)
	df0['measure'] = 'relative'
	total_df0 = pd.DataFrame({'App':['Total'],'Emissions': [df0['Emissions'].sum()], 'Capacity':[df0['Capacity'].sum()], 'measure':['total'], 'tag':['FOAK-NoPTC']})
	df0['tag'] = 'FOAK<br>NoPTC'

	df = foak_positive[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df1 = df.reset_index()
	df1 = df1.replace('Industrial Hydrogen-', '', regex=True)
	df1['measure'] = 'relative'
	total_df1 = pd.DataFrame({'App':['Total'],'Emissions': [df1['Emissions'].sum()], 'Capacity':[df1['Capacity'].sum()], 'measure':['total'], 'tag':['FOAK']})
	df1['tag'] = 'FOAK'


	df = noak_noPTC[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df2 = df.reset_index()
	df2 = df2.replace('Industrial Hydrogen-', '', regex=True)
	df2['measure'] = 'relative'
	total_df2 = pd.DataFrame({'App':['Total'],'Emissions': [df2['Emissions'].sum()], 'Capacity':[df2['Capacity'].sum()], 'measure':['total'], 'tag':['NOAK-NoPTC']})
	df2['tag'] = 'NOAK<br>NoPTC'


	df = noak_positive[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df3 = df.reset_index()
	df3 = df3.replace('Industrial Hydrogen-', '', regex=True)
	df3['measure'] = 'relative'
	df3.sort_values(by=['Capacity', 'Emissions'], ascending=True, inplace=True)
	total_emissions = df3['Emissions'].sum()+df2['Emissions'].sum()+df1['Emissions'].sum()+df0['Emissions'].sum()
	total_cap = df3['Capacity'].sum()+df2['Capacity'].sum()+df1['Capacity'].sum()+df0['Capacity'].sum()
	total_df3 = pd.DataFrame({'App':['Total'],'Emissions': [total_emissions], 'Capacity':[total_cap], 'measure':['total'], 'tag':['NOAK']})
	df3['tag'] = 'NOAK'


	# Now create a combined DataFrame from df1 and the adjusted df2
	#combined_df = pd.concat([df0, total_df0,df1, total_df1, df3, total_df3,df2, total_df2], ignore_index=True)
	
	# Combined DataFrame from df0, df1,... and using reset_index() to maintain correct order
	combined_df = pd.concat([df0, df1, df2,df3, total_df3], ignore_index=True)
	combined_df = combined_df.replace('Process Heat', 'Process<br>Heat')

	print(combined_df)


	# input to your Waterfall traces to use the sorted combined_df:
	measure = combined_df['measure'].to_list()
	x = [combined_df['tag'].to_list(), combined_df['App'].to_list()]
	emissions = combined_df['Emissions'].to_list()

	fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
	combined_df['text_em'] = combined_df.apply(lambda x: int(x['Emissions']) if x['Emissions']>=1 else round(x['Emissions'],1), axis=1)

	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = measure, 
		x = x,
		textposition = "outside",
		text = combined_df['text_em'],
		y = emissions,
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "paleGreen"}},
		decreasing = {"marker":{"color": "firebrick"}},
		totals = {"marker":{"color": "limeGreen"}}
		),
		row=1, col=1
	)

	combined_df['text_cap'] = combined_df.apply(lambda x: int(x['Capacity']) if x['Capacity']>=1 else round(x['Capacity'],1), axis=1)
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
	fig.update_yaxes(title_text='Avoided emissions (MMtCO2/y)', row=1, col=1, titlefont_color='black')
	fig.update_yaxes(title_text='SMR Capacity (GWe)', row=1, col=2, titlefont_color='black')
	fig.update_xaxes(tickangle=53)
	# Set chart layout
	fig.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		showlegend = False,
		width=1050,  # Set the width of the figure
		height=550,  # Set the height of the figure
		plot_bgcolor='white', 

	)
	fig.update_yaxes(gridcolor='grey', gridwidth=0.3, tickfont_color='black')
	fig.update_xaxes(tickfont_color='black')
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
	#print('FOAK h2')
	h2_data = h2_data[['ANR type', 'Abatement cost ($/tCO2)', 'Industry']]
	#print(h2_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	h2_data = h2_data.rename(columns={'ANR type':'SMR'})
	# Direct heat
	heat_data = pd.read_excel(f'./results/process_heat/best_pathway_{anr_tag}_cogen_PTC_True.xlsx')
	heat_data = heat_data[heat_data['Pathway Net Ann. Rev. (M$/y)']>=0]
	heat_data['Cost ANR ($/y)'] = (heat_data['CAPEX ($/y)']+heat_data['O&M ($/y)']+heat_data['Conversion']-heat_data['Avoided NG Cost ($/y)'])
	heat_data['Abatement cost ($/tCO2)'] = heat_data['Cost ANR ($/y)']/(heat_data['Emissions_mmtco2/y']*1e6)
	heat_data = heat_data[['SMR', 'Abatement cost ($/tCO2)']]
	#print('FOAK heat')
	#print(heat_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	heat_data['Industry'] = 'Process Heat'

	foak = pd.concat([h2_data, heat_data], ignore_index=True)

	sns.boxplot(ax=ax[0], data=foak, y='Industry', x='Abatement cost ($/tCO2)', color='k', fill=False, width=.4)
	sns.stripplot(ax=ax[0], data=foak,y='Industry', x='Abatement cost ($/tCO2)', hue='SMR', palette=palette, marker='P',alpha=.7)
	ax[0].get_legend().set_visible(False)
	ax[0].set_xlim(xmin, xmax)
	ax[0].set_title('FOAK')
	ax[0].get_legend().set_visible(False)
	ax[0].set_ylabel('')
	#ax[0].axvline(x=rff_scc, ls='--', color='orange')
	ax[0].axvline(x=epa_scc, ls='--', color='tomato')
	#ax[0].text(x=rff_scc, y=-.4, s='RFF',color='orange')
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
	#print('NOAK h2')
	#print(h2_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	h2_data = h2_data[['ANR type', 'Abatement cost ($/tCO2)', 'Industry']]
	h2_data = h2_data.rename(columns={'ANR type':'SMR'})
	# Direct heat
	heat_data = pd.read_excel(f'./results/process_heat/best_pathway_{anr_tag}_cogen_PTC_True.xlsx')
	heat_data = heat_data[heat_data['Pathway Net Ann. Rev. (M$/y)']>=0]
	heat_data['Cost ANR ($/y)'] = (heat_data['CAPEX ($/y)']+heat_data['O&M ($/y)']+heat_data['Conversion']-heat_data['Avoided NG Cost ($/y)'])
	heat_data['Abatement cost ($/tCO2)'] = heat_data['Cost ANR ($/y)']/(heat_data['Emissions_mmtco2/y']*1e6)
	heat_data = heat_data[['SMR', 'Abatement cost ($/tCO2)']]
	#print('NOAK heat')
	#print(heat_data['Abatement cost ($/tCO2)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	heat_data['Industry'] = 'Process Heat'

	noak = pd.concat([h2_data, heat_data], ignore_index=True)
	sns.boxplot(ax=ax[1], data=noak,y='Industry', x='Abatement cost ($/tCO2)', color='k', fill=False, width=.4)
	sns.stripplot(ax=ax[1], data=noak, y='Industry', x='Abatement cost ($/tCO2)', hue='SMR', palette=palette, marker='P',alpha=.7)
	ax[1].set_xlim(xmin,xmax)
	ax[1].set_title('NOAK')
	ax[1].get_legend().set_visible(False)
	ax[1].set_ylabel('')
	#ax[1].axvline(x=rff_scc, ls='--', color='orange')
	ax[1].axvline(x=epa_scc, ls='--', color='tomato')
	#ax[1].set_yticks(ax[1].get_xticks(), ax[1].get_xticklabels(), rotation=-30, ha='left')

	h3, l3 = ax[0].get_legend_handles_labels()
	h4, l4 = ax[1].get_legend_handles_labels()
	by_label = dict(zip(l3+l4, h3+h4))
	fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(1.,.8),ncol=1)


	sns.despine()
	fig.savefig(save_path, bbox_inches='tight', dpi=500)

	


def main():
	plot_bars(foak_noPTC=foak_noPTC, foak_positive=foak_positive, noak_positive=noak_positive, noak_noPTC=noak_noPTC)
	abatement_cost_plot()

if __name__ =='__main__':
	main()
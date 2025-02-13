import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ANR_application_comparison
import matplotlib.pyplot as plt
import seaborn as sns
import os, argparse
import numpy as np
from utils import palette

def load_foaknoPTC(printinfo=False):
	# profitable FOAK without the H2 PTC
	heat = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=False)
	heat= heat[['STATE','latitude', 'longitude', 'NG price ($/MMBtu)', 'Emissions_mmtco2/y', 'SMR',
											 'Depl. ANR Cap. (MWe)', 'Industry','Annual Net Revenues (M$/y)', 'Application', 'IRR wo PTC']]
	heat = heat[heat['Annual Net Revenues (M$/y)']>0]
	heat.rename(columns={'Emissions_mmtco2/y':'Emissions',
														'NG price ($/MMBtu)':'State price ($/MMBtu)', 'STATE':'state'}, inplace=True)
	heat['application'] = 'Process Heat'
	heat['App'] = 'Process Heat'
	heat.reset_index(inplace=True, names=['id'])
	if printinfo:
		print('# process heat facilities profitable wo PTc :{}'.format(len(heat[heat['Annual Net Revenues (M$/y)']>0])))
		print(heat['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))
		print(heat['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]))
		print(heat['SMR'].unique())


	h2 = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2 = h2.loc[:,~h2.columns.duplicated()]
	h2 = h2.reset_index()
	h2['Annual Net Revenues wo PTC (M$/y)'] = h2['Electricity revenues ($/y)']+h2['Net Revenues ($/year)']
	if printinfo:	print('# process hydrogen facilities profitable wo PTc :{}'.format(len(h2[h2['Annual Net Revenues wo PTC (M$/y)']>0])))
	heat.to_excel('./results/results_FOAK_noPTC.xlsx')
	return heat

def load_foak_positive(dropnoptc=False):
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)', 'state','IRR w PTC']]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'ANR':'SMR'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
	heat_data = heat_data[['latitude', 'longitude', 'STATE','Emissions_mmtco2/y', 'SMR','Pathway', 'Batch_Temp_degC', 'max_temp_degC', 'Surplus SMR Cap. (MWe)',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)','NG price ($/MMBtu)', 'Electricity revenues ($/y)','Avoided NG Cost ($/y)','H2 PTC',
													'Application', 'IRR w PTC','Annual Net Revenues (M$/y)']]
	heat_data = heat_data.rename(columns={'Emissions_mmtco2/y':'Emissions',
																			 'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)', 'STATE':'state'})
	heat_data['App'] = 'Process Heat'
	heat_data.reset_index(inplace=True, names='id')

	foak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]

	if dropnoptc:
		foak_noPTC = load_foaknoPTC()
		foak_positive.set_index('id', inplace=True)
		foak_noPTC.set_index('id', inplace=True)
		foak_to_drop = foak_noPTC.index.to_list()
		foak_positive = foak_positive.drop(foak_to_drop, errors='ignore')
	foak_positive.to_excel('./results/results_FOAK_PTC.xlsx')
	foak_positive = foak_positive.reset_index()
	return foak_positive


def load_noak_positive(foak_ptc=True, foak_noptc=False):
	# NOAK data
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'state', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)','IRR w PTC']]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 
													'ANR':'SMR',
													'IRR w PTC':'IRR (%)'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen')
	heat_data = heat_data[['latitude', 'longitude', 'STATE', 'Emissions_mmtco2/y', 'SMR','Pathway', 'Batch_Temp_degC', 'max_temp_degC', 'Surplus SMR Cap. (MWe)',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)','NG price ($/MMBtu)', 'Electricity revenues ($/y)',
												'Avoided NG Cost ($/y)','H2 PTC',
													'Application', 'IRR w PTC','Annual Net Revenues (M$/y)']]
	heat_data.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)',
													'Emissions_mmtco2/y':'Emissions', 
													'IRR w PTC':'IRR (%)', 
													'STATE':'state'}, inplace=True)
	heat_data.reset_index(inplace=True, names='id')
	heat_data['App'] = 'Process Heat'

	noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]
	noak_positive['IRR (%)'] *=100

	noak_positive.set_index('id', inplace=True)
	tag='all'
	if foak_ptc:
		# Drop FAOK PTC sites
		tag='foak_ptc'
		foak_positive = load_foak_positive()
		foak_positive.set_index('id', inplace=True)
		foak_to_drop = foak_positive.index.to_list()
		noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')
	elif foak_noptc:
		# Drop FOAK no PTC sites
		tag='foak_noptc'
		foak_noPTC = load_foaknoPTC()
		foak_noPTC.set_index('id', inplace=True)
		foak_noPTC_todrop = foak_noPTC.index.to_list()
		noak_positive = noak_positive.drop(foak_noPTC_todrop, errors='ignore')
	
	noak_positive.to_excel(f'./results/results_NOAK_PTC_{tag}.xlsx')
	noak_positive = noak_positive.reset_index()
	return noak_positive


def load_noak_noPTC(foak_ptc=True, foak_noptc=False):
	# NOAK data
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Net Revenues ($/year)','Electricity revenues ($/y)', 'IRR wo PTC', 'state']]

	h2_data['Annual Net Revenues (M$/y)'] =h2_data.loc[:,['Net Revenues ($/year)','Electricity revenues ($/y)']].sum(axis=1)
	h2_data['Annual Net Revenues (M$/y)'] /=1e6
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'ANR':'SMR', 'IRR wo PTC': 'IRR (%)'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data = h2_data.drop(columns=['Net Revenues ($/year)','Electricity revenues ($/y)' ])
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=False)
	heat_data = heat_data[['latitude', 'longitude', 'STATE','Emissions_mmtco2/y', 'SMR','Pathway', 'Batch_Temp_degC', 'max_temp_degC', 'Surplus SMR Cap. (MWe)',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)', 'NG price ($/MMBtu)',
													'Application', 'Annual Net Revenues (M$/y)', 'Electricity revenues ($/y)','Avoided NG Cost ($/y)','IRR wo PTC']]
	heat_data.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)',
												'Emissions_mmtco2/y':'Emissions', 'IRR wo PTC': 'IRR (%)', 'STATE':'state'}, inplace=True)
	heat_data.reset_index(inplace=True, names=['id'])
	heat_data['App'] = 'Process Heat'

	# Only profitable facilities
	noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]
	noak_positive['IRR (%)'] *=100
	
	noak_positive.set_index('id', inplace=True)
	tag='all'
	if foak_ptc:
		# Drop FAOK PTC sites
		tag='foak_ptc'
		foak_positive = load_foak_positive()
		foak_positive.set_index('id', inplace=True)
		foak_to_drop = foak_positive.index.to_list()
		noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')
	elif foak_noptc:
		# Drop FOAK no PTC sites
		tag='foak_noptc'
		foak_noPTC = load_foaknoPTC()
		foak_noPTC.set_index('id', inplace=True)
		foak_noPTC_todrop = foak_noPTC.index.to_list()
		noak_positive = noak_positive.drop(foak_noPTC_todrop, errors='ignore')
	noak_positive.to_excel(f'./results/results_NOAK_noPTC_{tag}.xlsx')
	noak_positive = noak_positive.reset_index()
	return noak_positive



def get_aggregated_data(dftoagg, tag):
	df = dftoagg[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df = df.reset_index()
	df.sort_values(by=['App'], ascending=False, inplace=True) # so that process heat is first
	df = df.replace('Industrial Hydrogen-', '', regex=True)
	df['measure'] = 'relative'
	df['tag'] = tag
	return df


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
	df1.sort_values(by=['App'], ascending=False, inplace=True) # so that process heat is first
	df1 = df1.replace('Industrial Hydrogen-', '', regex=True)
	df1['measure'] = 'relative'
	total_df1 = pd.DataFrame({'App':['Total'],'Emissions': [df1['Emissions'].sum()], 'Capacity':[df1['Capacity'].sum()], 'measure':['total'], 'tag':['FOAK']})
	df1['tag'] = 'FOAK'


	df = noak_noPTC[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df2 = df.reset_index()
	df2.sort_values(by=['App'], ascending=False, inplace=True) # so that process heat is first
	df2 = df2.replace('Industrial Hydrogen-', '', regex=True)
	df2['measure'] = 'relative'
	total_df2 = pd.DataFrame({'App':['Total'],'Emissions': [df2['Emissions'].sum()], 'Capacity':[df2['Capacity'].sum()], 'measure':['total'], 'tag':['NOAK-NoPTC']})
	df2['tag'] = 'NOAK<br>NoPTC'


	df = noak_positive[['App', 'Emissions', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df3 = df.reset_index()
	df3.sort_values(by=['App'], ascending=False, inplace=True) # so that process heat is first
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
	combined_df = pd.concat([df0, df1, df2, df3], ignore_index=True)#, df2,df3, total_df3], ignore_index=True)
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
	fig.write_image('./results/waterfall_foak_noak_noaknoPTC_emissions_capacity.pdf', scale=4)



def plot_scenarios_waterfall(foak_noPTC, foak_positive, noak_noPTC_foaknoptc, noak_noPTC_foakptc, noak_positive_foaknoptc, noak_positive_foakptc):
	totem_noptc = foak_noPTC['Emissions'].sum()+noak_noPTC_foaknoptc['Emissions'].sum()
	totcap_noptc = foak_noPTC['Capacity'].sum()+noak_noPTC_foaknoptc['Capacity'].sum()
	tot_noptc= pd.DataFrame({'App':['Total'],'Emissions': [totem_noptc], 'Capacity':[totcap_noptc], 'measure':['total'], 'tag':[' ']})
	noptc = pd.concat([foak_noPTC, noak_noPTC_foaknoptc, tot_noptc], ignore_index=True)
	noptc = noptc.replace('Process Heat', 'Process<br>Heat')
	noptc['text_em'] = noptc.apply(lambda x: int(x['Emissions']) if x['Emissions']>=1 else round(x['Emissions'],1), axis=1)
	noptc['text_cap'] = noptc.apply(lambda x: int(x['Capacity']) if x['Capacity']>=1 else round(x['Capacity'],1), axis=1)

	fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.01, shared_yaxes=True, vertical_spacing=0.25, column_widths=[.23,.35,.42],
										 column_titles=['FOAK without the H2 PTC<br>NOAK without the H2 PTC', 'FOAK with the H2 PTC<br>NOAK without the H2 PTC', 
													'FOAK with the H2 PTC<br>NOAK with the H2 PTC'])
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = noptc['measure'],
		x = [noptc['tag'], noptc['App']],
		textposition = "outside",
		text = noptc['text_cap'],
		y = noptc['Capacity'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "lightBlue"}},
		decreasing = {"marker":{"color": "darkOrange"}},
		totals = {"marker":{"color": "royalBlue"}}, 
		showlegend=False
		),
		row=1, col=1
	)
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = noptc['measure'], 
		x = [noptc['tag'], noptc['App']],
		textposition = "outside",
		text = noptc['text_em'],
		y = noptc['Emissions'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "paleGreen"}},
		decreasing = {"marker":{"color": "firebrick"}},
		totals = {"marker":{"color": "limeGreen"}},
		showlegend=False
		),
		row=2, col=1
	)

	# FOAK PTC then NOAK no ptc
	totem_ptcfirst = foak_positive['Emissions'].sum()+noak_noPTC_foakptc['Emissions'].sum()
	totcap_ptcfirst = foak_positive['Capacity'].sum()+noak_noPTC_foakptc['Capacity'].sum()
	tot_ptcfirst= pd.DataFrame({'App':['Total'],'Emissions': [totem_ptcfirst], 'Capacity':[totcap_ptcfirst], 'measure':['total'], 'tag':[' ']})
	ptcfirst = pd.concat([foak_positive, noak_noPTC_foakptc, tot_ptcfirst], ignore_index=True)
	ptcfirst = ptcfirst.replace('Process Heat', 'Process<br>Heat')
	ptcfirst['text_em'] = ptcfirst.apply(lambda x: int(x['Emissions']) if x['Emissions']>=1 else round(x['Emissions'],1), axis=1)
	ptcfirst['text_cap'] = ptcfirst.apply(lambda x: int(x['Capacity']) if x['Capacity']>=1 else round(x['Capacity'],1), axis=1)


	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = ptcfirst['measure'],
		x = [ptcfirst['tag'], ptcfirst['App']],
		textposition = "outside",
		text = ptcfirst['text_cap'],
		y = ptcfirst['Capacity'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "lightBlue"}},
		decreasing = {"marker":{"color": "darkOrange"}},
		totals = {"marker":{"color": "royalBlue"}},
		showlegend=False
		),
		row=1, col=2
	)
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = ptcfirst['measure'], 
		x = [ptcfirst['tag'], ptcfirst['App']],
		textposition = "outside",
		text = ptcfirst['text_em'],
		y = ptcfirst['Emissions'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "paleGreen"}},
		decreasing = {"marker":{"color": "firebrick"}},
		totals = {"marker":{"color": "limeGreen"}},
		showlegend=False
		),
		row=2, col=2
	)

	# FOAK PTC then NOAK with PTC
	totem_ptc = foak_positive['Emissions'].sum()+noak_positive_foakptc['Emissions'].sum()
	totcap_ptc = foak_positive['Capacity'].sum()+noak_positive_foakptc['Capacity'].sum()
	tot_ptc= pd.DataFrame({'App':['Total'],'Emissions': [totem_ptc], 'Capacity':[totcap_ptc], 'measure':['total'], 'tag':[' ']})
	ptc = pd.concat([foak_positive, noak_positive_foakptc, tot_ptc], ignore_index=True)
	ptc = ptc.replace('Process Heat', 'Process<br>Heat')
	ptc['text_em'] = ptc.apply(lambda x: int(x['Emissions']) if x['Emissions']>=1 else round(x['Emissions'],1), axis=1)
	ptc['text_cap'] = ptc.apply(lambda x: int(x['Capacity']) if x['Capacity']>=1 else round(x['Capacity'],1), axis=1)


	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = ptc['measure'],
		x = [ptc['tag'], ptc['App']],
		textposition = "outside",
		text = ptc['text_cap'],
		y = ptc['Capacity'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "lightBlue"}},
		decreasing = {"marker":{"color": "darkOrange"}},
		totals = {"marker":{"color": "royalBlue"}},
		showlegend=False
		),
		row=1, col=3
	)
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = ptc['measure'], 
		x = [ptc['tag'], ptc['App']],
		textposition = "outside",
		text = ptc['text_em'],
		y = ptc['Emissions'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "paleGreen"}},
		decreasing = {"marker":{"color": "firebrick"}},
		totals = {"marker":{"color": "limeGreen"}},
		showlegend=False
		),
		row=2, col=3
	)
	fig.update_yaxes(title_text='Avoided emissions (MMtCO2/y)', row=2, col=1, titlefont_color='black')
	fig.update_yaxes(title_text='SMR Capacity (GWe)', row=1, col=1, titlefont_color='black')
	# Set chart layout
	fig.update_layout(
		margin=dict(l=0, r=0, t=40, b=0),
		showlegend = False,
		width=1120,  # Set the width of the figure
		height=700,  # Set the height of the figure
		plot_bgcolor='white', 
	)
	fig.update_yaxes(gridcolor='grey', gridwidth=0.1, tickfont_color='black' )
	fig.update_yaxes(range=[0, 295], row=1, col=1)
	fig.update_yaxes(range=[0, 245], row=2, col=1)
	fig.update_xaxes(tickfont_color='black')
	fig.update_xaxes(tickangle=56)
	fig.write_image('./results/waterfall_scenarios.pdf', scale=4)
	fig.show()



def abatement_cost_plot():
	save_path = './results/abatement_cost_cogen.pdf'
	
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
	fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(1.,.82),ncol=1)


	sns.despine()
	fig.savefig(save_path, bbox_inches='tight', dpi=500)

	



def cashflow_breakdown_plots(scenario, heat, h2):
	if scenario['OAK']=='NOAK':
		width_ratios = [10,1]
	elif scenario['OAK']=='FOAK' and scenario['PTC']==False:
		width_ratios = [10,1]
	else: width_ratios = [1,1]
	fig, ax = plt.subplots(1,2, figsize=(9,5), width_ratios=width_ratios)
	from utils import cashflows_color_map
	OAK = scenario['OAK']
	with_ptc = scenario['PTC']
	cdf = heat.copy()
	cdf.fillna(0, inplace=True)

	cdf['SMR CAPEX'] = (-cdf[f'Annual_CAPEX_{OAK}']-cdf['Annual ANR CAPEX'])/1e6
	cdf['H2 CAPEX'] = -cdf['Annual H2 CAPEX']/1e6
	cdf['SMR O&M'] = -(cdf[f'FOPEX_{OAK}']+cdf[f'VOPEX_{OAK}'])/1e6-(cdf['ANR VOM']+cdf['ANR FOM'])/1e6
	cdf['H2 O&M'] = -(cdf['H2 VOM']+cdf['H2 FOM'])/1e6
	cdf['Conversion'] = -(cdf['Conversion'])/1e6
	cdf['Avoided Fossil Fuel Costs'] = cdf['Avoided NG Cost ($/y)']/1e6
	cdf['H2 PTC'] = cdf['H2 PTC']/1e6
	cdf['Electricity (cogen)'] = cdf['Electricity revenues ($/y)']/1e6
	cdfheat = cdf.sort_values(by='Annual Net Revenues (M$/y)', ascending=True, ignore_index=True)
	cashflow_list = ['SMR CAPEX','H2 CAPEX','SMR O&M',	'H2 O&M','Conversion','Avoided Fossil Fuel Costs','H2 PTC', 'Electricity (cogen)']


	cdf = h2.copy()

	cdf['SMR CAPEX'] = -cdf['ANR CAPEX ($/year)']/1e6
	cdf['H2 CAPEX'] = -cdf['H2 CAPEX ($/year)']/1e6
	cdf['SMR O&M'] = -cdf['ANR O&M ($/year)']/1e6
	cdf['H2 O&M'] = -cdf['H2 O&M ($/year)']/1e6
	cdf['Conversion'] = -cdf['Conversion costs ($/year)']/1e6
	cdf['Avoided Fossil Fuel Costs'] = cdf['Avoided NG costs ($/year)']/1e6
	cdf['H2 PTC'] = cdf['H2 PTC Revenues ($/year)']/1e6
	cdf['Electricity (cogen)'] = cdf['Electricity revenues ($/y)']/1e6
	cdfh2 = cdf.sort_values(by='Annual Net Revenues (M$/y)', ascending=True)
	
	if len(cdfheat)>0:
		cdfheat[cashflow_list].plot(ax=ax[0], kind='bar', stacked=True, color=cashflows_color_map, width=1)
		ax02 = ax[0].twinx()
		cdfheat[['Annual Net Revenues (M$/y)']].plot(ax=ax02, color='royalblue', marker='+', linestyle='')
		ax[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
		ax[0].set_ylabel('Cashflow (M$/y)')
		ax[0].set_xlabel('Industrial site')
		ax02.get_legend().set_visible(False)
		ax02.tick_params(axis='y', colors='royalblue')
		ax[0].yaxis.grid(True)
		ax[0].get_legend().set_visible(False)
		ax[0].set_title('Process Heat')
	else: ax[0].axis('off')
	if len(cdfh2)>0:
		cdfh2[cashflow_list].plot(ax=ax[1], kind='bar', stacked=True, color=cashflows_color_map, width=1)
		ax12 = ax[1].twinx()
		cdfh2[['Annual Net Revenues (M$/y)']].plot(ax=ax12, color='royalblue', marker='+', linestyle='')
		ax[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
		ax[1].set_xlabel('Industrial site')
		ax12.set_ylabel('Annual Net Revenues (M$/y)', color='royalblue')
		ax12.get_legend().set_visible(False)
		ax12.tick_params(axis='y', colors='royalblue')
		ax[1].yaxis.grid(True)
		ax[1].get_legend().set_visible(False)
		ax[1].set_title('Process Hydrogen')
	else: 
		ax[1].axis('off')
		ax02.set_ylabel('Annual Net Revenues (M$/y)', color='royalblue')
	h00, l00 = ax[0].get_legend_handles_labels()
	h01, l01 = ax[1].get_legend_handles_labels()
	by_label = dict(zip(l00+l01, h00+h01))
	fig.legend(by_label.values(), by_label.keys(),  bbox_to_anchor=(.5,.075),loc='upper center', ncol=4)
	plt.subplots_adjust(wspace=0.25)
	if OAK =='NOAK':
		savepath = './results/cashflows_{}_PTC_{}_FOAK_PTC_{}.pdf'.format(OAK, with_ptc, scenario['FOAK_PTC'])
	else:
		savepath = './results/cashflows_{}_PTC_{}.pdf'.format(OAK, with_ptc)
	fig.savefig(savepath,bbox_inches='tight')

def main():
	foak_noPTC = get_aggregated_data(load_foaknoPTC(), tag='FOAK<br>NoPTC')
	foak_positive = get_aggregated_data(load_foak_positive(dropnoptc=False), tag='FOAK')
	noak_positive_foakptc = get_aggregated_data(load_noak_positive(foak_ptc=True), tag='NOAK')
	noak_positive_foaknoptc = get_aggregated_data(load_noak_positive(foak_ptc=False), tag='NOAK')
	noak_noPTC_foakptc= get_aggregated_data(load_noak_noPTC(foak_ptc=True), tag='NOAK<br>NoPTC')
	noak_noPTC_foaknoptc= get_aggregated_data(load_noak_noPTC(foak_ptc=False, foak_noptc=True), tag='NOAK<br>NoPTC')	
	plot_scenarios_waterfall(foak_noPTC, foak_positive, noak_noPTC_foaknoptc, noak_noPTC_foakptc, noak_positive_foaknoptc, noak_positive_foakptc)

if __name__ =='__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	parser = argparse.ArgumentParser()
	parser.add_argument('-w','--waterfall', required=False, help='Waterfall plot')
	parser.add_argument('-c', '--cashflow', required=False, help='Cashflows plot')
	parser.add_argument('-a', '--abatement', required=False, help='Abatement cost plot')
	args = parser.parse_args()
	if args.waterfall:
		main()
	elif args.abatement:
		abatement_cost_plot()
	elif args.cashflow:
		if args.cashflow == 'FOAK':
			scenario = {'OAK':'FOAK', 'PTC':False}
			heat = ANR_application_comparison.load_heat_results(anr_tag=scenario['OAK'], cogen_tag='cogen', with_PTC=scenario['PTC'])
			heat = heat[heat['Annual Net Revenues (M$/y)']>0]
			h2 = ANR_application_comparison.load_h2_results(anr_tag=scenario['OAK'], cogen_tag='cogen', with_PTC=scenario['PTC'])
			h2 = h2[h2['Annual Net Revenues (M$/y)']>0]
			cashflow_breakdown_plots(scenario=scenario, heat=heat, h2=h2)
			scenario = {'OAK':'FOAK', 'PTC':True}
			heat = ANR_application_comparison.load_heat_results(anr_tag=scenario['OAK'], cogen_tag='cogen', with_PTC=scenario['PTC'])
			heat = heat[heat['Annual Net Revenues (M$/y)']>0]
			h2 = ANR_application_comparison.load_h2_results(anr_tag=scenario['OAK'], cogen_tag='cogen', with_PTC=scenario['PTC'])
			h2 = h2[h2['Annual Net Revenues (M$/y)']>0]
			cashflow_breakdown_plots(scenario=scenario, heat=heat, h2=h2)
		if args.cashflow == 'NOAK':
			## NOAK no ptc after foak no ptc
			scenario = {'OAK':'NOAK', 'PTC':False, 'FOAK_PTC':False}
			# heat
			heatf = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=scenario['FOAK_PTC'])
			heatf = heatf[heatf['Annual Net Revenues (M$/y)']>0]
			heatn = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=scenario['PTC'])
			heatn = heatn[heatn['Annual Net Revenues (M$/y)']>0]
			to_drop = heatf.index.to_list()
			heatn = heatn.drop(to_drop, errors='ignore')
			# h2
			h2f = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=scenario['FOAK_PTC'])
			h2f = h2f[h2f['Annual Net Revenues (M$/y)']>0]
			h2n = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=scenario['PTC'])
			h2n = h2n[h2n['Annual Net Revenues (M$/y)']>0]
			to_drop = h2f.index.to_list()
			h2n = h2n.drop(to_drop, errors='ignore')
			# Plot cashflows
			cashflow_breakdown_plots(scenario=scenario, heat=heatn, h2=h2n)
			## NOAK no ptc after foak ptc
			scenario = {'OAK':'NOAK', 'PTC':False, 'FOAK_PTC':True}
			# heat
			heatf = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=scenario['FOAK_PTC'])
			heatf = heatf[heatf['Annual Net Revenues (M$/y)']>0]
			heatn = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=scenario['PTC'])
			heatn = heatn[heatn['Annual Net Revenues (M$/y)']>0]
			to_drop = heatf.index.to_list()
			heatn = heatn.drop(to_drop, errors='ignore')
			# h2
			h2f = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=scenario['FOAK_PTC'])
			h2f = h2f[h2f['Annual Net Revenues (M$/y)']>0]
			h2n = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=scenario['PTC'])
			h2n = h2n[h2n['Annual Net Revenues (M$/y)']>0]
			to_drop = h2f.index.to_list()
			h2n = h2n.drop(to_drop, errors='ignore')
			# Plot cashflows
			cashflow_breakdown_plots(scenario=scenario, heat=heatn, h2=h2n)
			# NOAK ptc after foak ptc
			scenario = {'OAK':'NOAK', 'PTC':True, 'FOAK_PTC':True}
			# heat
			heatf = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=scenario['FOAK_PTC'])
			heatf = heatf[heatf['Annual Net Revenues (M$/y)']>0]
			heatn = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=scenario['PTC'])
			heatn = heatn[heatn['Annual Net Revenues (M$/y)']>0]
			to_drop = heatf.index.to_list()
			heatn = heatn.drop(to_drop, errors='ignore')
			# h2
			h2f = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=scenario['FOAK_PTC'])
			h2f = h2f[h2f['Annual Net Revenues (M$/y)']>0]
			h2n = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=scenario['PTC'])
			h2n = h2n[h2n['Annual Net Revenues (M$/y)']>0]
			to_drop = h2f.index.to_list()
			h2n = h2n.drop(to_drop, errors='ignore')
			# Plot cashflows
			cashflow_breakdown_plots(scenario=scenario, heat=heatn, h2=h2n)
			

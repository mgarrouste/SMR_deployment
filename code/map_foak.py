import pandas as pd
import plotly.graph_objects as go
from utils import palette
import matplotlib.pyplot as plt
import ANR_application_comparison
from plotly.subplots import make_subplots
import waterfalls_cap_em 


# Create figure
fig = go.Figure()

# List of the state abbreviations you want to color
nuclear_restrictions = ['CA', 'CT', 'VT', 'MA', 'IL', 'OR', 'NJ', 'HI', 'ME', 'RI', 'VT']
nuclear_ban = ['MN', 'NY']
all_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', \
							'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', \
								'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', \
									'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

state_colors = {state: 1 if state in nuclear_restrictions else (2 if state in nuclear_ban else 0) for state in all_states}

# List of index values from state_to_index, corresponding to the custom colorscale
z = [state_colors[state] for state in state_colors.keys()]

# nuclear moratoriums layers
"""
fig.add_trace(go.Choropleth(
		locations=list(state_colors.keys()), # Spatial coordinates
		z=z, # Data to be color-coded (state colors)
		locationmode='USA-states', # Set of locations match entries in `locations`
		showscale=False, # Hide color bar
		colorscale='Reds',
))"""

fig.add_trace(go.Choropleth(
    locationmode='USA-states',
    locations=all_states,  # List of state codes
    z=[1]*len(all_states),  # Dummy variable for coloring
    colorscale=['white', 'white'],  # Set the color scale to white
    showscale=False,  # Hide the color scale
    marker_line_color='grey',  # Set the border color to grey
    marker_line_width=0.7,  # Set the border width
))


def load_foak_positive_2():
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'state', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)', 'IRR w PTC', 'IRR wo PTC' ]]
	h2_data['Emissions_mmtco2/y'] = h2_data['Ann. avoided CO2 emissions (MMT-CO2/year)']
	h2_data.rename(columns={'ANR':'SMR'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data.reset_index(inplace=True)

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
	heat_data = heat_data[['latitude', 'longitude', 'STATE', 'Emissions_mmtco2/y', 'SMR',
												'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
												'Annual Net Revenues (M$/y)', 'Application', 'IRR w PTC', 'IRR wo PTC']]
	heat_data['App'] = 'Process Heat'
	heat_data.rename(columns={'STATE':'state'}, inplace=True)
	heat_data.reset_index(inplace=True)

	foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
	foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]
	return foak_positive

def save_foak_positive():
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2_data = h2_data[['state', 'Depl. ANR Cap. (MWe)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)', 'IRR w PTC']]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions (MMtCO2/y)', 'state':'State', 'ANR':'SMR'}, inplace=True)
	h2_data['application'] = h2_data.apply(lambda x:'H2-'+x['Industry'].capitalize(), axis=1)
	h2_data = h2_data.reset_index(names=['id'])

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
	heat_data = heat_data[['STATE', 'Emissions_mmtco2/y', 'SMR',
												'Depl. ANR Cap. (MWe)', 'Industry',
												'Annual Net Revenues (M$/y)', 'Application', 'IRR w PTC']]
	heat_data.rename(columns={'Emissions_mmtco2/y':'Emissions (MMtCO2/y)', 'STATE':'State'}, inplace=True)
	heat_data['application'] = 'Process Heat'
	heat_data = heat_data.reset_index(names=['id'])

	
	foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
	foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]
	foak_positive.set_index('id', inplace=True)
	foak_positive['Depl. ANR Cap. (MWe)'] = foak_positive['Depl. ANR Cap. (MWe)'].astype(int)
	foak_positive['IRR (%)'] = foak_positive['IRR w PTC']*100
	foak_positive.sort_values(by='IRR (%)', ascending=False)

	foak_noPTC = waterfalls_cap_em.load_foaknoPTC()
	foak_noPTC.set_index('id', inplace=True)
	foak_to_drop = foak_noPTC.index.to_list()
	foak_positive = foak_positive.drop(foak_to_drop, errors='ignore')
	foak_positive = foak_positive.reset_index()


	foak_positive = foak_positive.drop(columns=['Industry', 'Application', 'IRR w PTC'])
	foak_positive.set_index('id', inplace=True)

	foak_positive.to_latex('./results/foak_positive.tex',float_format="{:0.1f}".format, longtable=True, escape=True,\
														label='tab:foak_positive_detailed_results',\
														caption='Detailed results for FOAK deployment stage: Profitable industrial sites and associated SMR capacity deployed and annual revenues')
	return foak_positive



foak_positive = waterfalls_cap_em.load_foak_positive()
foak_positive['IRR (%)'] = foak_positive['IRR w PTC']*100
plot_data = save_foak_positive()
print(foak_positive['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print(foak_positive['IRR (%)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print(foak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print('Micro deployed capacity : ',sum(foak_positive[foak_positive.SMR=='Micro']['Depl. ANR Cap. (MWe)']))
print('Micro deployed units : ',sum(foak_positive[foak_positive.SMR=='Micro']['Depl. ANR Cap. (MWe)'])/6.7)
print('iMSR deployed capacity : ',sum(foak_positive[foak_positive.SMR=='iMSR']['Depl. ANR Cap. (MWe)']))
print('iMSR deployed units : ',sum(foak_positive[foak_positive.SMR=='iMSR']['Depl. ANR Cap. (MWe)'])/141)
print('PBR-HTGR deployed capacity : ',sum(foak_positive[foak_positive.SMR=='PBR-HTGR']['Depl. ANR Cap. (MWe)']))
print('PBR-HTGR deployed units: ',sum(foak_positive[foak_positive.SMR=='PBR-HTGR']['Depl. ANR Cap. (MWe)'])/80)
print('iPWR deployed capacity : ',sum(foak_positive[foak_positive.SMR=='iPWR']['Depl. ANR Cap. (MWe)']))
print('iPWR deployed units : ',sum(foak_positive[foak_positive.SMR=='iPWR']['Depl. ANR Cap. (MWe)'])/77)
print('Total capacity deployed GWe : ', sum(foak_positive['Depl. ANR Cap. (MWe)'])/1e3)
processheat = foak_positive[foak_positive.Application=='Process Heat']
processh2 = foak_positive[foak_positive.Application!='Process Heat']
print('Process heat capacity: ', sum(processheat['Depl. ANR Cap. (MWe)'])/1e3 )
print('Process heat SMR-H2 capacity: ', sum(processheat[processheat.Pathway =='SMR-H2']['Depl. ANR Cap. (MWe)'])/1e3 )
print('Process heat SMR+SMR-H2 capacity: ', sum(processheat[processheat.Pathway =='SMR+SMR-H2']['Depl. ANR Cap. (MWe)'])/1e3 )
print('H2 AMmonia: ', sum(foak_positive[foak_positive.App=='Industrial Hydrogen-Ammonia']['Depl. ANR Cap. (MWe)'])/1e3 )
print('H2 Steel: ', sum(foak_positive[foak_positive.App=='Industrial Hydrogen-Steel']['Depl. ANR Cap. (MWe)'])/1e3 )
print('H2 Refining: ', sum(foak_positive[foak_positive.App=='Industrial Hydrogen-Refining']['Depl. ANR Cap. (MWe)'])/1e3 )



print('/n IRR')
print(foak_positive['IRR (%)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print('\n Heat')
print(processheat['IRR (%)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print('\n H2')
print(processh2['IRR (%)'].describe(percentiles=[.1,.25,.5,.75,.9]))

print('\n Deployment in states with bans')
banss = foak_positive[foak_positive.state.isin(nuclear_ban)]
print('Ban % capacity : ', 100*sum(banss['Depl. ANR Cap. (MWe)'])/sum(foak_positive['Depl. ANR Cap. (MWe)']))
restrs = foak_positive[foak_positive.state.isin(nuclear_restrictions)]
print('Restrictions % capacity : ', 100*sum(restrs['Depl. ANR Cap. (MWe)'])/sum(foak_positive['Depl. ANR Cap. (MWe)']))
# Size based on capacity deployed
percentiles =  foak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]).to_frame()


def plot_irr(data, save_path, app_col='application'):
	import seaborn as sns
	fig, ax = plt.subplots(figsize=(5,3))
	print(save_path)
	sns.kdeplot(ax=ax, data=data, x='IRR (%)', cumulative=True, hue=app_col, common_norm=False, common_grid=True)
	#sns.stripplot(ax=ax, data=data, x='IRR (%)', y='application', palette=palette, hue='SMR', alpha=0.6)
	#sns.boxplot(ax=ax, data=data, x='IRR (%)', y='application', color='black',fill=False, width=0.5)
	sns.despine()
	#ax.set_ylabel('')
	ax.get_legend().set_title('Application')
	#ax.xaxis.set_ticks(np.arange(-1, 1, 0.25))
	ax.xaxis.grid(True)
	#handles, labels = ax.get_legend_handles_labels()
	#fig.legend(handles, labels,  bbox_to_anchor=(1.15,1), ncol=1)
	fig.tight_layout()
	fig.savefig(save_path, bbox_inches='tight')

plot_irr(plot_data, save_path = './results/IRR_foak.png')
def set_size(cap):
	if cap <= 150:
		size = 10
	elif cap <= 500:
		size = 17
	elif cap<=750:
		size = 25
	else:
		size = 35

	return size

foak_positive['size'] = foak_positive['Depl. ANR Cap. (MWe)'].apply(set_size)

def plot_waterfall(foak_positive):
	df = foak_positive[['App', 'Emissions_mmtco2/y', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Emissions_mmtco2/y':'Emissions', 'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df = df.reset_index()
	total_emissions = df['Emissions'].sum()
	total_capacity = df['Capacity'].sum()
	total_row = pd.DataFrame({'App': ['FOAK-Total'], 'Emissions': [total_emissions], 'Capacity':[total_capacity]})
	df = pd.concat([df, total_row], ignore_index=True)

	fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18)


	# Get measures list with all "relative" and the last one as "total"
	measures = ["relative"] * (len(df['Emissions']) - 1) + ["total"]
	df['text_em'] = df.apply(lambda x: int(x['Emissions']), axis=1)
	
	# Create waterfall chart
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = measures,
		x = df['App'],
		textposition = "outside",
		text = df['text_em'],
		y = df['Emissions'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "paleGreen"}},
		totals = {"marker":{"color": "limeGreen"}}
		),
		row=1, col=1
	)
	# Get measures list with all "relative" and the last one as "total"
	measures = ["relative"] * (len(df['Capacity']) - 1) + ["total"]
	df['text_cap'] = df.apply(lambda x: int(x['Capacity']), axis=1)
	# Create waterfall chart
	fig.add_trace(go.Waterfall(
		orientation = "v",
		measure = measures,
		x = df['App'],
		textposition = "outside",
		text = df['text_cap'],
		y = df['Capacity'],
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
		increasing = {"marker":{"color": "lightBlue"}},
		totals = {"marker":{"color": "royalBlue"}}
		),
		row=1, col=2
	)
	# Set y-axis titles
	fig.update_yaxes(title_text='Avoided emissions (MMtCO2/y)', row=1, col=1)
	fig.update_yaxes(title_text='ANR Capacity (GWe)', row=1, col=2)
	fig.update_xaxes(tickangle=270)
	# Set chart layout
	fig.update_layout(
		margin=dict(l=20, r=20, t=20, b=20),
		showlegend = False,
		width=400,  # Set the width of the figure
		height=550,  # Set the height of the figure
	)

	fig.write_image('./results/foak_cogen_positive_emissions_capacity.png')

#plot_waterfall(foak_positive)

scaler = 0.02

		
# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle'}
marker_symbols = foak_positive['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in foak_positive['SMR']]

foak_positive = foak_positive.sort_values(by=['Application'], ascending=False)
sup = foak_positive[foak_positive['IRR (%)'] >=36]
foak_positive = foak_positive[foak_positive['IRR (%)']<36]

fig.add_trace(go.Scattergeo(
		lon=foak_positive['longitude'],
		lat=foak_positive['latitude'],
		mode='markers',
		marker=dict(
				size=foak_positive['size'],
				color=foak_positive['IRR (%)'],
				colorscale='Greys',
				colorbar = dict(
						title='IRR (%)',
						titlefont = dict(size=16),
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.15,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.8,  # Length of the colorbar (80% of figure width)
						tickvals = [6,10,35],
						ticktext = [6,10,35],
						tickmode='array',
						tickfont=dict(size=16)
				),
				symbol=marker_symbols,
				line_color=line_colors,
				line_width=2,
				sizemode='diameter'
		),
		showlegend=False
))


fig.add_trace(go.Scattergeo(
		lon=sup['longitude'],
		lat=sup['latitude'],
		mode='markers',
		marker=dict(
				size=sup['size'],
				color='black',
				symbol=marker_symbols,
				line_color=line_colors,
				line_width=2,
				sizemode='diameter'
		),
		showlegend=False
))
# Create custom legend
custom_legend = {'iMSR - Process Heat':[palette['iMSR'], 'cross'],
								 #'HTGR - Process Heat':[palette['HTGR'], 'cross'],
								 'iPWR - Process Heat':[palette['iPWR'], 'cross'],
								 'PBR-HTGR - Process Heat':[palette['PBR-HTGR'], 'cross'],
								 #'Micro - Process Heat':[palette['Micro'], 'cross'],
								 'iMSR - Industrial H2':[palette['iMSR'], 'circle'],
								 #'HTGR - Industrial H2':[palette['HTGR'], 'circle'],
								 #'iPWR - Industrial H2':[palette['iPWR'], 'circle'],
								 'PBR-HTGR - Industrial H2':[palette['PBR-HTGR'], 'circle'],
								 'Micro - Industrial H2':[palette['Micro'], 'circle']}

reactors_used = foak_positive['SMR'].unique()

# Create symbol and color legend traces
for name, cm in custom_legend.items():
		reactor = name.split(' - ')[0].strip()
		if reactor in reactors_used:
			fig.add_trace(go.Scattergeo(
					lon=[None],
					lat=[None],
					marker=dict(
							size=15,
							color='white',
							line_color=cm[0],
							line_width=4,
							symbol=cm[1]
					),
					name=name
			))


# Custom legend for size
sizes = foak_positive['size'].unique()
sizes.sort()
perc_cap = ['<100 MWe', '100-500 MWe', '>500 MWe']

for size, cap in zip(sizes, perc_cap):
	fig.add_trace(go.Scattergeo(
					lon=[None],
					lat=[None],
					marker=dict(
							size=size,
							color='white',
							line_color='black',
							line_width=1,
							symbol='circle'
					),
					name=cap
			))



# Update layout
fig.update_layout(
		geo=dict(
				scope='usa',
				projection_type='albers usa',
				showlakes=True,
				lakecolor='rgb(255, 255, 255)',
		),
		width=1200,  # Set the width of the figure
		height=600,  # Set the height of the figure
		margin=go.layout.Margin(
				l=20,  # left margin
				r=20,  # right margin
				b=20,  # bottom margin
				t=20  # top margin
		),
		legend=dict(
				x=0.90,
				y=1,
				traceorder="normal",
				font = dict(size = 16, color = "black"),
				bgcolor="rgba(255, 255, 255, 0.5)"  # semi-transparent background
		),
)

# Save
fig.write_image('./results/map_FOAK_cogen.pdf', scale=4)


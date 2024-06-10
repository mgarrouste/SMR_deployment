import pandas as pd
import plotly.graph_objects as go
from utils import palette, app_palette
from plotly.subplots import make_subplots
import ANR_application_comparison

# Create figure
fig = go.Figure()


# Nuclear moratoriums layers
nuclear_restrictions = ['CA', 'CT', 'VT', 'MA', 'IL', 'OR', 'NJ', 'HI', 'ME', 'RI', 'VT']
nuclear_ban = ['MN', 'NY']
all_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', \
							'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', \
								'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', \
									'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

state_colors = {state: 1 if state in nuclear_restrictions else (2 if state in nuclear_ban else 0) for state in all_states}
z = [state_colors[state] for state in state_colors.keys()]

fig.add_trace(go.Choropleth(
		locations=list(state_colors.keys()), # Spatial coordinates
		z=z, # Data to be color-coded (state colors)
		locationmode='USA-states', # Set of locations match entries in `locations`
		showscale=False, # Hide color bar
		colorscale='Reds',
))

import waterfalls_cap_em

noak_positive = waterfalls_cap_em.load_noak_noPTC()


def save_noak_noPTC():
	# NOAK data
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
	h2_data = h2_data[['state', 'Depl. ANR Cap. (MWe)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Net Revenues ($/year)','Electricity revenues ($/y)' ]]

	h2_data['Annual Net Revenues (M$/y)'] =h2_data.loc[:,['Net Revenues ($/year)','Electricity revenues ($/y)']].sum(axis=1)
	h2_data['Annual Net Revenues (M$/y)'] /=1e6
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions (MMtCO2/y)', 'state':'State'}, inplace=True)
	h2_data = h2_data.drop(columns=['Net Revenues ($/year)','Electricity revenues ($/y)'])
	h2_data['application'] = h2_data.apply(lambda x:'H2-'+x['Industry'].capitalize(), axis=1)

	h2_data = h2_data.reset_index(names=['id'])

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen', with_PTC=False)
	heat_data = heat_data[['STATE', 'Emissions_mmtco2/y', 'ANR',
												'Depl. ANR Cap. (MWe)', 'Industry',
													'Application', 'Annual Net Revenues (M$/y)']]
	heat_data['application'] = 'Process Heat'
	heat_data.rename(columns={'Emissions_mmtco2/y':'Emissions (MMtCO2/y)', 'STATE':'State'}, inplace=True)
	heat_data = heat_data.reset_index(names=['id'])

	noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]
	noak_positive.set_index('id', inplace=True)
	foak_positive = waterfalls_cap_em.load_foak_positive()
	foak_positive.set_index('id', inplace=True)
	foak_to_drop = foak_positive.index.to_list()
	noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')
	noak_positive['Depl. ANR Cap. (MWe)'] = noak_positive['Depl. ANR Cap. (MWe)'].astype(int)
	noak_positive = noak_positive.drop(columns=['Industry', 'Application'])
	noak_positive.to_latex('./results/noak_noPTC_positive.tex',float_format="{:0.1f}".format, longtable=True, escape=True,\
                            label='tab:noak_noPTC_positive_detailed_results',\
														caption='Detailed results for NOAK without the H2 PTC deployment stage: Profitable industrial sites and associated SMR capacity deployed and annual revenues')


save_noak_noPTC()


# Size based on capacity deployed
percentiles =  noak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]).to_frame()
print(noak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print('Micro deployed capacity : ',sum(noak_positive[noak_positive.ANR=='Micro']['Depl. ANR Cap. (MWe)']))
print('Micro deployed units : ',sum(noak_positive[noak_positive.ANR=='Micro']['Depl. ANR Cap. (MWe)'])/6.7)
print('iMSR deployed capacity : ',sum(noak_positive[noak_positive.ANR=='iMSR']['Depl. ANR Cap. (MWe)']))
print('iMSR deployed units : ',sum(noak_positive[noak_positive.ANR=='iMSR']['Depl. ANR Cap. (MWe)'])/141)
print('PBR-HTGR deployed capacity : ',sum(noak_positive[noak_positive.ANR=='PBR-HTGR']['Depl. ANR Cap. (MWe)']))
print('PBR-HTGR deployed units: ',sum(noak_positive[noak_positive.ANR=='PBR-HTGR']['Depl. ANR Cap. (MWe)'])/80)
print('iPWR deployed capacity : ',sum(noak_positive[noak_positive.ANR=='iPWR']['Depl. ANR Cap. (MWe)']))
print('iPWR deployed units : ',sum(noak_positive[noak_positive.ANR=='iPWR']['Depl. ANR Cap. (MWe)'])/77)
print('Total capacity deployed GWe : ', sum(noak_positive['Depl. ANR Cap. (MWe)'])/1e3)

scaler = 30

# Set marker symbol based on the application's type
markers_applications = {'Industrial Hydrogen':'circle','Process Heat':'cross' }
marker_symbols = noak_positive['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in noak_positive['ANR']]

# size based on deployed capacity
def set_size(cap):
	if cap <= 200:
		size = 8
	elif cap <= 500:
		size = 15
	else:
		size = 40
	return size

noak_positive['size'] = noak_positive['Depl. ANR Cap. (MWe)'].apply(set_size)


print(noak_positive['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))
max_rev = 5.1
noak_above90th = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >max_rev]
noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] <= max_rev]


fig.add_trace(go.Scattergeo(
		lon=noak_positive['longitude'],
		lat=noak_positive['latitude'],
		text="Capacity: " + noak_positive['Depl. ANR Cap. (MWe)'].astype(str) + " MWe",
		mode='markers',
		marker=dict(
				size=noak_positive['size'],
				color=noak_positive['Annual Net Revenues (M$/y)'],
				colorscale='Greys',
				colorbar = dict(
						title='Annual Net Revenues (M$/y)',
						titlefont = dict(size=16),
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.1,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.7,  # Length of the colorbar (80% of figure width)
						tickvals = [0.4,2.1,5, 6.9],
						ticktext = [0.4,2.1,5, 6.9],
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

# Add points above 90th percentile separately
fig.add_trace(go.Scattergeo(
		lon=noak_above90th['longitude'],
		lat=noak_above90th['latitude'],
		text="Capacity: " + noak_positive['Depl. ANR Cap. (MWe)'].astype(str) + " MWe",
		mode='markers',
		marker=dict(
				size=noak_above90th['size'],
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
								 #'HTGR - Process Heat':[palette['HTGR'], 'square'],
								 'iPWR - Process Heat':[palette['iPWR'], 'cross'],
								 'PBR-HTGR - Process Heat':[palette['PBR-HTGR'], 'cross'],
								 #'Micro - Process Heat':[palette['Micro'], 'cross'],
								 'iMSR - Industrial H2':[palette['iMSR'], 'circle'],
								 #'HTGR - Industrial H2':[palette['HTGR'], 'circle'],
								 #'iPWR - Industrial H2':[palette['iPWR'], 'circle'],
								 'PBR-HTGR - Industrial H2':[palette['PBR-HTGR'], 'circle'],
								 #'Micro - Industrial H2':[palette['Micro'], 'circle']
								 }

reactors_used = noak_positive['ANR'].unique()

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
sizes = noak_positive['size'].unique()
sizes.sort()
perc_cap = ['<200 MWe', '200-500 MWe', '>500 MWe']

for size, cap in zip(sizes, perc_cap):
	fig.add_trace(go.Scattergeo(
					lon=[None],
					lat=[None],
					marker=dict(
							size=size*0.9,
							color='white',
							line_color='black',
							line_width=1,
							symbol='circle'
					),
					name=cap
			))


nuclear_legend = {'Nuclear ban':'darkRed', 
									'Nuclear restrictions':'salmon'}
for b, color in nuclear_legend.items():
	fig.add_trace(go.Scattergeo(
			lon=[None],
			lat=[None],
			marker=dict(
					size=15,
					color=color,
					symbol='square',
			),
			name=b
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
				x=1,
				y=1,
				traceorder="normal",
				font = dict(size = 16, color = "black"),
				bgcolor="rgba(255, 255, 255, 0.5)"  # semi-transparent background
		),
)

# Save
fig.write_image('./results/map_NOAK_cogen_noPTC.png', scale=4)
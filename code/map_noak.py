import pandas as pd
import plotly.graph_objects as go
from utils import palette
from plotly.subplots import make_subplots
import ANR_application_comparison


tag = 'foak_ptc'
# If tag is 'all' all locations profitable at NOAK with PTC are on the map
# if tag is 'foak_ptc' plot only additional profitable locations compared to foak with ptc
if tag =='all':
	foak_ptc, foak_noptc = False, False
elif tag =='foak_ptc':
	foak_ptc, foak_noptc = True, False
# FOAK no ptc not possible since NOAK ptc in this script



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


import waterfalls_cap_em
def save_noak_positive(tag):
	# NOAK data
	h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
	h2_data = h2_data[['state', 'Depl. ANR Cap. (MWe)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)', 'IRR w PTC']]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions (MMtCO2/y)', 'state':'State', 'ANR':'SMR'}, inplace=True)
	h2_data['application'] = h2_data.apply(lambda x:'H2-'+x['Industry'].capitalize(), axis=1)
	h2_data = h2_data.reset_index(names=['id'])

	heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen')
	heat_data = heat_data[['STATE', 'Emissions_mmtco2/y', 'SMR',
												'Depl. ANR Cap. (MWe)', 'Industry', 'IRR w PTC',
												  'Application', 'Annual Net Revenues (M$/y)']]
	heat_data.rename(columns={'Emissions_mmtco2/y':'Emissions (MMtCO2/y)', 'STATE':'State'}, inplace=True)
	heat_data['application'] = 'Process Heat'
	heat_data = heat_data.reset_index(names=['id'])

	noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
	noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]
	noak_positive.set_index('id', inplace=True)
	foak_positive = waterfalls_cap_em.load_foak_positive()
	foak_positive.set_index('id', inplace=True)
	foak_to_drop = foak_positive.index.to_list()
	noak_positive = noak_positive.drop(foak_to_drop, errors='ignore')
	noak_positive['Depl. ANR Cap. (MWe)'] = noak_positive['Depl. ANR Cap. (MWe)'].astype(int)

	noak_positive['IRR (%)'] = noak_positive['IRR w PTC']*100
	noak_positive = noak_positive.drop(columns=['Industry', 'Application', 'IRR w PTC'])
	noak_positive.to_latex(f'./results/noak_positive_{tag}.tex',float_format="{:0.1f}".format, longtable=True, escape=True,\
                            label='tab:noak_positive_detailed_results',\
														caption='Detailed results for NOAK deployment stage: Profitable industrial sites and associated SMR capacity deployed and annual revenues')
	return noak_positive




noak_positive = waterfalls_cap_em.load_noak_positive(foak_ptc=foak_ptc, foak_noptc=foak_noptc)
plot_data = save_noak_positive(tag=tag)
from map_foak import plot_irr
plot_irr(plot_data, save_path=f'./results/IRR_noak_{tag}.png')

# Size based on capacity deployed
percentiles =  noak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]).to_frame()


print(f'Case NOAK positive :{tag}')
print(noak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print('Micro deployed capacity : ',sum(noak_positive[noak_positive.SMR=='Micro']['Depl. ANR Cap. (MWe)']))
print('Micro deployed units : ',sum(noak_positive[noak_positive.SMR=='Micro']['Depl. ANR Cap. (MWe)'])/6.7)
print('iMSR deployed capacity : ',sum(noak_positive[noak_positive.SMR=='iMSR']['Depl. ANR Cap. (MWe)']))
print('iMSR deployed units : ',sum(noak_positive[noak_positive.SMR=='iMSR']['Depl. ANR Cap. (MWe)'])/141)
print('PBR-HTGR deployed capacity : ',sum(noak_positive[noak_positive.SMR=='PBR-HTGR']['Depl. ANR Cap. (MWe)']))
print('PBR-HTGR deployed units: ',sum(noak_positive[noak_positive.SMR=='PBR-HTGR']['Depl. ANR Cap. (MWe)'])/80)
print('iPWR deployed capacity : ',sum(noak_positive[noak_positive.SMR=='iPWR']['Depl. ANR Cap. (MWe)']))
print('iPWR deployed units : ',sum(noak_positive[noak_positive.SMR=='iPWR']['Depl. ANR Cap. (MWe)'])/77)
print('Total capacity deployed GWe : ', sum(noak_positive['Depl. ANR Cap. (MWe)'])/1e3)
processheat = noak_positive[noak_positive.Application=='Process Heat']
print('Process heat capacity: ', sum(processheat['Depl. ANR Cap. (MWe)'])/1e3 )
print('Process heat SMR-H2 capacity: ', sum(processheat[processheat.Pathway =='SMR-H2']['Depl. ANR Cap. (MWe)'])/1e3 )
print('Process heat SMR+SMR-H2 capacity: ', sum(processheat[processheat.Pathway =='SMR+SMR-H2']['Depl. ANR Cap. (MWe)'])/1e3 )
print('/n REvenues and IRR')
print(noak_positive['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))
heat = noak_positive[noak_positive.Application=='Process Heat']
print('\n Heat')
print(heat['IRR (%)'].describe(percentiles=[.1,.25,.5,.75,.9]))
print('\n H2')
processh2 = noak_positive[noak_positive.Application!='Process Heat']
print(processh2['IRR (%)'].describe(percentiles=[.1,.25,.5,.75,.9]))

scaler = 30

# Set marker symbol based on the application's type
markers_applications = {'Industrial Hydrogen':'circle','Process Heat':'cross' }
marker_symbols = noak_positive['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in noak_positive['SMR']]

# size based on deployed capacity
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
noak_positive['size'] = noak_positive['Depl. ANR Cap. (MWe)'].apply(set_size)

fig.add_trace(go.Scattergeo(
		lon=noak_positive['longitude'],
		lat=noak_positive['latitude'],
		mode='markers',
		marker=dict(
				size=noak_positive['size'],
				color=noak_positive['IRR (%)'],
				colorscale='Greys',
				colorbar = dict(
						title='IRR (%)',
						titlefont = dict(size=16),
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.1,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.7,  # Length of the colorbar (80% of figure width)
						tickvals = [9,12,13],
						ticktext = [9,12,13],
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




# Create custom legend
custom_legend = {'iMSR - Process Heat':[palette['iMSR'], 'cross'],
								 #'HTGR - Process Heat':[palette['HTGR'], 'square'],
								 #'iPWR - Process Heat':[palette['iPWR'], 'cross'],
								 #'PBR-HTGR - Process Heat':[palette['PBR-HTGR'], 'cross'],
								 #'Micro - Process Heat':[palette['Micro'], 'cross'],
								 'iMSR - Industrial H2':[palette['iMSR'], 'circle'],
								 #'HTGR - Industrial H2':[palette['HTGR'], 'circle'],
								 #'iPWR - Industrial H2':[palette['iPWR'], 'circle'],
								 'PBR-HTGR - Industrial H2':[palette['PBR-HTGR'], 'circle'],
								 #'Micro - Industrial H2':[palette['Micro'], 'circle']
								 }

reactors_used = noak_positive['SMR'].unique()

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
perc_cap = ['<100 MWe', '100-500 MWe', '>500 MWe']


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

# Create symbol and color legend traces
"""
for anr, color in palette.items():
		fig.add_trace(go.Scattergeo(
				lon=[None],
				lat=[None],
				marker=dict(
						size=15,
						color='white',
						line_color=color,
						line_width=5,
				),
				name=anr
		))

# Create symbol and color legend traces
for app, marker in markers_applications.items():
		fig.add_trace(go.Scattergeo(
				lon=[None],
				lat=[None],
				marker=dict(
						size=15,
						color='white',
						symbol=marker,
						line_color='black',
						line_width=2,
				),
				name=app
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
"""

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
fig.write_image(f'./results/map_NOAK_cogen_{tag}.pdf', scale=4)
# Show figure


### Plot waterfalls

#foak_positive = map_foak.load_foak_positive()

def plot_bars(foak_positive, noak_positive):
	df = foak_positive[['App', 'Emissions_mmtco2/y', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Emissions_mmtco2/y':'Emissions', 'Depl. ANR Cap. (MWe)':'Capacity'})
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

	df = noak_positive[['App', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions', 'Depl. ANR Cap. (MWe)':'Capacity'})
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
	df2_adjusted = pd.concat([pd.DataFrame([total_df1]), df2, pd.DataFrame([total_df2])], ignore_index=True)
	
	# Now create a combined DataFrame from df1 and the adjusted df2
	combined_df = pd.concat([df1, df2_adjusted], ignore_index=True)
	print(combined_df)

	fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13)
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
    totals = {"marker":{"color": "royalBlue"}}
		),
		row=1, col=2
	)
	# Set y-axis titles
	fig.update_yaxes(title_text='Avoided emissions (MMtCO2/y)', row=1, col=1)
	fig.update_yaxes(title_text='SMR Capacity (GWe)', row=1, col=2)
	fig.update_xaxes(tickangle=45)
	# Set chart layout
	fig.update_layout(
		margin=dict(l=20, r=20, t=20, b=20),
		showlegend = False,
		width=600,  # Set the width of the figure
		height=550,  # Set the height of the figure
	)

	fig.write_image('./results/noak_cogen_positive_emissions_capacity.png')




	
#plot_bars(foak_positive, noak_positive)
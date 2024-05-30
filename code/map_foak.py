import pandas as pd
import plotly.graph_objects as go
from utils import palette, app_palette
import matplotlib.pyplot as plt
import ANR_application_comparison
from plotly.subplots import make_subplots
import numpy as np


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
fig.add_trace(go.Choropleth(
		locations=list(state_colors.keys()), # Spatial coordinates
		z=z, # Data to be color-coded (state colors)
		locationmode='USA-states', # Set of locations match entries in `locations`
		showscale=False, # Hide color bar
		colorscale='Reds',
))

h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
									 'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/y)' ]]
h2_data['Emissions_mmtco2/y'] = h2_data['Ann. avoided CO2 emissions (MMT-CO2/year)']
h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'ANR',
											 'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
											 'Annual Net Revenues (M$/y)', 'Application']]
heat_data['App'] = 'Process Heat'
heat_data.reset_index(inplace=True)

foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]
print(foak_positive['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))

# Size based on capacity deployed
percentiles =  foak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]).to_frame()

def set_size(cap):
	if cap <= 25:
		size = 5
	elif cap <= 100:
		size = 10
	elif cap <= 500:
		size = 25
	else:
		size = 40
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

plot_waterfall(foak_positive)

scaler = 0.02

		
# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle'}
marker_symbols = foak_positive['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in foak_positive['ANR']]

max_rev = 510
foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] <= max_rev]

fig.add_trace(go.Scattergeo(
		lon=foak_positive['longitude'],
		lat=foak_positive['latitude'],
		text="Capacity: " + foak_positive['Depl. ANR Cap. (MWe)'].astype(str) + " MWe",
		mode='markers',
		marker=dict(
				size=foak_positive['size'],
				color=foak_positive['Annual Net Revenues (M$/y)'],
				colorscale='Greys',
				colorbar = dict(
						title='Annual Net Revenues (M$/y)',
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.1,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.8,  # Length of the colorbar (80% of figure width)
						tickvals = [1,10,100,250,500],
						ticktext = [1,10,100,250,500],
						tickmode='array'
				),
				symbol=marker_symbols,
				line_color=line_colors,
				line_width=3,
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

reactors_used = foak_positive['ANR'].unique()

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
perc_cap = ['<25 MWe', '25-100 MWe', '100-500 MWe', '>500 MWe']

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
				title="<b>Application & ANR</b>",
				x=1,
				y=1,
				traceorder="normal",
				bgcolor="rgba(255, 255, 255, 0.5)"  # semi-transparent background
		),
)

# Save
fig.write_image('./results/map_FOAK_cogen.png')


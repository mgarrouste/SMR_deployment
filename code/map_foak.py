import pandas as pd
import plotly.graph_objects as go
from utils import palette, app_palette
import matplotlib.pyplot as plt
import ANR_application_comparison


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

h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='nocogen')
h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
									 'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/MWe/y)' ]]
h2_data['Emissions_mmtco2/y'] = h2_data['Ann. avoided CO2 emissions (MMT-CO2/year)']
h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='nocogen')
heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'ANR',
											 'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
											 'Annual Net Revenues (M$/MWe/y)', 'Application']]
heat_data['App'] = 'Process Heat'
heat_data.reset_index(inplace=True)

foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/MWe/y)'] >=0]


def plot_bars(foak_positive):
	df = foak_positive[['App', 'Emissions_mmtco2/y', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Emissions_mmtco2/y':'Emissions', 'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	grouped_df = df.groupby('App').sum()

	# We then create the subplots
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(4, 5))  # 1 row, 2 columns of subplots

	# Build a 'bottom' series to stack the Apps in each bar
	bottom_emissions = [0]   # Starting at 0 for the first App in Emissions
	bottom_capacity = [0]    # Starting at 0 for the first App in Capacity

	# Stacked bar for Emissions and Capacity
	for app, values in grouped_df.iterrows():
			ax1.bar('Emissions', values['Emissions'], bottom=bottom_emissions[-1], color=app_palette[app], label=app, width=0.3)
			ax2.bar('Capacity', values['Capacity'], bottom=bottom_capacity[-1], color=app_palette[app], width=0.3)
			
			# Update the 'bottom' for the next App
			bottom_emissions.append(bottom_emissions[-1] + values['Emissions'])
			bottom_capacity.append(bottom_capacity[-1] + values['Capacity'])

	ax1.set_ylabel('Avoided emissions (MMtCO2/y)')
	ax1.set_xlabel('FOAK')

	ax2.set_ylabel('Deployed ANR capacity (GWe)')
	ax2.set_xlabel('FOAK')
	# Remove top and right spines for both subplots
	for ax in (ax1, ax2):
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

	# Place the legend outside of the subplots
	handles, labels = ax1.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
	plt.subplots_adjust(bottom=0.05, top=0.85, wspace=0.4)
	fig.savefig('./results/foak_positive_emissions_capacity.png', bbox_inches='tight')

plot_bars(foak_positive)

scaler = 40

		
# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'square', 'Industrial Hydrogen':'circle'}
marker_symbols = foak_positive['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in foak_positive['ANR']]

fig.add_trace(go.Scattergeo(
		lon=foak_positive['longitude'],
		lat=foak_positive['latitude'],
		text="Capacity: " + foak_positive['Depl. ANR Cap. (MWe)'].astype(str) + " MWe",
		mode='markers',
		marker=dict(
				size=foak_positive['Annual Net Revenues (M$/MWe/y)']*scaler+5,
				color=foak_positive['Annual Net Revenues (M$/MWe/y)'],
				colorscale='Greys',
				colorbar = dict(
						title='Annual Net Revenues (M$/MWe/y)',
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.1,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.8,  # Length of the colorbar (80% of figure width)
						tickvals = [0.1,0.25,0.5,0.75,1,1.25],
						ticktext = [0.1,0.25,0.5,0.75,1,1.25],
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
custom_legend = {'iMSR - Process Heat':[palette['iMSR'], 'square'],
								 'HTGR - Process Heat':[palette['HTGR'], 'square'],
								 'iPWR - Process Heat':[palette['iPWR'], 'square'],
								 'PBR-HTGR - Process Heat':[palette['PBR-HTGR'], 'square'],
								 'Micro - Process Heat':[palette['Micro'], 'square'],
								 'iMSR - Industrial H2':[palette['iMSR'], 'circle'],
								 'HTGR - Industrial H2':[palette['HTGR'], 'circle'],
								 'iPWR - Industrial H2':[palette['iPWR'], 'circle'],
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
fig.write_image('./results/map_FOAK.png')


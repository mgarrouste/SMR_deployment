import pandas as pd
import plotly.graph_objects as go
from utils import palette, app_palette
import ANR_application_comparison
import matplotlib.pyplot as plt

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



# NOAK data
h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
									 'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/MWe/y)', 'Annual Net Revenues (M$/y)' ]]
h2_data['Emissions_mmtco2/y'] = h2_data['Ann. avoided CO2 emissions (MMT-CO2/year)']
h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen')
heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'ANR',
											 'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
											 'Annual Net Revenues (M$/MWe/y)', 'Application', 'Annual Net Revenues (M$/y)']]
heat_data.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)',
												'Emissions_mmtco2/y':'Ann. avoided CO2 emissions (MMT-CO2/year)'}, inplace=True)
heat_data.reset_index(inplace=True, names=['id'])
heat_data['App'] = 'Process Heat'

noak_positive = pd.concat([heat_data, h2_data], ignore_index=True)
noak_positive = noak_positive[noak_positive['Annual Net Revenues (M$/y)'] >=0]

print(noak_positive['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))

# Size based on capacity deployed
percentiles =  noak_positive['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]).to_frame()
print(percentiles)

scaler = 30

# Set marker symbol based on the application's type
markers_applications = {'Industrial Hydrogen':'circle','Process Heat':'cross' }
marker_symbols = noak_positive['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in noak_positive['ANR']]

# size based on deployed capacity
def set_size(cap):
	if cap <= 200:
		size = 5
	elif cap <= 500:
		size = 15
	else:
		size = 40
	return size

noak_positive['size'] = noak_positive['Depl. ANR Cap. (MWe)'].apply(set_size)

max_rev = 52
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
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.1,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.7,  # Length of the colorbar (80% of figure width)
						tickvals = [0.1,10,25,50,100,500],
						ticktext = [0.1,10,25,50,100,500],
						tickmode='array'
						#dtick=0.1,
						#tick0=0.1
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
								 'PBR-HTGR - Process Heat':[palette['PBR-HTGR'], 'cross'],
								 'Micro - Process Heat':[palette['Micro'], 'cross'],
								 'iMSR - Industrial H2':[palette['iMSR'], 'circle'],
								 #'HTGR - Industrial H2':[palette['HTGR'], 'circle'],
								 #'iPWR - Industrial H2':[palette['iPWR'], 'circle'],
								 'PBR-HTGR - Industrial H2':[palette['PBR-HTGR'], 'circle'],
								 'Micro - Industrial H2':[palette['Micro'], 'circle']}

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
				title="<b>Application & ANR</b>",
				x=1,
				y=1,
				traceorder="normal",
				bgcolor="rgba(255, 255, 255, 0.5)"  # semi-transparent background
		),
)

# Save
fig.write_image('./results/map_NOAK_cogen.png')
# Show figure


### Plot waterfalls

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


def plot_bars(foak_positive, noak_positive):
	df = foak_positive[['App', 'Emissions_mmtco2/y', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Emissions_mmtco2/y':'Emissions', 'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df1 = df.reset_index()
	df1['measure'] = 'relative'
	total_df1 = df1.sum()
	total_df1['App'] = 'FOAK-Total'
	total_df1['measure'] = 'total'

	df = noak_positive[['App', 'Emissions_mmtco2/y', 'Depl. ANR Cap. (MWe)']]
	df = df.rename(columns={'Emissions_mmtco2/y':'Emissions', 'Depl. ANR Cap. (MWe)':'Capacity'})
	df['Capacity'] = df['Capacity']/1e3
	df = df.groupby('App').sum()
	df2 = df.reset_index()
	df2['measure'] = 'relative'
	total_df2 = df2.sum()
	total_df2['App'] = 'NOAK-Total'
	total_df2['measure'] = 'total'
	df2_adjusted = pd.concat([pd.DataFrame([total_df1]), df2, pd.DataFrame([total_df2])], ignore_index=True)
	
	# Now create a combined DataFrame from df1 and the adjusted df2
	combined_df = pd.concat([df1, df2_adjusted], ignore_index=True)
	print(combined_df)



	
plot_bars(foak_positive, noak_positive)
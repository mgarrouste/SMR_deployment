import pandas as pd
import plotly.graph_objects as go
from utils import palette
import ANR_application_comparison
from utils import compute_average_electricity_prices


# Create figure
fig = go.Figure()


"""
# Electricity: average state prices and breakeven prices
elec_path = './results/average_electricity_prices_MidCase_2024.xlsx'
if os.path.isfile(elec_path):
	elec_df = pd.read_excel(elec_path)
else:
	compute_average_electricity_prices(cambium_scenario='MidCase', year=2024)
	elec_df = pd.read_excel(elec_path)

# Define tick values and corresponding custom tick texts
colorbar_ticks = [20, 30, 40, 46.1, 52.2, 56.9, 77.8, 116.9]
colorbar_texts = ['20', '30', '40', 
									'BE iMSR: 46', 'BE PBR-HTGR: 52', 'BE iPWR: 57', 'BE HTGR: 78', 'BE Micro: 117']

max_actual_value = max(elec_df['average price ($/MWhe)'])
print('Maximum state-level average electricity price: ', max_actual_value)
max_tick_value = max(colorbar_ticks)
# List of colors for the colorscale (light to dark blue)
color_list = ["#ebf3fb", "#d2e3f3", "#b6d2ee", "#85bcdb", "#57a0ce", "#3082be", "#1361a9", "#0a4a90", "#08306b"]

# Compute the proportion of the actual max value against the maximum tick value
actual_data_proportion = max_actual_value / max_tick_value

# Build a normalized colorscale
colorscale = []
for i, color in enumerate(color_list):
		# Normalize the color positions based on the actual data proportion and evenly distribute them
		colorscale.append((i * actual_data_proportion / (len(color_list) - 1), color))
# Ensure the last color anchors at 1.0
colorscale.append((1, color_list[-1]))

fig.add_trace(
		go.Choropleth(
				locationmode='USA-states',  # Set the location mode to 'USA-states'
				locations=elec_df['state'],  # Use the 'state' column from the dataframe for locations
				z=elec_df['average price ($/MWhe)'],  # Use the 'average_price' column from the dataframe for coloring
				marker_line_color='white',  # Set the state boundary color
				marker_line_width=0.5,  # Set the state boundary width
				zmin=min(colorbar_ticks),  # Optional: Adjust the zmin if desired
				zmax=max_tick_value,       # Extending zmax to cover custom tick values
				autocolorscale=False,
				colorscale=colorscale,     # Custom colorscale defined above
				colorbar = dict(
						title='Electricity price ($/MWhe)',
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.3,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.7,  # Length of the colorbar (80% of figure width)
						tickvals=colorbar_ticks,  # Custom tick values
						ticktext=colorbar_texts,
						tickfont=dict(size=14)
				),
		)
)
"""


# FOAK data with no PTC
h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
h2_data = h2_data[['state','latitude', 'longitude', 'State price ($/MMBtu)','Depl. ANR Cap. (MWe)', 'BE wo PTC ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
									 'Industry', 'Application', 'ANR' ]]
h2_data['application'] = h2_data.apply(lambda x:'H2-'+x['Industry'].capitalize(), axis=1)
h2_data.rename(columns={'ANR':'SMR'}, inplace=True)
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
heat_data = heat_data[['STATE','latitude', 'longitude', 'NG price ($/MMBtu)', 'Emissions_mmtco2/y', 'SMR',
											 'Depl. ANR Cap. (MWe)', 'Industry','BE wo PTC ($/MMBtu)', 'Application']]
heat_data.rename(columns={'Emissions_mmtco2/y':'Ann. avoided CO2 emissions (MMT-CO2/year)',
													'NG price ($/MMBtu)':'State price ($/MMBtu)', 'STATE':'state'}, inplace=True)
heat_data['application'] = 'Process Heat'
heat_data.reset_index(inplace=True, names=['id'])

noptc_be = pd.concat([heat_data,h2_data], ignore_index=True)


def plot_be_vs_state_price(data):
	import seaborn as sns
	g = sns.FacetGrid(data, col='application', hue='SMR', col_wrap=2, palette=palette)
	g.map(sns.scatterplot, 'State price ($/MMBtu)', 'Breakeven price ($/MMBtu)')
	g.set(xlim=(3.5, 12), ylim=(0, 31), xticks=[4.1,7.6,11.2], yticks=[0,5,10,20,30])
	g.set_titles(col_template="{col_name}".capitalize())
	g.set_axis_labels('State price ($/MMBtu)', 'Breakeven price ($/MMBtu)')
	g.add_legend()
	g.savefig('./results/foak_noPTC_be_vs_state.png')


tosave_noptc = noptc_be[['id','state', 'State price ($/MMBtu)', 'BE wo PTC ($/MMBtu)', 'application', 'SMR']]
print('Minimum state price : {}'.format(min(tosave_noptc['State price ($/MMBtu)'])))
tosave_noptc = tosave_noptc.rename(columns={'BE wo PTC ($/MMBtu)':'Breakeven price ($/MMBtu)'})
plot_be_vs_state_price(tosave_noptc)
tosave_noptc.set_index('id', inplace=True)
tosave_noptc.to_latex('./results/foak_noPTC.tex',float_format="{:0.1f}".format, longtable=True, escape=True,\
												label='tab:foak_noPTC_detailed_results',\
						caption='Detailed results for FOAK without the H2 PTC deployment stage')


print(noptc_be['BE wo PTC ($/MMBtu)'].describe(percentiles = [.01,.02,.03,.05,.07,.08,.1,.17,.2,.25,.5,.75,.9]))

max_be = 17.4 # show only up to median BE

profitable = noptc_be[noptc_be['BE wo PTC ($/MMBtu)']<noptc_be['State price ($/MMBtu)']]
print('Number of facilities profitable with the hydrogen PTC : ',len(profitable))
noptc_be = noptc_be[noptc_be['BE wo PTC ($/MMBtu)']<=max_be]
noptc_be = noptc_be[noptc_be['BE wo PTC ($/MMBtu)']>noptc_be['State price ($/MMBtu)']]


# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle'}
marker_symbols = noptc_be['Application'].map(markers_applications).to_list()

colorbar_ticks = [6.21, 7.56,11.18,  17.3]
colorbar_texts = ['1th: 6.2','Maximum state level: 7.6', '10 year peak (2008): 11.2', 'Median: 17.3']

fig.add_trace(go.Scattergeo(
		lon=noptc_be['longitude'],
		lat=noptc_be['latitude'],
		text="Breakeven price: " + noptc_be['BE wo PTC ($/MMBtu)'].astype(str) + " $/MMBtu",
		mode='markers',
		marker=dict(
				size=12,
				color=noptc_be['BE wo PTC ($/MMBtu)'],
				colorscale='Reds',
				colorbar = dict(
						title='Breakeven NG price ($/MMBtu)',
						orientation='v',  
						x=0.9, 
						y=0.45,  
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.8,  # Length of the colorbar (80% of figure width)
						tickvals=colorbar_ticks,  # Custom tick values
						ticktext=colorbar_texts,
						tickfont=dict(size=14)
				),
				symbol=marker_symbols,
				line_color='black',
				line_width=1,
		),
		showlegend=False
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
				l=0,  # left margin
				r=20,  # right margin
				b=20,  # bottom margin
				t=10  # top margin
		),
		legend=dict(
				title="<b>Industrial Application</b>",
				x=0.9,
				y=1,
				traceorder="normal",
				font = dict(size = 16, color = "black"),
				bgcolor="rgba(255, 255, 255, 0.5)"  # semi-transparent background
		),
)


# Save
fig.write_image('./results/map_noPTC.png', scale=4)
fig.show()

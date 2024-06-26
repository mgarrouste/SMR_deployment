import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import palette
import ANR_application_comparison
import os
import numpy as np
from utils import compute_average_electricity_prices

with_elec = True
two_graphs = True
# Create figure
fig = go.Figure()



def add_elec_layer(fig, col, row):
	# Electricity: average state prices and breakeven prices
	elec_path = './results/average_electricity_prices_MidCase_2024.xlsx'
	if os.path.isfile(elec_path):
		elec_df = pd.read_excel(elec_path)
	else:
		compute_average_electricity_prices(cambium_scenario='MidCase', year=2024)
		elec_df = pd.read_excel(elec_path)

	# Define tick values and corresponding custom tick texts
	colorbar_ticks = [20, 34, 46.1, 52.2, 56.9, 77.8, 116.9]
	colorbar_texts = ['20', '34',
										'iMSR: 46', 'PBR-HTGR: 52', 'iPWR: 57', 'HTGR: 78', 'Micro: 117']

	max_actual_value = max(elec_df['average price ($/MWhe)'])
	print(elec_df['average price ($/MWhe)'].describe())
	max_tick_value = max(colorbar_ticks)
	# List of colors for the colorscale (light to dark blue)
	color_list = ["#ebf3fb", "#d2e3f3", "#b6d2ee", "#85bcdb", '#ffccff', '#ffb3ff', '#ff00ff','#b300b3']#"#57a0ce", "#3082be", "#1361a9", "#0a4a90", "#08306b"]

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
							title='Electricity<br>price ($/MWhe)',
							orientation='v',  # Set the orientation to 'h' for horizontal
							x=1.11,  # Center the colorbar horizontally
							y=0.0,  # Position the colorbar below the x-axis
							xanchor='center',
							yanchor='bottom',
							lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
							len=0.4,  # Length of the colorbar (80% of figure width)
							tickvals=colorbar_ticks,  # Custom tick values
							ticktext=colorbar_texts,
							tickfont=dict(size=16)
					),
			), row=row, col=col)



# FOAK data with no PTC
h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
h2_data = h2_data[['state','latitude', 'longitude', 'State price ($/MMBtu)','Depl. ANR Cap. (MWe)', 'BE wo PTC ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
									 'Industry', 'Application', 'ANR', 'IRR wo PTC' ]]
h2_data['application'] = h2_data.apply(lambda x:'H2-'+x['Industry'].capitalize(), axis=1)
h2_data.rename(columns={'ANR':'SMR'}, inplace=True)
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='cogen')
heat_data = heat_data[['STATE','latitude', 'longitude', 'NG price ($/MMBtu)', 'Emissions_mmtco2/y', 'SMR',
											 'Depl. ANR Cap. (MWe)', 'Industry','BE wo PTC ($/MMBtu)', 'Application', 'IRR wo PTC']]
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


tosave_noptc = noptc_be[['id','state', 'application', 'SMR','State price ($/MMBtu)', 'BE wo PTC ($/MMBtu)', 'IRR wo PTC']]
tosave_noptc = tosave_noptc.rename(columns={'BE wo PTC ($/MMBtu)':'Breakeven price ($/MMBtu)', 'IRR wo PTC':'IRR'})
tosave_noptc['IRR'] *=100
plot_be_vs_state_price(tosave_noptc)
tosave_noptc.set_index('id', inplace=True)
tosave_noptc.to_latex('./results/foak_noPTC.tex',float_format="{:0.1f}".format, longtable=True, escape=True,\
												label='tab:foak_noPTC_detailed_results',\
						caption='Detailed results for FOAK without the H2 PTC deployment stage')


print(noptc_be['BE wo PTC ($/MMBtu)'].describe(percentiles = [.01,.02,.03,.05,.07,.08,.1,.17,.2,.25,.5,.75,.9]))

def histogram_and_kde(series):
		import scipy.stats
		counts, bins = np.histogram(series, bins=30)
		bins = 0.5 * (bins[:-1] + bins[1:]) # bin centers
		# Calculate KDE
		kde = scipy.stats.gaussian_kde(series)
		kde_x = np.linspace(series.min(), series.max(), 100)
		kde_y = kde(kde_x) * np.diff(bins)[0] * len(series) # Scale the KDE by number of observations and bin width
		return bins, counts, kde_x, kde_y

heat_datakde = noptc_be[noptc_be.Application=='Process Heat']
heat_datakde = heat_datakde[heat_datakde['BE wo PTC ($/MMBtu)'] <=110]
heat_bins, heat_counts, heat_kde_x, heat_kde_y = histogram_and_kde(heat_datakde['BE wo PTC ($/MMBtu)'])
h2_bins, h2_counts, h2_kde_x, h2_kde_y = histogram_and_kde(noptc_be[noptc_be.Application=='Industrial Hydrogen']['BE wo PTC ($/MMBtu)'])
max_be = 17.4 # show only up to median BE

profitable = noptc_be[noptc_be['BE wo PTC ($/MMBtu)']<noptc_be['State price ($/MMBtu)']]
print('Number of facilities profitable without the hydrogen PTC : ',len(profitable))
noptc_be = noptc_be[noptc_be['BE wo PTC ($/MMBtu)']<=max_be]
noptc_be = noptc_be[noptc_be['BE wo PTC ($/MMBtu)']>noptc_be['State price ($/MMBtu)']]


# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle'}
marker_symbols = noptc_be['Application'].map(markers_applications).to_list()

colorbar_ticks = [6.21, 7.56,11.18,  17.3]
colorbar_texts = ['1th: 6.2','Maximum state<br>level: 7.6', '10 year peak<br>(2008): 11.2', 'Median: 17.3']

if two_graphs:
	fig = make_subplots(rows=3, cols=1, row_heights=[0.4,0.2,0.4],
										specs=[[{"type": "scattergeo"}], 
								 					 [{'type':'xy'}],
													 [{"type": "scattergeo"}]])
	add_elec_layer(fig=fig, row=1, col=1)
	add_elec_layer(fig=fig, row=3, col=1)

	# Process heat on the left
	process_heat = noptc_be[noptc_be['Application'] == 'Process Heat']
	process_heat_markers = process_heat['Application'].map(markers_applications).to_list()
	fig.add_trace(go.Scattergeo(
			lon=process_heat['longitude'],
			lat=process_heat['latitude'],
			mode='markers',
			marker=dict(
					size=12,
					color=process_heat['BE wo PTC ($/MMBtu)'],
					colorscale='Reds',
					colorbar = dict(
							title='Breakeven NG<br>price ($/MMBtu)',
							orientation='v',  
							x=1., 
							y=.7,  
							lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
							len=.4,  # Length of the colorbar (80% of figure width)
							tickvals=colorbar_ticks,  # Custom tick values
							ticktext=colorbar_texts,
							tickfont=dict(size=16)
					),
					symbol=process_heat_markers,
					line_color='black',
					line_width=1,
			),
			showlegend=False
	), row=1, col=1)


	# Histogram and kde for 'BE wo PTC' column
	fig.add_trace(
			go.Bar(x=heat_bins, y=heat_counts, marker_color='red', name='Process Heat', textfont=dict(size=14), showlegend=False),
			row=2, col=1
	)
	fig.add_trace(
    go.Scatter(x=heat_kde_x, y=heat_kde_y,line=dict(color='red'), showlegend=False),
    row=2, col=1
	)
	fig.add_trace(
			go.Bar(x=h2_bins, y=h2_counts, marker_color='blue',name='Process Hydrogen', textfont=dict(size=14), showlegend=False),
			row=2, col=1
	)
	fig.add_trace(
    go.Scatter(x=h2_kde_x, y=h2_kde_y,line=dict(color='blue'), showlegend=False),
    row=2, col=1
	)
	fig.update_layout(barmode='overlay')

	# H2 on the right
	h2 = noptc_be[noptc_be['Application'] == 'Industrial Hydrogen']
	h2_markers = h2['Application'].map(markers_applications).to_list()
	fig.add_trace(go.Scattergeo(
			lon=h2['longitude'],
			lat=h2['latitude'],
			mode='markers',
			marker=dict(
					size=12,
					color=h2['BE wo PTC ($/MMBtu)'],
					colorscale='Reds',
					symbol=h2_markers,
					line_color='black',
					line_width=1,
			),
			showlegend=False, 
	),row=3, col=1)

	
	# Update layout to add a two-line x-axis label and font sizes to the distribution plots
	fig.update_xaxes(
		title_text='Breakeven NG price ($/MMBtu)',
		title_font=dict(size=14),
		tickfont=dict(size=14),
		row=2,
		col=1
	)

	fig.update_geos(
		scope="usa",  # Limits the map scope to North America
		showlakes=True,
		lakecolor='rgb(255, 255, 255)',
	)
	# Create symbol and color legend traces
	color_map_app = {'Process Heat':'red', 'Industrial Hydrogen':'blue'}
	for app, marker in markers_applications.items():
			fig.add_trace(go.Scattergeo(
					lon=[None],
					lat=[None],
					marker=dict(
							size=15,
							color=color_map_app[app],
							symbol=marker,
							line_color='black',
							line_width=2,
					),
					name=app
			))

	# Update layout
	fig.update_layout(
		height=1200,  # Set the height of the figure
		width=1000,  # Increase the width
		margin=dict(l=0, r=5, t=0, b=0, pad=.03),
		legend=dict(
				title="Industrial Application",
				x=1.01,
				y=0.98,
				traceorder="normal",
				font = dict(size = 16, color = "black"),
				bgcolor="rgba(255, 255, 255, 0.5)"  # semi-transparent background
		),
	)

else:

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


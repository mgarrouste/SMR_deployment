import pandas as pd
from ANR_application_comparison import load_heat_results, load_h2_results
import plotly.graph_objects as go
import os 
from utils import compute_average_electricity_prices, palette

def load_data():
	heat = load_heat_results(anr_tag='FOAK', cogen_tag='cogen', with_PTC=False)
	heat= heat[['STATE','latitude', 'longitude', 'NG price ($/MMBtu)', 'Emissions_mmtco2/y', 'SMR',
											 'Depl. ANR Cap. (MWe)', 'Industry','Annual Net Revenues (M$/y)', 'Application', 'IRR wo PTC']]
	heat = heat[heat['Annual Net Revenues (M$/y)']>0]
	heat.rename(columns={'Emissions_mmtco2/y':'Emissions',
														'NG price ($/MMBtu)':'State price ($/MMBtu)', 'STATE':'state', 'IRR wo PTC':'IRR (%)'}, inplace=True)
	heat['application'] = 'Process Heat'
	heat['IRR (%)'] *=100
	heat.reset_index(inplace=True, names=['id'])
	print('# process heat facilities profitable wo PTc :{}'.format(len(heat[heat['Annual Net Revenues (M$/y)']>0])))
	print(heat['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	print(heat['IRR (%)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	print('iPWR')
	print(heat[heat['SMR']=='iPWR']['Depl. ANR Cap. (MWe)'].sum())
	print('iMSR')
	print(heat[heat['SMR']=='iMSR']['Depl. ANR Cap. (MWe)'].sum())
	print('PBRHTGR')
	print(heat[heat['SMR']=='PBR-HTGR']['Depl. ANR Cap. (MWe)'].sum())
	print('Total capacity')
	print(heat['Depl. ANR Cap. (MWe)'].describe(percentiles=[.1,.25,.5,.75,.9]))
	print(heat['SMR'].unique())


	h2 = load_h2_results(anr_tag='FOAK', cogen_tag='cogen')
	h2 = h2.loc[:,~h2.columns.duplicated()]
	h2 = h2.reset_index()
	h2['Annual Net Revenues wo PTC (M$/y)'] = h2['Electricity revenues ($/y)']+h2['Net Revenues ($/year)']
	print('# process hydrogen facilities profitable wo PTc :{}'.format(len(h2[h2['Annual Net Revenues wo PTC (M$/y)']>0])))

	return heat

def add_elec_layer(fig):
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
	color_list = ["#d5e1f0","#abc9ed",'#ffccff', '#ffb3ff', '#ff00ff','#b300b3']#"#57a0ce", "#3082be", "#1361a9", "#0a4a90", "#08306b"]

	# Compute the proportion of the actual max value against the maximum tick value
	actual_data_proportion = max_actual_value / max_tick_value

	# Build a normalized colorscale
	colorscale = []
	for i, color in enumerate(color_list):
			# Normalize the color positions based on the actual data proportion and evenly distribute them
			colorscale.append((i * actual_data_proportion / (len(color_list) - 1), color))
	# Ensure the last color anchors at 1.0
	colorscale.append((1, color_list[-1]))

	# Fancy colorscale with LCOEs
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
							title='$/MWhe',
							orientation='v',  # Set the orientation to 'h' for horizontal
							x=.99,  # Center the colorbar horizontally
							y=-.1,  # Position the colorbar below the x-axis
							xanchor='center',
							yanchor='bottom',
							lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
							len=0.67,  # Length of the colorbar (80% of figure width)
							tickvals=colorbar_ticks,  # Custom tick values
							ticktext=colorbar_texts,
							tickfont=dict(size=16)
					),
			))


def save_to_latex(df):
	tosave_noptc = df[['id','state', 'application', 'SMR','State price ($/MMBtu)', 'Annual Net Revenues (M$/y)', 'IRR (%)']]
	tosave_noptc.set_index('id', inplace=True)
	tosave_noptc.to_latex('./results/foak_noPTC.tex',float_format="{:0.1f}".format, longtable=True, escape=True,\
													label='tab:foak_noPTC_detailed_results',\
							caption='Detailed results for FOAK without PTC deployment stage')


def add_nuclear_bans(fig):
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



def add_smr_layer(fig,df):
	# Markers
	markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle'}
	marker_symbols = df['Application'].map(markers_applications).to_list()
	# SMR colors
	line_colors = [palette[anr] for anr in df['SMR']]
	# DEployed capacity as size
	def set_size(cap):
		if cap <= 100:
			size = 6
		elif cap <= 500:
			size = 12
		else:
			size = 25
		return size

	df['size'] = df['Depl. ANR Cap. (MWe)'].apply(set_size)

	fig.add_trace(go.Scattergeo(
		lon=df['longitude'],
		lat=df['latitude'],
		mode='markers',
		marker=dict(
				size=df['size'],
				color=df['IRR (%)'],
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
						len=0.6,  # Length of the colorbar (80% of figure width)
						tickvals = [5,8,11],
						ticktext = [5,8,11],
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

	# Custom legend
	custom_legend = {'iMSR - Process Heat':[palette['iMSR'], 'cross'],
									'iPWR - Process Heat':[palette['iPWR'], 'cross'],
									'PBR-HTGR - Process Heat':[palette['PBR-HTGR'], 'cross']}
	reactors_used = df['SMR'].unique()

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
	sizes = df['size'].unique()
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




def main():
	df = load_data()
	from map_foak import plot_irr 
	plot_irr(data=df, save_path='./results/IRR_foaknoptc.png')
	save_to_latex(df)
	fig = go.Figure()
	add_elec_layer(fig)

	add_smr_layer(fig,df)

	#add_nuclear_bans(fig)
	
	fig.write_image('./results/map_noPTC.png', scale=4)

if __name__ == '__main__':
	main()
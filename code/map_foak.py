import pandas as pd
import plotly.graph_objects as go
from utils import palette
import matplotlib.pyplot as plt
import SMR_application_comparison
from plotly.subplots import make_subplots
import waterfalls_cap_em 
import seaborn as sns


def load_data():
	foak_positive = waterfalls_cap_em.load_foak_positive()
	foak_positive['IRR (%)'] = foak_positive['IRR w PTC']
	return foak_positive


def add_state_layer(fig):
	all_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', \
							'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', \
								'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', \
									'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
	fig.add_trace(
		go.Choropleth(
			locationmode='USA-states',
			locations=all_states,  # List of state codes
			z=[1]*len(all_states),  # Dummy variable for coloring
			colorscale=['white', 'white'],  # Set the color scale to white
			showscale=False,  # Hide the color scale
			marker_line_color='grey',  # Set the border color to grey
			marker_line_width=0.7,  # Set the border width
	))


def plot_irr(data, save_path, app_col='application'):
	fig, ax = plt.subplots(figsize=(5,3))
	sns.kdeplot(ax=ax, data=data, x='IRR (%)', cumulative=True, hue=app_col, common_norm=False, common_grid=True)
	sns.despine()
	ax.get_legend().set_title('Application')
	ax.xaxis.grid(True)
	fig.tight_layout()
	fig.savefig(save_path, bbox_inches='tight')


def add_smr_layer(fig,df,application):
	# Markers
	markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle'}
	marker_symbols = df['Application'].map(markers_applications).to_list()
	# SMR colors
	line_colors = [palette[SMR] for SMR in df['SMR']]
	# DEployed capacity as size
	def set_size(cap):
		if cap <= 150: size = 10
		elif cap <= 500:size = 17
		elif cap<=750:size = 25
		else:size = 35
		return size
	def set_size_nb(nb):
		if nb <=1: size = 3
		elif nb <=5: size = 10
		else: size = 25
		return size
	#df['size'] = df['Depl. SMR Cap. (MWe)'].apply(set_size)
	
	df['size'] = df['# SMR modules'].apply(set_size_nb)

	df = df.sort_values(by=['Application'], ascending=True)
	sup = df[df['IRR (%)'] >=4.5]
	df = df[df['IRR (%)']<4.5]


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
						#titlefont = dict(size=16),
						orientation='h',  # Set the orientation to 'h' for horizontal
						x=0.5,  # Center the colorbar horizontally
						y=-0.15,  # Position the colorbar below the x-axis
						xanchor='center',
						yanchor='bottom',
						lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
						len=0.6,  # Length of the colorbar (80% of figure width)
						tickvals = [0.6,2.5,4],
						ticktext = [0.6,2.5,4],
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
	if application == 'Process Heat':
		custom_legend = {'MSR - Process Heat':[palette['MSR'], 'cross'],
						'PWR - Process Heat':[palette['PWR'], 'cross'],
						'SFR - Process Heat':[palette['SFR'], 'cross'],
						'HTR - Process Heat':[palette['HTR'], 'cross'],
						'MR - Process Heat':[palette['MR'], 'cross']
						}
	elif application == 'Industrial Hydrogen':
		custom_legend = {
					'MSR - Industrial H2':[palette['MSR'], 'circle'],
					'PWR - Industrial H2':[palette['PWR'], 'circle'],
					'SFR - Industrial H2':[palette['SFR'], 'circle'],
					'HTR - Industrial H2':[palette['HTR'], 'circle'],
					'MR - Industrial H2':[palette['MR'], 'circle'],
					}
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


def load_foak_positive_2():
	h2_data = SMR_application_comparison.load_h2_results(OAK='FOAK', cogen_tag='cogen')
	h2_data = h2_data[['latitude', 'longitude', 'state', 'Depl. SMR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'SMR', 'Annual Net Revenues (M$/y)', 'IRR w PTC', 'IRR wo PTC' ]]
	h2_data['Emissions_mmtco2/y'] = h2_data['Ann. avoided CO2 emissions (MMT-CO2/year)']
	h2_data.rename(columns={'SMR':'SMR'}, inplace=True)
	h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
	h2_data.reset_index(inplace=True)

	heat_data = SMR_application_comparison.load_heat_results(OAK='FOAK', cogen_tag='cogen')
	heat_data = heat_data[['latitude', 'longitude', 'STATE', 'Emissions_mmtco2/y', 'SMR',
												'Depl. SMR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
												'Annual Net Revenues (M$/y)', 'Application', 'IRR w PTC', 'IRR wo PTC']]
	heat_data['App'] = 'Process Heat'
	heat_data.rename(columns={'STATE':'state'}, inplace=True)
	heat_data.reset_index(inplace=True)

	foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
	foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]
	return foak_positive


def save_foak_positive():
	h2_data = SMR_application_comparison.load_h2_results(OAK='FOAK', cogen_tag='cogen')
	h2_data = h2_data[['state', 'Depl. SMR Cap. (MWe)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
										'Industry', 'Application', 'SMR', 'Annual Net Revenues (M$/y)', 'IRR w PTC']]
	h2_data.rename(columns={'Ann. avoided CO2 emissions (MMT-CO2/year)':'Emissions (MMtCO2/y)', 'state':'State', 'SMR':'SMR'}, inplace=True)
	h2_data['application'] = h2_data.apply(lambda x:'H2-'+x['Industry'].capitalize(), axis=1)
	h2_data = h2_data.reset_index(names=['id'])

	heat_data = SMR_application_comparison.load_heat_results(OAK='FOAK', cogen=True)
	heat_data = heat_data[['STATE', 'Emissions_mmtco2/y', 'SMR','Depl. SMR Cap. (MWe)', 'Annual Net Revenues (M$/y)', 'Application', 'IRR w PTC']]
	heat_data.rename(columns={'Emissions_mmtco2/y':'Emissions (MMtCO2/y)', 'STATE':'State'}, inplace=True)
	heat_data['application'] = 'Process Heat'
	heat_data = heat_data.reset_index(names=['id'])

	
	foak_positive = pd.concat([h2_data, heat_data], ignore_index=True)
	foak_positive = foak_positive[foak_positive['Annual Net Revenues (M$/y)'] >=0]
	foak_positive.set_index('id', inplace=True)
	foak_positive['Depl. SMR Cap. (MWe)'] = foak_positive['Depl. SMR Cap. (MWe)'].astype(int)
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


def plot_waterfall(foak_positive):
	df = foak_positive[['App', 'Emissions_mmtco2/y', 'Depl. SMR Cap. (MWe)']]
	df = df.rename(columns={'Emissions_mmtco2/y':'Emissions', 'Depl. SMR Cap. (MWe)':'Capacity'})
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
	fig.update_yaxes(title_text='SMR Capacity (GWe)', row=1, col=2)
	fig.update_xaxes(tickangle=270)
	# Set chart layout
	fig.update_layout(
		margin=dict(l=20, r=20, t=20, b=20),
		showlegend = False,
		width=400,  # Set the width of the figure
		height=550,  # Set the height of the figure
	)

	fig.write_image('./results/foak_cogen_positive_emissions_capacity.png')



def main(subplots=True):
	df = load_data()
	plot_irr(data=df, app_col='Application', save_path='./results/IRR_foak.png')

	if subplots:
		heat = df[df['Application'] == 'Process Heat']
		indh2 = df[df['Application'] == 'Industrial Hydrogen']
		fig_heat = go.Figure()
		fig_indh2 = go.Figure()
		add_state_layer(fig_heat)
		add_state_layer(fig_indh2)
		add_smr_layer(fig_heat,heat,application='Process Heat')
		add_smr_layer(fig_indh2,indh2,application='Industrial Hydrogen')
		ITC = 0.3
		fig_heat.write_image(f'./results/map_heat_FOAK_PTC_ITC_{ITC}.png', scale=4)
		fig_indh2.write_image(f'./results/map_indh2_FOAK_PTC_ITC_{ITC}.png', scale=4)
	else:
		fig = go.Figure()
		add_state_layer(fig)
		add_smr_layer(fig,df)
		ITC = 0.3
		fig.write_image(f'./results/map_FOAK_PTC_ITC_{ITC}.png', scale=4)

if __name__ == '__main__':
	main()
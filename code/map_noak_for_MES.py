import pandas as pd
import plotly.graph_objects as go
from utils import palette, app_palette
from plotly.subplots import make_subplots
import ANR_application_comparison, map_foak

# Create figure
fig = go.Figure()


import waterfalls_cap_em

noak_positive = waterfalls_cap_em.load_noak_positive()

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


# size based on deployed capacity
def set_size(cap):
	if cap <= 200:
		size = 7
	elif cap <= 500:
		size = 15
	else:
		size = 40
	return size

noak_positive['size'] = noak_positive['Depl. ANR Cap. (MWe)'].apply(set_size)

print(noak_positive['Annual Net Revenues (M$/y)'].describe(percentiles=[.1,.25,.5,.75,.9]))
max_rev = 12
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
						tickvals = [2.6,5.1,11.3,25,50,100,500],
						ticktext = [2.6,5.1,11.3,25,50,100,500],
						tickmode='array',
						tickfont=dict(size=16)
				),
				symbol=marker_symbols,
				line_color='black',
				line_width=2,
				sizemode='diameter'
		),
		showlegend=False
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
	
fig.add_trace(go.Scattergeo(
					lon=[None],
					lat=[None],
					marker=dict(
							size=15,
							color='white',
							line_color='black',
							line_width=4,
							symbol='cross'
					),
					name='Process Heat'
			))
fig.add_trace(go.Scattergeo(
					lon=[None],
					lat=[None],
					marker=dict(
							size=15,
							color='white',
							line_color='black',
							line_width=4,
							symbol='circle'
					),
					name='Industrial Hydrogen'
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
fig.write_image('./results/map_NOAK_cogen_MES.png', scale=4)
# Show figure



import pandas as pd
import plotly.graph_objects as go
from utils import  app_palette
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
h2_data = h2_data[['latitude', 'longitude', 'Industry', 'Application']]
h2_data['App'] = h2_data.apply(lambda x: x['Application']+'-'+x['Industry'].capitalize(), axis=1)
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='nocogen')
heat_data = heat_data[['latitude', 'longitude', 'Batch_Temp_degC','Application']]
heat_data['App'] = 'Process Heat'
heat_data.reset_index(inplace=True)


facilities = pd.concat([heat_data, h2_data], ignore_index=True)

scaler = 0.02

		
# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'cross-open-dot', 'Industrial Hydrogen':'circle-open-dot'}
marker_symbols = facilities['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [app_palette[app] for app in facilities['App']]


fig.add_trace(go.Scattergeo(
		lon=facilities['longitude'],
		lat=facilities['latitude'],
		mode='markers',
		marker=dict(
				size=10,
				symbol=marker_symbols,
        color = line_colors,
				line_color=line_colors,
				line_width=2,
		),
		showlegend=False
))

# Create symbol and color legend traces
for app, color in app_palette.items():
  if 'Hydrogen' in app: symbol = 'circle-open-dot'
  else: symbol = 'cross-open-dot'
  fig.add_trace(go.Scattergeo(
      lon=[None],
      lat=[None],
      marker=dict(
          size=15,
          color=color,
          symbol=symbol,
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



# Update layout
fig.update_layout(
		geo=dict(
				scope='usa',
				projection_type='albers usa',
				showlakes=True,
				lakecolor='rgb(255, 255, 255)',
		),
		width=900,  # Set the width of the figure
		height=500,  # Set the height of the figure
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
fig.write_image('./results/map_context.png', scale=4)


import plotly.graph_objects as go
import os 
import pandas as pd
from utils import compute_average_electricity_prices

fig = go.Figure()


# Electricity: average state prices and breakeven prices
elec_path = './results/average_electricity_prices_MidCase_2024.xlsx'
if os.path.isfile(elec_path):
  elec_df = pd.read_excel(elec_path)
else:
  compute_average_electricity_prices(cambium_scenario='MidCase', year=2024)
  elec_df = pd.read_excel(elec_path)

print(elec_df['average price ($/MWhe)'].describe(percentiles=[.1,.25,.5,.75,.9]))

fig.add_trace(
  go.Choropleth(
    locationmode='USA-states',
    locations=elec_df['state'],
    z=elec_df['average price ($/MWhe)'],
    marker_line_color='white',  # Set the state boundary color
    marker_line_width=0.5,  # Set the state boundary width
    colorscale='Blues',
    colorbar = dict(
							title='Average price ($/MWhe)',
							orientation='h',  # Set the orientation to 'h' for horizontal
							x=0.5,  # Center the colorbar horizontally
							y=.1,  # Position the colorbar below the x-axis
							xanchor='center',
							yanchor='bottom',
							lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
							len=0.7,  # Length of the colorbar (80% of figure width)
              tickvals=[21.5, 26, 31.3, 34],  # Custom tick values
							ticktext=['Min: 21','10th: 26','Median: 31','Max: 34'],
							tickfont=dict(size=18)
    ),
  )
)

fig.update_layout(
  geo=dict(
    scope='usa',
    projection_type='albers usa',
    showlakes=True,
    lakecolor='rgb(255, 255, 255)',
  ),
  height=800,  # Set the height of the figure
  width=800,  # Increase the width
  margin=dict(l=0, r=0, t=0, b=0, pad=.5),
)

# Save
fig.write_image('./results/map_electricity.png', scale=4)
fig.show()
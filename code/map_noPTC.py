import pandas as pd
import plotly.graph_objects as go
from utils import palette
import ANR_application_comparison
import os
from utils import compute_average_electricity_prices


# Create figure
fig = go.Figure()



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
                  'iMSR: $46/MWhe', 'PBR-HTGR: $52/MWhe', 'iPWR: $57/MWhe', 'HTGR: $78/MWhe', 'Micro: $117/MWhe']

max_actual_value = max(elec_df['average price ($/MWhe)'])
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
        ),
    )
)



# FOAK data with no PTC
h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='nocogen')
h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'BE wo PTC ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
                   'Industry', 'Application', 'ANR' ]]
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='nocogen')
heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'ANR',
                       'Depl. ANR Cap. (MWe)', 'Industry','BE wo PTC ($/MMBtu)', 'Application']]
heat_data.rename(columns={'Emissions_mmtco2/y':'Ann. avoided CO2 emissions (MMT-CO2/year)'}, inplace=True)
heat_data.reset_index(inplace=True, names=['id'])

noptc_be = pd.concat([h2_data, heat_data], ignore_index=True)
print(noptc_be)
print(noptc_be['BE wo PTC ($/MMBtu)'].describe(percentiles = [.1,.25,.5,.75,.9]))


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

fig.show()
exit()




scaler = 40

    
# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'square', 'Industrial Hydrogen':'circle'}
marker_symbols = noak_positive['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in noak_positive['ANR']]

fig.add_trace(go.Scattergeo(
    lon=noak_positive['longitude'],
    lat=noak_positive['latitude'],
    text="Capacity: " + noak_positive['Depl. ANR Cap. (MWe)'].astype(str) + " MWe",
    mode='markers',
    marker=dict(
        size=noak_positive['Annual Net Revenues (M$/MWe/y)']*scaler,
        color=noak_positive['Annual Net Revenues (M$/MWe/y)'],
        colorscale='Greys',
        colorbar = dict(
            title='Annual Net Revenues (M$/MWe/y)',
            orientation='h',  # Set the orientation to 'h' for horizontal
            x=0.5,  # Center the colorbar horizontally
            y=-0.1,  # Position the colorbar below the x-axis
            xanchor='center',
            yanchor='bottom',
            lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
            len=0.8  # Length of the colorbar (80% of figure width)
        ),
        symbol=marker_symbols,
        line_color=line_colors,
        line_width=5,
        sizemode='diameter'
    ),
    showlegend=False
))

# Create symbol and color legend traces
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

# Show figure
fig.show()

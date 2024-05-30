import pandas as pd
import plotly.graph_objects as go
from utils import palette
import ANR_application_comparison


# Create figure
fig = go.Figure()


# Electricity: Breakeven CAPEX
# Select design with minimum cost reduction for each state
elec_data = pd.read_excel('./results/price_taker_FOAK_MidCase.xlsx', index_col=0)
elec_data.reset_index(inplace=True, drop=True)
elec_data = elec_data[['state', 'Reactor', 'Cost red CAPEX BE']]
elec_data.rename(columns={'Reactor':'ANR'}, inplace=True)
idx = elec_data.groupby(['state'])['Cost red CAPEX BE'].idxmin()
elec_data = elec_data.loc[idx]
# Add coordinates of states geographic centers
geo_centers = pd.read_excel('./input_data/us_states_centers.xlsx')
elec_data = elec_data.merge(geo_centers, on='state', how='left')
elec_data['Cost red CAPEX BE']  *= 100

print(elec_data['Cost red CAPEX BE'].describe(percentiles=[.1,.25,.5,.75,.9]))

# Add electricity data separately to the figure
fig.add_trace(go.Scattergeo(
    lon=elec_data['longitude'],
    lat=elec_data['latitude'],
    mode='markers',
    text="Cost reduction needed " + elec_data['Cost red CAPEX BE'].astype(str) + " % FOAK",
    marker=dict(
        size=elec_data['Cost red CAPEX BE']*0.2,
        color=elec_data['Cost red CAPEX BE'],
        colorscale='Reds',
        symbol='triangle-up',
        line_color='blue',
        line_width=3,
        sizemode='diameter'
    ),
    showlegend=False
))


# NOAK data
h2_data = ANR_application_comparison.load_h2_results(anr_tag='NOAK', cogen_tag='cogen')
h2_data = h2_data[['latitude', 'longitude', 'Depl. ANR Cap. (MWe)', 'Breakeven price ($/MMBtu)', 'Ann. avoided CO2 emissions (MMT-CO2/year)', 
                   'Industry', 'Application', 'ANR', 'Annual Net Revenues (M$/MWe/y)' ]]
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='NOAK', cogen_tag='cogen')
heat_data = heat_data[['latitude', 'longitude', 'Emissions_mmtco2/y', 'ANR',
                       'Depl. ANR Cap. (MWe)', 'Industry', 'Breakeven NG price ($/MMBtu)',
                       'Annual Net Revenues (M$/MWe/y)', 'Application']]
heat_data.rename(columns={'Breakeven NG price ($/MMBtu)':'Breakeven price ($/MMBtu)',
                        'Emissions_mmtco2/y':'Ann. avoided CO2 emissions (MMT-CO2/year)'}, inplace=True)
heat_data.reset_index(inplace=True, names=['id'])

# Negative NOAK sites
noak_neg = pd.concat([h2_data, heat_data], ignore_index=True)
noak_neg = noak_neg[noak_neg['Annual Net Revenues (M$/MWe/y)'] <0]


# Load FOAK capex breakeven data for those sites
h2_data = ANR_application_comparison.load_h2_results(anr_tag='FOAK', cogen_tag='nocogen')
h2_data = h2_data[['Breakeven CAPEX ($/MWe)', 'Cost red CAPEX BE']]
h2_data.reset_index(inplace=True)

heat_data = ANR_application_comparison.load_heat_results(anr_tag='FOAK', cogen_tag='nocogen')
heat_data = heat_data[['Breakeven CAPEX ($/MWe)', 'Cost red CAPEX BE']]
heat_data.reset_index(inplace=True, names=['id'])

be_foak = pd.concat([h2_data, heat_data], ignore_index=True)



# Merge data to get breakeven capex and cost reduction for breakeven for sites with negative revenues in NOAK cogen case
noak_neg = noak_neg.merge(right=be_foak, on=['id'],how='left')
noak_neg['Cost red CAPEX BE'] *=100
print(noak_neg['Cost red CAPEX BE'].describe(percentiles=[.1,.25,.5,.75,.9]))


#Scale up
scaler = 0.2


# First only show up to 100 % 
noak_neg100 = noak_neg[noak_neg['Cost red CAPEX BE']<=100]
noak_negsup100 = noak_neg[noak_neg['Cost red CAPEX BE']>100]

# Above 100%
# Set marker symbol based on the application's type
markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle', 'Electricity':'triangle-up'}
marker_symbols = noak_negsup100['Application'].map(markers_applications).to_list()
# Get colors for each marker
line_colors = [palette[anr] for anr in noak_negsup100['ANR']]

fig.add_trace(go.Scattergeo(
    lon=noak_negsup100['longitude'],
    lat=noak_negsup100['latitude'],
    text="Cost reduction needed " + noak_negsup100['Cost red CAPEX BE'].astype(str) + " % FOAK",
    mode='markers',
    marker=dict(
        size=30,
        color='#610000',
        symbol=marker_symbols,
        line_color=line_colors,
        line_width=3,
    ),
    showlegend=False
))



# Below 100%
# Set marker symbol based on the application's type
noak_neg100 = noak_neg100[['Application', 'latitude', 'longitude', 'Cost red CAPEX BE', 'ANR']]
markers_applications = {'Process Heat':'cross', 'Industrial Hydrogen':'circle', 'Electricity':'triangle-up'}
marker_symbols = noak_neg100['Application'].map(markers_applications).to_list()

# Get colors for each marker
line_colors = [palette[anr] for anr in noak_neg100['ANR']]


fig.add_trace(go.Scattergeo(
    lon=noak_neg100['longitude'],
    lat=noak_neg100['latitude'],
    text="Cost reduction needed: " + noak_neg100['Cost red CAPEX BE'].astype(str) + " % FOAK CAPEX",
    mode='markers',
    marker=dict(
        size=noak_neg100['Cost red CAPEX BE']*scaler,
        color=noak_neg100['Cost red CAPEX BE'],
        colorscale='Reds',
        colorbar = dict(
            title='CAPEX cost reduction for breakeven (% FOAK)',
            orientation='h',  # Set the orientation to 'h' for horizontal
            x=0.5,  # Center the colorbar horizontally
            y=-0.1,  # Position the colorbar below the x-axis
            xanchor='center',
            yanchor='bottom',
            lenmode='fraction',  # Use 'fraction' to specify length in terms of fraction of the plot area
            len=0.8,  # Length of the colorbar (80% of figure width)
            tickvals = [45,50,60,70,80,90],
            ticktext = [45,50,60,70,80,90],
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
custom_legend = {'iMSR - Process Heat':[palette['iMSR'], 'cross'],
                 #'HTGR - Process Heat':[palette['HTGR'], 'cross'],
                 #'iPWR - Process Heat':[palette['iPWR'], 'cross'],
                 'PBR-HTGR - Process Heat':[palette['PBR-HTGR'], 'cross'],
                 'Micro - Process Heat':[palette['Micro'], 'cross'],
                 'iMSR - Industrial H2':[palette['iMSR'], 'circle'],
                 #'HTGR - Industrial H2':[palette['HTGR'], 'circle'],
                 #'iPWR - Industrial H2':[palette['iPWR'], 'circle'],
                 'PBR-HTGR - Industrial H2':[palette['PBR-HTGR'], 'circle'],
                 'Micro - Industrial H2':[palette['Micro'], 'circle'],
                 'iMSR - Electricity':[palette['iMSR'], 'triangle-up']}

reactors_used = noak_neg['ANR'].unique()

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
fig.write_image('./results/map_above_NOAK.png')

# Show figure
fig.show()

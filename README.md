# Hydrogen from Advanced Nuclear Reactors
Optimal deployment of Advanced Nuclear Reactors coupled with electrolysis for the decarbonization of four major industrial 
sectors: ammonia, high-temperature process heat, refining and steel.

## Parameters 
- ANRs.xlsx FOAK sheet- Engineering and cost parameters for Advanced Nuclear Reactors First Of A Kind 

- h2_tech.xlsx Summary sheet- Engineering and cost parameters of coupled hydrogen production technologies with all designs of reactors in 'ANRs.xlsx'

## Data

- h2_demand_refineries.xlsx - hydrogen demand in kg/d from refineries calculated from hydrogen production on-site from Steam Methane Reforming, refineries without data for 2022 excluded from the analysis, source is Energy Information Administration; Refinery Capacity Report, June 21, 2022; https://www.eia.gov/petroleum/refinerycapacity/, original sheet represents Production capacity (MMSCFD or million standard cubic feet per day) on January 1 of each year, processed in kg-H2, using Hydrogen Tools data convertor 1 SCF = 0.002408 kg-H2

- industry_heat_demand_characterization_nrel.csv: Facility level process heat demand survey conducted by NREL. Combustion energy calculated from emissions data from the EPA. Energy values in TJ total, by fuel type, by temperature range. GHG emissions in MMTCO2e by fuel type, end use, temperature range (https://data.nrel.gov/submissions/91)

- h2_demand_heat_process.xlsx: results of pre-processing NREL heat process demand data to equivalent hydrogen demand in kg/day

## Code
- compute_plot_lcoh.ipynb : notebook for LCOH calculations and plots
- opt_deployment_[industry].py : python scripts for the optimization of ANR-H2 deployment at industrial sites
- res_be_calculations.ipynb : notebook for the computation of breakeven prices for Renewable Energy Sources
- pp_[industry].ipynb : post-processing notebooks for each industrial sector, creating deployment results figure for each sector
- opt_dedicated_electricity.py: optimization for dedicated electricity production from industrial results

- deployment_w_learning.ipynb: optimization of industrial deployment with cost reductions from learning

- analysis_be_demand.ipynb: notebook for plotting Breakeven Price VS hydrogen demand
- analysis_be_state.ipynb: notebook for plotting BR VS state-level NG prices



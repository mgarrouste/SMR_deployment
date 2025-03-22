import pandas as pd
from utils import palette, WACC
from math import log2

"""
This script reads the results from the FOAK deployment stage and computes the CAPEX of SMRs for the NOAK 
deployment stage using a learning rate of 7.5% and assuming the learning curve flattens out after 5 units deployed
"""

# Use a 7.5% learning rate
learning_rate = 0.075


def compute_SMR_deployment(with_inc):
    """
    Computes the total of SMR profitably deployed at the FOAK stage for each design
    In with_inc (bool) : whether the FOAK benefited from the 45V and 48E incentives
    """
    foak_r = pd.read_excel('./results/all_FOAK.xlsx')
    if with_inc: foak_r = foak_r[foak_r.tag == 'PTC_ITC']
    else: foak_r = foak_r[foak_r.tag == 'noPTC_noITC']
    # Select profitable sites: positive annual net revenues and IRR > WACC
    foak_r = foak_r[foak_r['Annual Net Revenues (M$/y)']>0]
    foak_r['IRR'] = foak_r['IRR'].astype(float)
    foak_r = foak_r[foak_r.IRR >= WACC*100]
    # Compute the total deoloyed units for each design
    smr_depl = {}
    for smr in palette.keys():
        smr_depl[smr] = foak_r[foak_r.SMR == smr]['# SMR modules'].sum()
    return smr_depl


def apply_learning_rate(smr_depl, with_inc):
    excel_file = './SMR_inputs.xlsx'
    foak_inputs = pd.read_excel(excel_file, sheet_name='FOAK', index_col='Type')
    noak_inputs = foak_inputs.copy()
    for smr,nb_units in smr_depl.items():
        # Learning flattens out after 5 units
        if nb_units >=5: nb_units =5
        foak_capex = foak_inputs.loc[smr, 'CAPEX $/MWe']
        if nb_units >0:
            noak_capex = foak_capex*(nb_units**log2(1-learning_rate))
        else:
            noak_capex = foak_capex
        noak_inputs.loc[smr, 'CAPEX $/MWe'] = noak_capex
    # Write NOAK
    if with_inc: sheet_name = 'NOAK_with_inc'
    else: sheet_name = 'NOAK_wo_inc'
    try:
        # Load the existing Excel file
        with pd.ExcelFile(excel_file, engine='openpyxl') as xls:
            # Check if the sheet exists
            if sheet_name in xls.sheet_names:
                # If the sheet exists, replace the data
                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    noak_inputs.to_excel(writer, sheet_name=sheet_name)
            else:
                # If the sheet doesn't exist, create a new sheet
                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                    noak_inputs.to_excel(writer, sheet_name=sheet_name)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and write the DataFrame to it
        noak_inputs.to_excel(excel_file, sheet_name=sheet_name)

def main():
    smr_with_inc = compute_SMR_deployment(with_inc=True)
    smr_wo_inc = compute_SMR_deployment(with_inc=False)
    apply_learning_rate(smr_with_inc, with_inc=True)
    apply_learning_rate(smr_wo_inc, with_inc=False)


if __name__ == '__main__':
    main()
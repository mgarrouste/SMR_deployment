import numpy as np
import matplotlib.pyplot as plt
from SALib import ProblemSpec
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from opt_deployment_ammonia import main as ammonia_dep

problem = {
    'num_vars': 3,
    'names': ["LR ANR CAPEX", "LR H2 CAPEX", 'WACC'],
    'bounds': [[0.03,0.10], [0.03,0.10], [0.05, 0.1]]
}

param_values = sobol_sample.sample(problem,1)
Y = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values): 
  lr_anr_capex, lr_h2_capex, wacc = X.T 
  mean_be_ng = ammonia_dep(learning_rate_anr_capex = lr_anr_capex, learning_rate_h2_capex =lr_h2_capex, wacc=wacc, print_results=False)
  #mean_be_ng = np.random.normal(20,5)
  Y[i] = mean_be_ng

sobol_indices = sobol_analyze.analyze(problem, Y)
total_Si, first_Si, second_Si = sobol_indices.to_df()

si_df = total_Si.merge(first_Si, left_index=True, right_index=True)
print(si_df)


fig, ax = plt.subplots()



si_df[['ST', 'S1']].plot.bar(yerr = si_df[['ST_conf', 'S1_conf']].apply(np.abs), ax=ax, capsize=5, rot=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()

import pandas as pd
import numpy as np

from scipy.stats import wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests


from test_significant import run_sig_global
from test_significant_fam import run_sig_fam



# GET RESULTS
full1, prior1, sim1 = run_sig_global()
fam1 = run_sig_fam()

models = {
    'full': full1,
    'prior': prior1,
    'sim': sim1,
    'fam': fam1
}


# RUN SIGNIFICANCE EVALUATION
results = []

for (name1, data1), (name2, data2) in combinations(models.items(), 2):
    stat, p = wilcoxon(data1, data2)
    results.append((name1, name2, p))

pvals = [r[2] for r in results]

reject, pvals_corrected, _, _ = multipletests(pvals, method='holm')


# CHECK - PRINT RESULTS
# for i, (name1, name2, p) in enumerate(results):
#     print(f"{name1} vs {name2}:")
#     print(f"  raw p = {p:.5f}")
#     print(f"  corrected p = {pvals_corrected[i]:.5f}")
#     print(f"  significant = {reject[i]}")

for i, (name1, name2, p) in enumerate(results):
    mean1 = np.mean(models[name1])
    mean2 = np.mean(models[name2])

    print(f"{name1} vs {name2}:")
    print(f"  means: {mean1:.3f} vs {mean2:.3f}")
    print(f"  corrected p = {pvals_corrected[i]:.5f}")

    if reject[i]:
        if mean1 > mean2:
            print(f"  → {name1} is SIGNIFICANTLY better")
        else:
            print(f"  → {name2} is SIGNIFICANTLY better")
    else:
        print("  → no significant difference")


    print()

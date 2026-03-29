import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_file import run_eval
from train_test import run_feval


# get the results
full_1, full_5 = run_eval('full')
prior_1, prior_5 = run_eval('prior')
sim_1, sim_5 = run_eval('similarity')
fam_1, fam_5 = run_feval()

fulls = [full_1, full_5]
priors = [prior_1, prior_5]
sims = [sim_1, sim_5]
fams = [fam_1, fam_5]
cats = ['top-1', 'top-5']


# make graph
w, x = 0.2, np.arange(len(cats))

fig, ax = plt.subplots()
b1 = ax.bar(x-0.3, fulls, width=w, label='P(T|S)', color='#1f497d')
b2 = ax.bar(x-0.1, sims, width=w, label='P(S|T)', color='#4f81bd')
b3 = ax.bar(x+0.1, priors, width=w, label='P(T)', color='#4bacc6')
b4 = ax.bar(x+0.3, fams, width=w, label='P(Tf|S)', color='#cadeefff')

for bars in [b1, b2, b3, b4]:
    ax.bar_label(bars, fmt='%.3f', padding=1)

ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel('Accuracy')
ax.set_title('Top-1 and Top-5 Accuracy Across Models')
ax.legend()

fig.savefig("plot1.png")
plt.show()

# print(f'FULL - top1: {full_1}, top5: {full_5}')
# print(f'PRIOR - top1: {prior_1}, top5: {prior_5}')
# print(f'SIM - top1: {sim_1}, top5: {sim_5}')
# print(f'FAM - top1: {fam_1}, top5: {fam_5}')


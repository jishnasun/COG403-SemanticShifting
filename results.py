import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from test_file_mod import run_eval, run_eval_lang
from train_test_mod import run_feval, run_feval_lang
from singLangTestCase import get_best_global, get_worst, get_best_fam

PTYPE = 0
# TYPES:
# 0 - overall bar plots
# 1 - single lang plots

PLOT = 1
# TYPES:
# 0 - don't generate plot
# 1 - generate plot


################################################################################
#################################### BAR PLOTS #################################
################################################################################
def plots(cols, data):
    fulls = [data['full'][0], data['full'][1]]
    priors = [data['prior'][0], data['prior'][1]]
    sims = [data['sim'][0], data['sim'][1]]
    fams = [data['fam'][0], data['fam'][1]]
    cats = ['top-1', 'top-5']

    w, x = 0.2, np.arange(len(cats))

    fig, ax = plt.subplots()

    b1 = ax.bar(x-0.3, fulls, width=w, label='P(T|S)')
    b2 = ax.bar(x-0.1, sims, width=w, label='P(S|T)')
    b3 = ax.bar(x+0.1, priors, width=w, label='P(T)')
    b4 = ax.bar(x+0.3, fams, width=w, label='P(Tf|S)')

    for bars in [b1, b2, b3, b4]:
        ax.bar_label(bars, fmt='%.3f', padding=1)

    ax.set_xticks(x)
    plt.yticks(np.arange(0, 1.1, step=0.2))
    ax.set_xticklabels(cats)
    ax.set_ylabel('Accuracy')
    ax.set_title('Top-1 and Top-5 Accuracy Across Models')

    if cols == 1:
        for bar in b1:
            bar.set_color('#1f497d')
        for bar in b2:
            bar.set_color('#4f81bd')
        for bar in b3:
            bar.set_color('#4bacc6')
        for bar in b4:
            bar.set_color('#cadeefff')

    ax.legend(loc='upper center', ncol=2)

    if cols == 1:
        fig.savefig(f"plot_overall2.0_c.png")
        plt.close(fig)
        return

    fig.savefig(f"plot_overall2.0.png")
    plt.close(fig)
    return


if PTYPE == 0:
    results = {
        'full': run_eval('full'),
        'prior': run_eval('prior'),
        'sim': run_eval('similarity'),
        'fam': run_feval()
    }

    if PLOT == 1:
        plots(0, results)
        plots(1, results)


################################################################################
############################### SINGLE LANG PLOTS ##############################
################################################################################
def single_plot(cols, data, label):
    print("LOCATION: single plot func")
    lang = data['language']

    fulls = [data['full'][0], data['full'][1]]
    priors = [data['prior'][0], data['prior'][1]]
    sims = [data['sim'][0], data['sim'][1]]
    fams = [data['fam'][0], data['fam'][1]]
    cats = ['top-1', 'top-5']

    w, x = 0.2, np.arange(len(cats))

    fig, ax = plt.subplots()

    b1 = ax.bar(x-0.3, fulls, width=w, label='P(T|S)')
    b2 = ax.bar(x-0.1, sims, width=w, label='P(S|T)')
    b3 = ax.bar(x+0.1, priors, width=w, label='P(T)')
    b4 = ax.bar(x+0.3, fams, width=w, label='P(Tf|S)')

    for bars in [b1, b2, b3, b4]:
        ax.bar_label(bars, fmt='%.3f', padding=1)

    ax.set_xticks(x)
    plt.yticks(np.arange(0, 1.1, step=0.2))
    ax.set_xticklabels(cats)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{lang}: Top-1 and Top-5 Accuracy Across Models')

    if cols == 1:
        for bar in b1:
            bar.set_color('#1f497d')
        for bar in b2:
            bar.set_color('#4f81bd')
        for bar in b3:
            bar.set_color('#4bacc6')
        for bar in b4:
            bar.set_color('#cadeefff')

    ax.legend(loc='upper center', ncol=2)

    if cols == 1:
        fig.savefig(f"plot_{label}_c.png")
        plt.close(fig)
        return

    fig.savefig(f"plot_{label}.png")
    plt.close(fig)
    return


if PTYPE == 1:
    results_best_full = get_best_global()
    results_worst = get_worst()
    results_best_fam = get_best_fam()

    results = {
        'best_full': results_best_full,
        'best_fam': results_best_fam,
        'worst': results_worst
    }

    if PLOT == 1:
        for label, data in results.items():
            single_plot(0, data, label)
            single_plot(1, data, label)



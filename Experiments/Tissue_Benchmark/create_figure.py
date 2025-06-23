import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import pandas as pd
import sys
import numpy as np
import os


plt.rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

p = '../../'
data_path = '../../'
sys.path.insert(0, p)
from CHESRA.plot_utils import dataset_colors, latex_emph, dataset_annotate, plot_sheardata, funcname_annotate


########################################################################################################################
# Plot results
########################################################################################################################

def color_boxplot(res, color):
    # Set the color for the boxes
    for box in res['boxes']:
        box.set_facecolor(color)

    # Optionally, customize other elements like whiskers, caps, medians, etc.
    for whisker in res['whiskers']:
        whisker.set_color(color)
    for cap in res['caps']:
        cap.set_color(color)
    for median in res['medians']:
        median.set_color(color)

    for flier in res['fliers']:
        flier.set_markerfacecolor(dataset_colors[data_colname])  # Set the face color of the outliers
        flier.set_markeredgecolor("k")  # Optionally set the edge color

params_dist_df = pd.read_csv('plot_data/params_dist_df.csv')
gof_dists_df = pd.read_csv('plot_data/gof_dists_df.csv')
sRSS_df = pd.read_csv('plot_data/sRSS_df.csv')


num_params = [3, 4, 5, 8, 7]
psi_names = ['CH1', 'CH2', 'MA', 'CL', 'HO']

param_names = {'CH1':['$p_{1}$', '$p_{2}$', '$p_{3}$'],
               'CH2':['$p_{1}$', '$p_{2}$', '$p_{3}$', '$p_{4}$'],
               'MA':['$\mu$', '$a_{f}$', '$b_{f}$', '$a_{n}$', '$b_{n}$'],
               'HO':['$a$', '$b$', '$a_{f}$', '$b_{f}$', '$a_{n}$', '$b_{n}$', '$a_{fs}$', '$b_{fs}$'],
               'CL':['$a$', '$b_{ff}$', '$b_{fn}$', '$b_{fs}$', '$b_{nn}$', '$b_{ns}$', '$b_{ss}$'],
               'SFL': ['$a_{ff}$', '$a_{fn}$', '$a_{fs}$', '$a_{nn}$', '$a_{ns}$', '$a_{ss}$', '$b_{ff}$', '$b_{fn}$',
                       '$b_{fs}$', '$b_{nn}$', '$b_{ns}$', '$b_{ss}$'],
               'PZL': ['$a_{ff}$', '$a_{fn}$', '$a_{fs}$', '$a_{nn}$', '$a_{ns}$', '$a_{ss}$', '$k_{ff}$', '$k_{fn}$']}

data_names = ['Shear Dokos', 'Shear Sommer']

fitplot_all_figure = plt.figure(figsize=(16, 18), constrained_layout=False)

best_or_worst = 'worst'

model_sheardata_df = []
for funcname in psi_names:
    shear_fit_func = pd.read_csv(os.path.join('plot_data/%s/' % (best_or_worst), 'data_shear_dokos_%s.csv' % (funcname))).drop(columns = "Unnamed: 0")
    shear_fit_func["function"] = funcname
#     print(shear_fit_func)
    model_sheardata_df.append(shear_fit_func)
model_sheardata_df = pd.concat(model_sheardata_df)

model_sheardata_sommer_df = []
for funcname in psi_names:
    shear_fit_func_sommer = pd.read_csv(os.path.join('plot_data/%s/' % (best_or_worst), 'data_shear_sommer_%s.csv' % (funcname))).drop(columns = "Unnamed: 0")
    shear_fit_func_sommer["function"] = funcname
    model_sheardata_sommer_df.append(shear_fit_func_sommer)
model_sheardata_sommer_df = pd.concat(model_sheardata_sommer_df)

dokos_shear_experiment_df = pd.read_csv("../../CHESRA/data/dokos/dokos_shear_experiment.csv")
sommer_shear_experiment_df = pd.read_csv("../../CHESRA/data/sommer/sommer_shear_experiment.csv")

# Font sizes
fs = 18
funcannot_size = fs

plt.rc('text', usetex=True)
plt.rc('font', size=fs)  # controls default text sizes
plt.rc('axes', titlesize=fs)  # fontsize of the axes title
plt.rc('axes', labelsize=fs)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)  # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)  # fontsize of the tick labels
plt.rc('legend', fontsize=fs)  # legend fontsize

# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

funcnames_latex = [r'${\psi}_\mathrm{%s}$' % label for label in psi_names]

################################
# Composition and Layout of plots
################################

outer_grid_pad_w = 0.3
outer_grid_pad_h = 0.55

inner_grid_pad_w = 0.2
inner_grid_pad_h = 0.35

outer_grid = fitplot_all_figure.add_gridspec(2, 1,
                                             height_ratios=[8, 1],
                                             hspace=0.15)

paramsfit_grid = outer_grid[0].subgridspec(1, 4,
                                           wspace=outer_grid_pad_w,
                                           hspace=outer_grid_pad_h,
                                           width_ratios=[2, 1, 2, 1])

gofdist_grid = outer_grid[1].subgridspec(1, 2,
                                         wspace=0.1,
                                         hspace=outer_grid_pad_h)

gofdist_axs = gofdist_grid.subplots(sharex=True, sharey=True)

sommer_params_grid = paramsfit_grid[0].subgridspec(len(psi_names), 1,
                                                   wspace=inner_grid_pad_w,
                                                   hspace=inner_grid_pad_h)

sommer_params_axs = sommer_params_grid.subplots()

dokos_params_grid = paramsfit_grid[2].subgridspec(len(psi_names), 1,
                                                  wspace=inner_grid_pad_w,
                                                  hspace=inner_grid_pad_h)

dokos_params_axs = dokos_params_grid.subplots()

sommer_fit_grid = paramsfit_grid[1].subgridspec(len(psi_names), 1,
                                                wspace=inner_grid_pad_w,
                                                hspace=inner_grid_pad_h)

sommer_fit_axs = sommer_fit_grid.subplots(sharex=True, sharey=True)

dokos_fit_grid = paramsfit_grid[3].subgridspec(len(psi_names), 1,
                                               wspace=inner_grid_pad_w,
                                               hspace=inner_grid_pad_h)

dokos_fit_axs = dokos_fit_grid.subplots(sharex=True,
                                        sharey=True)

###########################
# Annotations
###########################
for i, funcname in enumerate(funcnames_latex):
    funcname_annotate(sommer_params_axs[i],
                      funcname,
                      funcannot_size + 3)

dataset_annotate(sommer_params_axs[0],
                 "a",
                 "k",
                 funcannot_size,
                 y=1.25,
                 x=-0.32)

dataset_annotate(sommer_params_axs[0],
                 "Sommer Shear",
                 dataset_colors["Sommer Shear"],
                 funcannot_size,
                 y=1.25,
                 x=0.0)

dataset_annotate(dokos_params_axs[0],
                 "Dokos Shear",
                 dataset_colors["Dokos Shear"],
                 funcannot_size,
                 y=1.25,
                 x=0.0)

dataset_annotate(sommer_params_axs[0],
                 "Parameter Distributions",
                 dataset_colors["Sommer Shear"],
                 funcannot_size,
                 y=1.05,
                 x=0.0)

dataset_annotate(dokos_params_axs[0],
                 "Parameter Distributions",
                 dataset_colors["Dokos Shear"],
                 funcannot_size,
                 y=1.05,
                 x=0.0)

dataset_annotate(sommer_fit_axs[0],
                 "Model-Data Fit",
                 dataset_colors["Sommer Shear"],
                 funcannot_size,
                 y=1.05,
                 x=-0.1)

dataset_annotate(dokos_fit_axs[0],
                 "Model-Data Fit",
                 dataset_colors["Dokos Shear"],
                 funcannot_size,
                 y=1.05,
                 x=-0.1)

dataset_annotate(gofdist_axs[0],
                 "b",
                 "k",
                 funcannot_size,
                 y=1.3,
                 x=-0.18)

dataset_annotate(gofdist_axs[0],
                 "Goodness of Fit Distributions",
                 dataset_colors["Sommer Shear"],
                 funcannot_size,
                 y=1.05,
                 x=0.0)

dataset_annotate(gofdist_axs[1],
                 "Goodness of Fit Distributions",
                 dataset_colors["Dokos Shear"],
                 funcannot_size,
                 y=1.05,
                 x=0)

dokos_fit_axs[-1].set_xlabel("amount of shear", fontsize=fs)
sommer_fit_axs[-1].set_xlabel("amount of shear", fontsize=fs)

for i in range(len(psi_names)):
    dokos_fit_axs[i].set_ylabel("shear stress ($\sigma$)", fontsize=fs)
    sommer_fit_axs[i].set_ylabel("shear stress ($\sigma$)", fontsize=fs)

# dokos_params_axs[0].set_title("Dokos Shear Parameter Distribution", ha='left', x=-0, size = funcannot_size)
# sommer_params_axs[0].set_title("Sommer Shear Parameter Distribution", ha='left', x=-0, size = funcannot_size)

# dokos_fit_axs[0].set_title("Model-data Fit", ha='left', x=-0, size = funcannot_size, color = dataset_colors["Dokos Shear"])
# sommer_fit_axs[0].set_title("Model-data Fit", ha='left', x=-0, size = funcannot_size)

###########################
# Data plotting
###########################
for j, dataset, axs in zip(range(2),
                           ["Shear Sommer", "Shear Dokos"],
                           [sommer_params_axs, dokos_params_axs]):

    for i, funcname in enumerate(psi_names):
        nfevs_df = gof_dists_df[gof_dists_df['model'] == funcname]
        nfev = (np.mean(nfevs_df[nfevs_df['dataset'] == dataset]['nfev'].values))

        param_df = params_dist_df[(params_dist_df['model'] == funcname) & (params_dist_df['dataset'] == dataset)]
        if funcname == 'MA':
            print(param_df)

        s_params = []
        for param_name in param_names[funcname]:
            param_lst = param_df[param_df['variable'] == param_name]['value'].values

            s_params.append((np.std(param_lst) / np.average(param_lst)))
        sav = np.average(s_params)


        if funcname == 'PZL' or funcname == 'SFL':
            rot = -90
        else:
            rot = 0
        model_sommer_df = params_dist_df.query("model == '{}' and dataset == '{}'".format(funcname, dataset))
        ax = axs[i]
        ax.annotate(r'$\overline{c}_p = %.1f,$ $\overline{n}_\mathrm{fev}= %.0f$' % (sav, nfev),
                        (0.05, 0.92), xycoords="axes fraction", size=fs, va='top')

        data_colname = " ".join(np.flip(dataset.split()))
        flierprops = dict(marker='d', markerfacecolor='grey', markersize=3)
        res = ax.boxplot([np.array(model_sommer_df.query("variable == '{}'".format(varname))["value"]) for varname in
                          param_names[funcname]],
                         patch_artist=True, medianprops=dict(color='k'), flierprops=flierprops)

        for pc in res['boxes']:
            data_colname = " ".join(np.flip(dataset.split()))
            pc.set_facecolor(dataset_colors[data_colname])
            pc.set_alpha(0.6)

        ax.set_yscale('symlog')

        ax.set_ylabel("")
        ax.set_xlabel("")
        num_p = len(param_names[funcname])
        ax.set_xticks(list(range(1, num_p + 1)),
                      param_names[funcname],
                      ha='center',
                      rotation=rot)

for axs in [sommer_params_axs, dokos_params_axs]:
    for i in range(4):
        axs[i].set_yticks([0, 10, 1000, 100000])
        # sommer_params_axs[i].set_yticks([0, 1, 10, 100])

for i, funcname in enumerate(psi_names):
    ax = dokos_fit_axs[i]
    plot_sheardata(ax,
                   model_sheardata_df.query("function == '{}'".format(funcname)),
                   dokos_shear_experiment_df,
                   dataset_colors["Dokos Shear"],
                   lim=[1.02, 1.01, 1.01, 1.05, 1.5, 1.5], markersize=3, alpha=0.5
                   )
    sRSS = sRSS_df[(sRSS_df['model'] == funcname) & (sRSS_df['dataset'] == 'Shear Dokos')]['sRSS'].values[0]

    num_print = '%0.1E' % sRSS
    ax.annotate(r'$f_\mathrm{GoF} = $' + '\n' + r'%s' % (num_print),
                    (0.05, 0.92),
                    xycoords="axes fraction", size=fs, va='top')
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_ylim((-0.5, 15))

for i, funcname in enumerate(psi_names):
    ax = sommer_fit_axs[i]
    plot_sheardata(ax,
                   model_sheardata_sommer_df.query("function == '{}'".format(funcname)),
                   sommer_shear_experiment_df,
                   dataset_colors["Sommer Shear"],
                   lim=[1.05] * 6, markersize=3, alpha=0.3)
    sRSS = sRSS_df[(sRSS_df['model'] == funcname) & (sRSS_df['dataset'] == 'Shear Sommer')]['sRSS'].values[0]
    num_print = '%0.1E' % sRSS
    ax.annotate(r'$f_\mathrm{GoF} = $' + '\n' + r'%s' % (num_print),
                    (0.05, 0.92),
                    xycoords="axes fraction", size=fs, va='top')
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_ylim((-0.25,7))

for i, dataset in enumerate(["Shear Sommer", "Shear Dokos"]):
    gofdist_data = [np.array(gof_dists_df.query("model == '{}' and dataset == '{}'".format(modelname, dataset))["GoF"])
                    for modelname in psi_names]

    # print(gofdist_data)
    data_colname = " ".join(np.flip(dataset.split()))
    res = gofdist_axs[i].boxplot(gofdist_data, patch_artist=True, medianprops=dict(color='k'), flierprops=flierprops)

    for pc in res['boxes']:
        data_colname = " ".join(np.flip(dataset.split()))
        pc.set_facecolor(dataset_colors[data_colname])
        pc.set_alpha(0.6)


    gofdist_axs[i].set_xticks(list(range(1, len(psi_names) + 1)),
                              funcnames_latex,
                              ha='center',
                              rotation=0)

    #for pc in res['bodies']:
    #    data_colname = " ".join(np.flip(dataset.split()))
    #    pc.set_facecolor(dataset_colors[data_colname])
    #    pc.set_edgecolor(dataset_colors[data_colname])
    #res["cmedians"].set_color("k")

gofdist_axs[0].set_ylabel("goodness of fit ($f_\mathrm{GoF}$)")
gofdist_axs[0].set_yscale('log')
gofdist_axs[0].set_yticks(10.0 ** np.arange(-3, 8, 2))


plt.tight_layout()
plt.show()
plt.savefig('Fig5_InverseTest.png')
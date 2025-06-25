from matplotlib import pyplot as plt
import pandas as pd
import sys
import matplotlib
matplotlib.use('TkAgg')

p = '../../'
sys.path.insert(0, p)
from CHESRA.plot_utils import dataset_colors, dataset_annotate, plot_sheardata, plot_biaxial_data, add_legend_biaxial, \
    dokos_shear_annotate_curves, sommer_shear_annotate_curves

plt.rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


yin_biaxial_model_df = pd.read_csv("plot_data/data_biaxial_yin.csv").drop(columns = "Unnamed: 0")
yin_biaxial_experiment_df = pd.read_csv("../../CHESRA/data/yin/yin_biaxial_experiment.csv")
yin_biaxial_experiment_df["r"] = yin_biaxial_experiment_df[["r"]].applymap(lambda x: '{0:.2f}'.format(x))
data_markers_yin_biaxial = {"2.05": "^",
                            "1.02": "s",
                            "0.48": "o"}

sommer_biaxial_model_df = pd.read_csv("plot_data/data_biaxial_sommer.csv").drop(columns = "Unnamed: 0")
sommer_biaxial_experiment_df = pd.read_csv("../../CHESRA/data/sommer/sommer_biaxial_experiment.csv")
sommer_biaxial_experiment_df["r"] = sommer_biaxial_experiment_df[["r"]].applymap(lambda x: '{0:.2f}'.format(x))

data_markers_sommer_biaxial = {"0.50": "o",
                               "0.75": "s",
                               "1.00": "D",
                               "1.33": "*",
                               "2.00": "^"}

dokos_shear_model_df = pd.read_csv("plot_data/data_shear_dokos.csv").drop(columns = "Unnamed: 0")
dokos_shear_experiment_df = pd.read_csv("../../CHESRA/data/dokos/dokos_shear_experiment.csv")

sommer_shear_model_df = pd.read_csv("plot_data/data_shear_sommer.csv").drop(columns = "Unnamed: 0")
sommer_shear_experiment_df = pd.read_csv("../../CHESRA/data/sommer/sommer_shear_experiment.csv")

novak_biaxial_model_df = pd.read_csv("plot_data/data_biaxial_novak.csv").drop(columns = "Unnamed: 0")
novak_biaxial_experiment_df = pd.read_csv("../../CHESRA/data/novak/novak_biaxial_experiment.csv")
novak_biaxial_experiment_df["protocol_otherextension"] = novak_biaxial_experiment_df[["protocol_otherextension"]].applymap(lambda x: '{0:.2f}'.format(x))

data_markers_novak_biaxial = {"1.20": "o",
                              "1.15": "s",
                              "1.10": "D"}

novak_equibiaxial_model1_df = pd.read_csv("plot_data/data_equibiax_novak1.csv")
novak_equibiaxial_model2_df = pd.read_csv("plot_data/data_equibiax_novak2.csv")
novak_equibiaxial_experiment_df = pd.read_csv("../../CHESRA/data/novak/novak_equibiaxial_experiment.csv")

data_markers_novak_equibiaxial = {"septum": "o",
                                  "sub-epi": "s",
                                  "mid-myo": "^",
                                  "sub-endo": "*"}

novak_equibiaxial_experiment_df["location"].unique()


fitplot_all_figure = plt.figure(figsize = (18, 11), constrained_layout=False)

#Font sizes
#plt.rcParams.update({'font.size': 16})
funcannot_size = 18
fs = 20

plt.rc('text', usetex=True)
plt.rc('font', size=fs)          # controls default text sizes
plt.rc('axes', titlesize=fs)     # fontsize of the axes title
plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
plt.rc('legend', fontsize=fs)    # legend fontsize

#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


with open("plot_data/optimal_function.txt", "r") as f:
    optfunc = f.readlines()[0]

################################
#Composition and Layout of plots
################################
outer_grid_pad_w = 0.2
outer_grid_pad_h = 0.55

inner_grid_pad_w = 0.1
inner_grid_pad_h = 0.25

outer_grid = fitplot_all_figure.add_gridspec(3, 3, 
                                             wspace=outer_grid_pad_w,
                                             hspace=outer_grid_pad_h)

novak_biaxial_grid = outer_grid[1, 0].subgridspec(1, 2,
                                                  wspace=inner_grid_pad_w,
                                                  hspace=inner_grid_pad_h)
novak_biaxial_axs = novak_biaxial_grid.subplots(sharex = True, sharey = True)

shear_grid = outer_grid[2,:].subgridspec(1, 2, wspace=0.1, hspace=inner_grid_pad_h)
shear_axs = shear_grid.subplots(sharex = True)

yin_biaxial_grid = outer_grid[1, 1].subgridspec(1, 2, wspace=inner_grid_pad_w, hspace=0.4)
yin_biaxial_axs = yin_biaxial_grid.subplots(sharey = True, sharex = True)

sommer_biaxial_grid = outer_grid[1, 2].subgridspec(1, 2, wspace=inner_grid_pad_w, hspace=0.4)
sommer_biaxial_axs = sommer_biaxial_grid.subplots(sharey = True, sharex = True)

novak_equibiaxial_grid = outer_grid[0, :].subgridspec(1, 4, wspace=inner_grid_pad_w, hspace=inner_grid_pad_h)
novak_equibiaxial_axs = novak_equibiaxial_grid.subplots(sharey = True, sharex = True)

################################
#Call Plotting Functions
################################
plot_biaxial_data(yin_biaxial_axs,
                 "Yin Biaxial",
                  data_markers_yin_biaxial,
                  "r",
                  yin_biaxial_model_df,
                  yin_biaxial_experiment_df)

plot_biaxial_data(sommer_biaxial_axs,
                 "Sommer Biaxial",
                  data_markers_sommer_biaxial,
                  "r",
                  sommer_biaxial_model_df,
                  sommer_biaxial_experiment_df)

plot_sheardata(shear_axs[1], 
               dokos_shear_model_df, 
               dokos_shear_experiment_df,
               dataset_colors["Dokos Shear"], lim=[1.02,1.01,1.01,1.05,1.5,1.5])

dokos_shear_annotate_curves([shear_axs[1]],
                             dokos_shear_experiment_df,
                             pad = 0.5)

plot_sheardata(shear_axs[0], 
               sommer_shear_model_df, 
               sommer_shear_experiment_df,
               dataset_colors["Sommer Shear"])

sommer_shear_annotate_curves([shear_axs[0]],
                              sommer_shear_experiment_df, pad = 0.4)

plot_biaxial_data(novak_biaxial_axs,
                 "Novak Biaxial",
                  data_markers_novak_biaxial,
                  "protocol_otherextension",
                  novak_biaxial_model_df,
                  novak_biaxial_experiment_df)

plot_biaxial_data(novak_equibiaxial_axs,
                 "Novak Equibiaxial",
                  data_markers_novak_equibiaxial,
                  "location",
                  novak_equibiaxial_model1_df,
                  novak_equibiaxial_experiment_df.query("specimen == 1"))

plot_biaxial_data(novak_equibiaxial_axs,
                 "Novak Equibiaxial",
                  data_markers_novak_equibiaxial,
                  "location",
                  novak_equibiaxial_model2_df,
                  novak_equibiaxial_experiment_df.query("specimen == 2"),
                  x_ax_offset =2)


###############################
#X-Y axis adjustments
###############################
yin_biaxial_axs[0].set_xlabel("strain $(E_{ff})$")
yin_biaxial_axs[1].set_xlabel("strain $(E_{ss})$")
yin_biaxial_axs[0].set_ylabel("stress ($S$) [kPa]") 
yin_biaxial_axs[0].set_yticks([0, 10, 20]) 
yin_biaxial_axs[0].set_xticks([0, 0.1,0.2]) 
yin_biaxial_axs[0].set_xlim(0, 0.22) 

sommer_biaxial_axs[0].set_ylabel("stress ($S$) [kPa]")
sommer_biaxial_axs[0].set_xlabel("strain ($E_{ff}$)")
sommer_biaxial_axs[1].set_xlabel("strain ($E_{ss}$)")

shear_axs[1].set_ylim(-1, 18)
shear_axs[1].set_yticks([0, 5, 10, 15])
shear_axs[1].set_xlim(0, 0.6)
shear_axs[0].set_xlim(0, 0.6)
shear_axs[0].set_ylim(-0.4, 7)

shear_axs[0].set_xlabel("amount of shear ($\gamma$)")
shear_axs[1].set_xlabel("amount of shear ($\gamma$)")
shear_axs[0].set_ylabel("stress ($\sigma$) [kPa]")

novak_biaxial_axs[0].set_xlim(1.0, 1.35)
novak_biaxial_axs[0].set_xticks([1.0, 1.1, 1.2, 1.3])
novak_biaxial_axs[0].set_xlabel("extension ($\lambda_{f}$)")
novak_biaxial_axs[1].set_xlabel("extension ($\lambda_{s}$)")
novak_biaxial_axs[0].set_ylabel("stress ($\sigma$) [kPa]")

novak_equibiaxial_axs[0].set_xlim(1.0, 1.35)
novak_equibiaxial_axs[0].set_xticks([1.0, 1.1, 1.2, 1.3])

novak_equibiaxial_axs[0].set_xlabel("extension ($\lambda_{f}$)")
novak_equibiaxial_axs[1].set_xlabel("extension ($\lambda_{s}$)")
novak_equibiaxial_axs[2].set_xlabel("extension ($\lambda_{f}$)")
novak_equibiaxial_axs[3].set_xlabel("extension ($\lambda_{s}$)")
novak_equibiaxial_axs[0].set_ylabel("stress ($\sigma$) [kPa]")

###############################
#Subplot Titles and annotations
###############################
dataset_annotate(yin_biaxial_axs[0],
                 r"\textbf{Yin Biaxial}",
                 dataset_colors["Yin Biaxial"],
                 funcannot_size)

dataset_annotate(sommer_biaxial_axs[0],
                 r"\textbf{Sommer Biaxial}",
                 dataset_colors["Sommer Biaxial"],
                 funcannot_size)

dataset_annotate(shear_axs[1],
                 r"\textbf{Dokos Shear}",
                 dataset_colors["Dokos Shear"],
                 funcannot_size)

dataset_annotate(shear_axs[0],
                 r"\textbf{Sommer Shear}",
                 dataset_colors["Sommer Shear"],
                 funcannot_size)

dataset_annotate(novak_biaxial_axs[0],
                 r"\textbf{Novak Biaxial}",
                 dataset_colors["Novak Biaxial"],
                 funcannot_size)

dataset_annotate(novak_equibiaxial_axs[0],
                 r"\textbf{Novak Equibiaxial}",
                 dataset_colors["Novak Equibiaxial"],
                 funcannot_size)

novak_equibiaxial_axs[0].annotate(r"\textbf{specimen 1}",
                                    (0.05, 0.85),
                                    xycoords = "axes fraction")#.set_title("specimen 1", ha='left', x=-0)
novak_equibiaxial_axs[2].annotate(r"\textbf{specimen 2}",
                                    (0.05, 0.85),
                                    xycoords = "axes fraction")#.set_title("specimen 2", ha='left', x=-0

###############################
#Legends
###############################
add_legend_biaxial(yin_biaxial_axs[0], 
                   data_markers_yin_biaxial,
                   r"\textbf{ratio}", 
                   dataset_colors["Yin Biaxial"], bbox=(-0.12, 1.03))

add_legend_biaxial(sommer_biaxial_axs[0], 
                   data_markers_sommer_biaxial, 
                   r"\textbf{ratio}", 
                   dataset_colors["Sommer Biaxial"], bbox=(-0.12, 1.03))

add_legend_biaxial(novak_biaxial_axs[0], 
                   {"1.15": data_markers_novak_biaxial["1.15"]}, 
                   r"$\boldsymbol{\lambda_{s}}$", 
                   dataset_colors["Novak Biaxial"], bbox=(-0.12, 1.02))

add_legend_biaxial(novak_biaxial_axs[1], 
                   data_markers_novak_biaxial, 
                   r"$\boldsymbol{\lambda_{f}}$", 
                   dataset_colors["Novak Biaxial"], bbox=(-0.12, 1.03))

add_legend_biaxial(novak_equibiaxial_axs[1],
                   data_markers_novak_equibiaxial,
                   r"\textbf{location}",
                   dataset_colors["Novak Equibiaxial"], bbox=(-0.07,1.02))
fitplot_all_figure.suptitle(optfunc)
plt.savefig("Fig4_optimal_chesra_fits.png", bbox_inches = "tight")
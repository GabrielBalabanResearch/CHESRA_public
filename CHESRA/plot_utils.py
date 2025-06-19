from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np


hue_order = ["Dokos Shear",
             "Sommer Shear",
             "Sommer Biaxial",
             "Novak Biaxial",
             "Novak Equibiaxial",
             "Yin Biaxial"]

palett = sns.color_palette("dark{}".format(len(hue_order)))
dataset_colors = dict(zip(hue_order, palett))

def latex_emph(text):
    return "$\\textrm{\\textbf{" + text + "}}$"

def funcname_annotate(ax, funcname, funcannot_size):
    ax.annotate(latex_emph(funcname),
                (-0.32, 1.0),
                xycoords = "axes fraction",
                annotation_clip = False,
                size = funcannot_size,
                rotation = "horizontal",
                ha = "left")

def dataset_annotate(ax, dataset_name, color, funcannot_size, x = 0, y = 1.05):
    ax.annotate(latex_emph(dataset_name),
                (x, y),
                 xycoords = "axes fraction",
                 color = color,
                 annotation_clip = False,
                 size = funcannot_size)


def plot_sheardata(ax, model_df, experiment_df, datacolor, markersize = 7, alpha=1, lim=[1.1]*6):
    modes = ["fs", "fn", "sn", "sf", "ns", "nf"]
    for i, mode in enumerate(modes):
        expr_modedata_df = experiment_df.query("mode == '{}'".format(mode))
        
        #Data
        ax.plot(expr_modedata_df["x"].array, 
                expr_modedata_df["y"].array,
                markersize = markersize,
                markeredgewidth = 2,
                marker = "o",
                markerfacecolor='none',
                color = datacolor,
                linestyle = 'None',
                zorder = 10,
                alpha=alpha)
        
        #Model
        expr_modedata_df = experiment_df.query("mode == '{}'".format(mode))
        #         model_cutoff_df = model_df[model_df[mode] <= 1.05*expr_modedata_df["y"].max()]
        model_cutoff_df = model_df[model_df['x'] <= lim[i]*expr_modedata_df["x"].max()]
        ax.plot(model_cutoff_df["x"].array, 
                model_cutoff_df[mode].array,
                color = "k",
                linestyle = '-',
                zorder = 20)

def add_legend_biaxial(ax, marker_dict, marker_var, color, markersize=7, bbox=(-0.15, 1.05)):
    #     data_legend_marks = [Line2D([0], [0],
    #                          #marker = "none",
    #                          linestyle = 'None',
    #                          label = marker_var)] +\
    data_legend_marks = [Line2D([0], [0],
                                color=color,
                                marker=marker_dict[mval],
                                linestyle='none',
                                markerfacecolor='none',
                                markersize=markersize,
                                label=mval) for mval in marker_dict.keys()]

    legend = ax.legend(loc="upper left",
                       handles=data_legend_marks,
                       #               fancybox=True,
                       #               shadow=True,
                       borderpad=0.1,
                       handletextpad=0.0,
                       bbox_to_anchor=bbox,
                       columnspacing=0,
                       frameon=False,
                       labelspacing=0, title=marker_var)
    legend._legend_box.align = "center"


#     frame = legend.get_frame()
#     frame.set_edgecolor('black')


def plot_biaxial_data(axs, dataset_name, marker_dict, marker_var, model_df, experiment_df, markersize=7, x_ax_offset=0):
    for i_strain, strain in enumerate(["ff", "ss"]):
        experiment_protocol_df = experiment_df.query("strain == '{}'".format(strain))
        # experiment_protocol_df[marker_var] = experiment_protocol_df[marker_var].astype(str)
        for marker_varval in experiment_protocol_df[marker_var].unique():
            querystr = "{} == '{}'".format(marker_var, marker_varval)
            # print(querystr)
            xy_experiment_df = experiment_protocol_df.query(querystr)

            i_x = i_strain + x_ax_offset
            axs[i_x].plot(xy_experiment_df["x"].array,
                          xy_experiment_df["y"].array,
                          marker=marker_dict[marker_varval],
                          markersize=markersize,
                          markerfacecolor="none",
                          color=dataset_colors[dataset_name],
                          linestyle='None', zorder=10)

            y_string = "{}, {}".format(strain, marker_varval)
            x_cutoff = model_df["x"] <= 1.02 * xy_experiment_df["x"].max()
            y_cutoff = model_df[y_string] <= 1.2 * xy_experiment_df["y"].max()

            model_cutoff_df = model_df[np.logical_and(x_cutoff, y_cutoff)]

            axs[i_x].plot(model_cutoff_df["x"].array,
                          model_cutoff_df[y_string].array,
                          color="k", zorder=20)


def dokos_shear_annotate_curves(axs, dokos_shear_experiment_df, pad=0.5):
    # Annotations
    for ax in axs:
        for mode in ["fs", "fn", "sf"]:
            expr_modedata_df = dokos_shear_experiment_df.query("mode == '{}'".format(mode))
            if mode == 'fs':
                ax.annotate("(" + mode + ")", (0.52, expr_modedata_df["y"].max() - pad))
            elif mode == 'sf':
                ax.annotate("(" + mode + ")", (0.52, expr_modedata_df["y"].max() + 2 * pad))
            else:
                ax.annotate("(" + mode + ")", (0.52, expr_modedata_df["y"].max()))

        ax.annotate("(sn)", (0.52, dokos_shear_experiment_df.query("mode == 'sn'")["y"].max() + 1.5 * pad))
        ax.annotate("(ns, nf)", (0.52, dokos_shear_experiment_df.query("mode == 'ns'")["y"].max() - 1.5 * pad))


def sommer_shear_annotate_curves(axs, sommer_shear_experiment_df, pad=0.2):
    # Annotations
    for ax in axs:
        # for mode in ["fs", "fn"]:
        #    expr_modedata_df = sommer_shear_experiment_df.query("mode == '{}'".format(mode))
        ax.annotate("(fs)", (0.52, sommer_shear_experiment_df.query("mode == 'fs'")["y"].max() + 0.5 * pad))
        ax.annotate("(fn)", (0.52, sommer_shear_experiment_df.query("mode == 'fn'")["y"].max() - pad))

        ax.annotate("(ns, nf)", (0.52, sommer_shear_experiment_df.query("mode == 'ns'")["y"].max() - pad))
        ax.annotate("(sf, sn)", (0.52, sommer_shear_experiment_df.query("mode == 'sn'")["y"].max() + pad))
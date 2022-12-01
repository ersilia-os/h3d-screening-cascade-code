import matplotlib.pyplot as plt
from matplotlib.pyplot import Rectangle
from stylia import NamedColors
from sklearn import metrics

import numpy as np

def draw_pie_marker(ax, xs, ys, ratios, sizes, colors):
   assert sum(ratios) <= 1, 'sum of ratios needs to be < 1'
   markers = []
   radians_start = 0
   # calculate the points of the pie pieces
   for color, ratio in zip(colors, ratios):
      radians_end = 2 * np.pi * ratio + radians_start
      x = [0] + np.cos(np.linspace(radians_start, radians_end, 100)).tolist() + [0]
      y = [0] + np.sin(np.linspace(radians_start, radians_end, 100)).tolist() + [0]
      xy = np.column_stack([x, y])
      radians_start = radians_end #mark the starting point for the next wedge
      markers.append({
      'marker': xy,
      's': np.abs(xy).max() ** 2 * np.array(sizes),
      'facecolor': color
      })
   # scatter each of the pie pieces to create pies
   for marker in markers:
      ax.scatter(xs, ys, ** marker, alpha=0.85)


def hit_enrichment(data, proba_col):
    data = data.copy(deep=True)
    data.sort_values(by=proba_col, axis=0, ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)
    total_mols = len(data)
    active_mols = len(data[data["bin"]==1])
    frac_tested = np.arange(1,total_mols+1, 1)/total_mols
    hit_enr = [x/active_mols for x in np.cumsum(data["bin"])]
    bins = [x+1 for x in range((active_mols))]+[active_mols]*(len(data[data["bin"]==0]))
    ideal=[x/active_mols for x in bins]
    random =  np.cumsum([active_mols/total_mols]*total_mols) / active_mols
    return frac_tested, hit_enr, ideal, random



def precision50(ax, x_r, y_r, c_r, x_rank_top, y_rank_top, c_rank_top, x_rank_bot, y_rank_bot, c_rank_bot):
    ax.scatter(x=x_r, y=y_r, c=c_r, marker="",)
    ax.scatter(x=x_rank_top, y=y_rank_top, c=c_rank_top, marker="")
    ax.scatter(x=x_rank_bot, y=y_rank_bot, c=c_rank_bot, marker="")

    # Add rectangles
    width = 1
    height = 1
    a=0
    for x_a, y_a in zip(x_r, y_r):
        ax.add_patch(Rectangle(
            xy=(x_a-(width/2), y_a) ,width=width, height=height,
            linewidth=0, color=c_r[a], fill=True))
        a = a+1
    a=0
    for x_b, y_b in zip(x_rank_top, y_rank_top):
        ax.add_patch(Rectangle(
            xy=(x_b-(width/2), y_b) ,width=width, height=height,
            linewidth=0, color=c_rank_top[a], fill=True))
        a = a+1

    a=0
    for x_c, y_c in zip(x_rank_bot, y_rank_bot):
        ax.add_patch(Rectangle(
            xy=(x_c-(width/2), y_c) ,width=width, height=height,
            linewidth=0, color=c_rank_bot[a], fill=True))
        a = a+1

    ax.set_xticks([-2.5, 0, 2.5])
    ax.set_xticklabels(["Random", "Top", "Bottom"])
    ytick_list = [i for i in range(1,51)]
    odds_yticks = [n for n in ytick_list if n % 2 == 1]
    ax.set_yticks(odds_yticks)
    reverse_yticks = list(reversed(odds_yticks))
    ax.set_yticklabels(reverse_yticks)
    ax.set_ylim([1,50])
    ax.set_xlim([-4, 4])

def scores_plot(ax, y_true, y_pred):
    noise = [np.random.uniform(-0.2, 0.2) for x in range(len(y_pred))]
    y_0 = []
    x_0 = []
    y_1 = []
    x_1 = []
    for i in range(len(y_pred)):
        if y_true[i] == 0:
            y_0 += [y_pred[i]]
            x_0 += [0 + noise[i]]
        else:
            y_1 += [y_pred[i]]
            x_1 += [1 + noise[i]]
    ax.scatter(x_0, y_0, color=NamedColors().red, alpha=0.5, s=20)
    ax.scatter(x_1, y_1, color=NamedColors().blue, alpha=0.5, s=20)
    ax.set_xlabel("")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Inactive", "Active"])
    ax.set_ylabel("Score")
    ax.set_xlim(-0.5,1.5)
    return ax

def roc(ax, btrue, proba1, assayname):
    fpr, tpr, _ = metrics.roc_curve(btrue, proba1)
    auc = metrics.roc_auc_score(btrue, proba1)
    ax.plot(fpr, tpr, lw=0.5, color = NamedColors().blue)
    ax.grid()
    ax.tick_params(axis="both", which ="major")
    ax.set_xlabel("1-Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(assayname)
    ax.legend(loc = 'lower right')
    return fpr, tpr, auc, ax

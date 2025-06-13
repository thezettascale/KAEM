import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True, 
    'font.family': 'serif', 
    'font.serif': ['Compute Modern'], 
    'axes.unicode_minus': False, 
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{bm} \newcommand{\probP}{\text{I\kern-0.15em P}}'  
})

latent_dim = pd.DataFrame({
    r"$n_z$" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "Time (s)" : [5.915, 13.075, 19.707, 25.572, 31.971, 39.648, 47.857, 56.762, 63.697, 71.476],
    "Memory Estimate (GiB)" : [0.551, 0.870, 1.18, 1.52, 1.89, 2.26, 2.65, 3.05, 3.49, 3.93],
    "Garbage Collection (%)" : [0.00, 1.24, 1.20, 1.52, 1.45, 1.65, 1.64, 1.60, 1.64, 1.62], 
    "Allocations" : [11085826, 21500020, 32371238, 43698173, 55485415, 67724846, 80420074, 93571538, 107179137, 121242782],
})

mala_steps = pd.DataFrame({
    r"$N_{t},\\(N_{\text{local}}=20)$" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Time (s)" : [2.972, 6.482, 10.111, 13.434, 17.672, 21.740, 25.657, 29.173, 33.484, 37.891],
    "Memory Estimate (GiB)" : [1.66, 2.25, 2.85, 3.45, 4.04, 4.64, 5.24, 5.84, 6.43, 7.02],
    "Garbage Collection (%)" : [6.68, 1.87, 1.81, 1.74, 7.69, 8.03, 8.50, 7.82, 8.96, 9.12],
    "Allocations" : [1203094, 2133663, 3061363, 3989154, 4916228, 5933549, 6962088, 7989669, 9018787, 10047865],
})

keys = [r"$n_z$", r"$N_{t},\\(N_{\text{local}}=20)$"]
colours = ["autumn", "Wistia"]
elevations = [0.545, 0.545]

def add_text_annotations(ax, round=False, elevation=0.45):
    for bar in ax.patches:
        if round:
            text = f'{bar.get_height():.0f}'
        else:
            text = f'{bar.get_height():.2f}'

        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            elevation * ax.get_ylim()[1], 
            text, ha='center', va='bottom', color='blue', 
            rotation=45, fontsize=7  
        )

for (idx, df) in enumerate([latent_dim, mala_steps]):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    selected_colors = sns.color_palette(colours[idx], len("Time (s)") + 1)
    colors = [selected_colors[0], selected_colors[2], selected_colors[3], selected_colors[4]]

    sns.barplot(
        x = keys[idx],
        y = "Time (s)",
        data = df,
        ax = axs[0, 0],
        palette = colors
    )
    add_text_annotations(axs[0, 0], round=False, elevation=elevations[idx])

    axs[0, 0].set_xlabel(keys[idx])
    axs[0, 0].set_ylabel("Time (s)")
    
    sns.barplot(
        x = keys[idx],
        y = "Memory Estimate (GiB)",
        data = df,
        ax = axs[0, 1],
        palette = colors
    )
    add_text_annotations(axs[0, 1], round=False, elevation=elevations[idx])

    axs[0, 1].set_xlabel(keys[idx])
    axs[0, 1].set_ylabel("Memory Estimate (GiB)")

    sns.barplot(
        x = keys[idx],
        y = "Garbage Collection (%)",
        data = df,
        ax = axs[1, 0],
        palette = colors
    )
    add_text_annotations(axs[1, 0], round=False, elevation=elevations[idx])

    axs[1, 0].set_xlabel(keys[idx])
    axs[1, 0].set_ylabel("Garbage Collection (\%)")
    
    sns.barplot(
        x = keys[idx],
        y = "Allocations",
        data = df,
        ax = axs[1, 1],
        palette = colors
    )
    add_text_annotations(axs[1, 1], round=True, elevation=elevations[idx])

    axs[1, 1].set_xlabel(keys[idx])
    axs[1, 1].set_ylabel("Allocations")
    
    plt.tight_layout()
    plt.savefig(f"figures/benchmark/plot_cost_{idx}.png", dpi=300, bbox_inches='tight')

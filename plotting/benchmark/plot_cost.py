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
    r"$n_z$" : [10, 20, 40, 80],
    "Time (s)" : [12.538, 27.069, 52.124, 116.419],
    "Memory Estimate (GiB)" : [0.765, 1.37, 2.75, 6.11],
    "Garbage Collection (%)" : [1.23, 1.29, 1.50, 1.50],
    "Allocations" : [19761865, 39969224, 84700446, 191387478],
})

mala_steps = pd.DataFrame({
    r"$N_{\text{local}}$" : [5, 10, 15, 20],
    "Time (s)" : [115.341, 219.694, 328.726, 434.245],
    "Memory Estimate (GiB)" : [5.07, 8.54, 12.09, 15.59],
    "Garbage Collection (%)" : [6.60, 9.56, 11.10, 11.58],
    "Allocations" : [135688338, 225947183, 318703993, 409591568],
})

keys = [r"$n_z$", r"$N_{\text{local}}$"]
colours = ["viridis", "cividis"]
elevations = [0.45, 0.1]

def add_text_annotations(ax, round=False, elevation=0.45):
    for bar in ax.patches:
        if round:
            text = f'{bar.get_height():.0f}'
        else:
            text = f'{bar.get_height():.2f}'

        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            elevation * ax.get_ylim()[1], 
            text, ha='center', va='bottom', color='red', 
            rotation=45
        )

for (idx, df) in enumerate([latent_dim, mala_steps]):
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
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
    
    


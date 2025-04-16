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
    "Time (s)" : [6.887, 14.235, 29.748, 66.514],
    "Memory Estimate (GiB)" : [0.868, 1.49, 2.95, 6.64],  
    "Garbage Collection (%)" : [0.00, 1.56, 1.64, 1.85],
    "Allocations" : [19991483, 40666450, 87218646, 201161187],
})

mala_steps = pd.DataFrame({
    r"$N_{\text{local}}$ variable \\ ($N_{\text{unadjusted}}=1, N_{t}=5$)" : [1, 5, 10, 15],
    "Time (s)" : [5.776, 69.266, 150.791, 226.335],
    "Memory Estimate (GiB)" : [1.97, 12.94, 26.46, 39.71],
    "Garbage Collection (%)" : [5.30, 3.83, 3.69, 3.62],
    "Allocations" : [22407508, 301594303, 645196852, 980779216],
})

keys = [r"$n_z$", r"$N_{\text{local}}$ variable \\ ($N_{\text{unadjusted}}=1, N_{t}=5$)"]
colours = ["viridis", "cividis"]
elevations = [0.2, 0.08]

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
    
    


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
    r"$n_z$" : [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "Time (s)" : [13.075, 19.707, 25.572, 31.971, 39.648, 47.857, 56.762, 63.697, 71.476],
    "Memory Estimate (GiB)" : [0.870, 1.18, 1.52, 1.89, 2.26, 2.65, 3.05, 3.49, 3.93],
    "Garbage Collection (%)" : [1.24, 1.20, 1.52, 1.45, 1.65, 1.64, 1.60, 1.64, 1.62], 
    "Allocations" : [21500020, 32371238, 43698173, 55485415, 67724846, 80420074, 93571538, 107179137, 121242782],
})

mala_steps_ref = {
    "Time (s)": 2.972,
    "Memory Estimate (GiB)": 1.66,
    "Garbage Collection (%)": 6.68,
    "Allocations": 1203094
}

mala_steps = pd.DataFrame({
    r"$N_{t},\\(N_{\text{local}}=20)$" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Time (s)" : [6.482, 10.111, 13.434, 17.672, 21.740, 25.657, 29.173, 33.484, 37.891],
    "Memory Estimate (GiB)" : [2.25, 2.85, 3.45, 4.04, 4.64, 5.24, 5.84, 6.43, 7.02],
    "Garbage Collection (%)" : [1.87, 1.81, 1.74, 7.69, 8.03, 8.50, 7.82, 8.96, 9.12],
    "Allocations" : [2133663, 3061363, 3989154, 4916228, 5933549, 6962088, 7989669, 9018787, 10047865],
})

prior_steps = pd.DataFrame({
    "N_{prior},(N_t=1)": [10, 20, 30, 40, 50, 60, 80, 90, 100],
    "Time (s)": [4.069, 4.132, 4.220, 4.230, 4.209, 4.163, 4.181, 4.222, 4.152],
    "Memory Estimate (GiB)": [1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26],
    "Allocations": [1275585, 1275990, 1275567, 1275563, 1275711, 1275724, 1275564, 1275559, 1275561]
})

keys = [r"$n_z$", r"$N_{t},\\(N_{\text{local}}=20)$"]
colours = ["viridis", "cividis"]
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
            text, ha='center', va='bottom', color='red', 
            rotation=45, fontsize=14
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
    
    if idx == 1:  
        axs[0, 0].axhline(y=mala_steps_ref["Time (s)"], color='red', linestyle='--', 
                          label='MLE criterion with single posterior ULA sampling')
        axs[0, 0].legend()

    axs[0, 0].set_xlabel(keys[idx], fontsize=12)
    axs[0, 0].set_ylabel("Time (s)", fontsize=12)
    
    sns.barplot(
        x = keys[idx],
        y = "Memory Estimate (GiB)",
        data = df,
        ax = axs[0, 1],
        palette = colors
    )
    add_text_annotations(axs[0, 1], round=False, elevation=elevations[idx])
    
    if idx == 1:  
        axs[0, 1].axhline(y=mala_steps_ref["Memory Estimate (GiB)"], color='red', linestyle='--',
                          label='MLE criterion with single posterior ULA sampling')
        axs[0, 1].legend()

    axs[0, 1].set_xlabel(keys[idx], fontsize=14)
    axs[0, 1].set_ylabel("Memory Estimate (GiB)", fontsize=14)

    sns.barplot(
        x = keys[idx],
        y = "Garbage Collection (%)",
        data = df,
        ax = axs[1, 0],
        palette = colors
    )
    add_text_annotations(axs[1, 0], round=False, elevation=elevations[idx])
    
    # Add reference line for N_t = 1 if this is the mala_steps plot
    if idx == 1:  # mala_steps plot
        axs[1, 0].axhline(y=mala_steps_ref["Garbage Collection (%)"], color='red', linestyle='--',
                          label='MLE criterion with single posterior ULA sampling')
        axs[1, 0].legend()

    axs[1, 0].set_xlabel(keys[idx], fontsize=14)
    axs[1, 0].set_ylabel("Garbage Collection (\%)", fontsize=14)
    
    sns.barplot(
        x = keys[idx],
        y = "Allocations",
        data = df,
        ax = axs[1, 1],
        palette = colors
    )
    
    # Add reference line for N_t = 1 if this is the mala_steps plot
    if idx == 1:  # mala_steps plot
        axs[1, 1].axhline(y=mala_steps_ref["Allocations"], color='red', linestyle='--',
                          label='MLE criterion with single posterior ULA sampling')
        axs[1, 1].legend()

    axs[1, 1].set_xlabel(keys[idx], fontsize=14)
    axs[1, 1].set_ylabel("Allocations", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"figures/benchmark/plot_cost_{idx}.png", dpi=300, bbox_inches='tight')

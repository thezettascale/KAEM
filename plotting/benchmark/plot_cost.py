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

# Resized dataset to (28, 28, 1)
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 12.325 s (1.20% GC) to evaluate,
#  with a memory estimate of 819.62 MiB, over 21760537 allocations.
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 25.240 s (1.30% GC) to evaluate,
#  with a memory estimate of 1.48 GiB, over 44223592 allocations.
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 40.650 s (1.76% GC) to evaluate,
#  with a memory estimate of 2.23 GiB, over 68530209 allocations.
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 52.286 s (1.59% GC) to evaluate,
#  with a memory estimate of 3.04 GiB, over 94673691 allocations.
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 66.928 s (1.61% GC) to evaluate,
#  with a memory estimate of 3.93 GiB, over 122657531 allocations.
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 82.800 s (1.63% GC) to evaluate,
#  with a memory estimate of 4.87 GiB, over 152481722 allocations.
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 112.640 s (1.56% GC) to evaluate,
#  with a memory estimate of 5.87 GiB, over 184145695 allocations.


latent_dim = pd.DataFrame({
    r"$n_z$" : [10, 20, 30, 40, 50, 60, 70],
    "Time (s)" : [12.325, 25.240, 40.650, 52.286, 66.928, 82.800, 112.640],
    "Memory Estimate (GiB)" : [0.819, 1.48, 2.23, 3.04, 3.93, 4.87, 5.87],  
    "Garbage Collection (%)" : [0.00, 1.30, 1.76, 1.59, 1.61, 1.63, 1.56],
    "Allocations" : [21760537, 44223592, 68530209, 94673691, 122657531, 152481722, 184145695],
})

mala_steps = pd.DataFrame({
    r"$N_{\text{local}}$ variable \\ ($N_{\text{unadjusted}}=1, N_{t}=5$)" : [1, 5, 10, 15],
    "Time (s)" : [3.678, 82.855, 199.496, 321.181],
    "Memory Estimate (GiB)" : [0.914, 12.48, 29.08, 46.17], 
    "Garbage Collection (%)" : [4.31, 3.81, 3.76, 3.76],
    "Allocations" : [16673999, 372935723, 882573427, 1412633594],
})

keys = [r"$n_z$", r"$N_{\text{local}}$ variable \\ ($N_{\text{unadjusted}}=1, N_{t}=5$)"]
colours = ["autumn", "cividis"]
elevations = [0.24, 0.04]

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
            rotation=45
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
    
    


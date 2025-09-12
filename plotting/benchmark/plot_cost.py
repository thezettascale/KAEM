import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Compute Modern"],
        "axes.unicode_minus": False,
        "text.latex.preamble": (
            r"\usepackage{amsmath} "
            r"\usepackage{amsfonts} "
            r"\usepackage{amssymb} "
            r"\usepackage{bm} "
            r"\newcommand{\probP}{\text{I\kern-0.15em P}}"
        ),
    }
)

latent_dim_df = pd.read_csv("benches/results/latent_dim.csv")
temperatures_df = pd.read_csv("benches/results/temperatures.csv")
prior_steps_df = pd.read_csv("benches/results/prior_steps.csv")
its_single_df = pd.read_csv("benches/results/ITS_single.csv")

mala_steps_ref = {
    "Time (s)": its_single_df.iloc[0]["time_mean"],
    "Memory Estimate (GiB)": its_single_df.iloc[0]["memory_estimate"],
    "Garbage Collection (%)": its_single_df.iloc[0]["gc_percent"],
    "Allocations": its_single_df.iloc[0]["allocations"],
}

latent_dim = pd.DataFrame(
    {
        r"$n_z$": latent_dim_df["n_z"],
        "Time (s)": latent_dim_df["time_mean"],
        "Memory Estimate (GiB)": latent_dim_df["memory_estimate"],
        "Garbage Collection (%)": latent_dim_df["gc_percent"],
        "Allocations": latent_dim_df["allocations"],
    }
)

mala_steps = pd.DataFrame(
    {
        r"$N_{t}$": temperatures_df["N_t"],
        "Time (s)": temperatures_df["time_mean"],
        "Memory Estimate (GiB)": temperatures_df["memory_estimate"],
        "Garbage Collection (%)": temperatures_df["gc_percent"],
        "Allocations": temperatures_df["allocations"],
    }
)

prior_steps = pd.DataFrame(
    {
        r"$N_{prior}$": prior_steps_df["N_l"],
        "Time (s)": prior_steps_df["time_mean"],
        "Memory Estimate (GiB)": prior_steps_df["memory_estimate"],
        "Garbage Collection (%)": prior_steps_df["gc_percent"],
        "Allocations": prior_steps_df["allocations"],
    }
)

prior_steps_ref = {
    "Time (s)": its_single_df.iloc[0]["time_mean"],
    "Memory Estimate (GiB)": its_single_df.iloc[0]["memory_estimate"],
    "Garbage Collection (%)": its_single_df.iloc[0]["gc_percent"],
    "Allocations": its_single_df.iloc[0]["allocations"],
}

keys = [r"$n_z$", r"$N_{t}$", r"$N_{prior}$"]
colours = ["viridis", "cividis", "plasma"]
elevations = [0.3, 0.35, 0.4]
datasets = [latent_dim, mala_steps, prior_steps]
references = [None, mala_steps_ref, prior_steps_ref]
titles = [r"Latent Dimension", r"Power Posteriors", "Prior ULA Steps"]


def add_text_annotations(ax, round=False, elevation=0.45):
    for bar in ax.patches:
        if round:
            text = f"{bar.get_height():.0f}"
        else:
            text = f"{bar.get_height():.2f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            elevation * ax.get_ylim()[1],
            text,
            ha="center",
            va="bottom",
            color="red",
            rotation=45,
            fontsize=14,
        )


for idx, (df, ref, title) in enumerate(zip(datasets, references, titles)):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    colors = sns.color_palette(colours[idx], n_colors=len(df))

    sns.barplot(x=keys[idx], y="Time (s)", data=df, ax=axs[0, 0], color=colors[0])
    add_text_annotations(axs[0, 0], round=False, elevation=elevations[idx])

    if ref is not None:
        if idx == 1:  # MALA steps plot
            axs[0, 0].axhline(
                y=ref["Time (s)"],
                color="red",
                linestyle="--",
                label=(
                    "Reference MLE criterion\n"
                    + "with single posterior\n"
                    + "ULA sampling"
                ),
            )
        else:
            axs[0, 0].axhline(
                y=ref["Time (s)"],
                color="red",
                linestyle="--",
                label=("Reference ITS\n" + "with Mixture Prior"),
            )
        axs[0, 0].legend()

    axs[0, 0].set_xlabel(keys[idx], fontsize=12)
    axs[0, 0].set_ylabel("Time (s)", fontsize=12)
    axs[0, 0].set_title(f"{title} - Time", fontsize=14)

    sns.barplot(
        x=keys[idx],
        y="Memory Estimate (GiB)",
        data=df,
        ax=axs[0, 1],
        color=colors[1] if len(colors) > 1 else colors[0],
    )
    add_text_annotations(axs[0, 1], round=False, elevation=elevations[idx])

    if ref is not None:
        if idx == 1:  # MALA steps plot
            axs[0, 1].axhline(
                y=ref["Memory Estimate (GiB)"],
                color="red",
                linestyle="--",
                label=(
                    "Reference MLE criterion\n"
                    + "with single posterior\n"
                    + "ULA sampling"
                ),
            )
        else:
            axs[0, 1].axhline(
                y=ref["Memory Estimate (GiB)"],
                color="red",
                linestyle="--",
                label=("Reference ITS\n" + "with Mixture Prior"),
            )
        axs[0, 1].legend()

    axs[0, 1].set_xlabel(keys[idx], fontsize=14)
    axs[0, 1].set_ylabel("Memory Estimate (GiB)", fontsize=14)
    axs[0, 1].set_title(f"{title} - Memory", fontsize=14)

    sns.barplot(
        x=keys[idx],
        y="Garbage Collection (%)",
        data=df,
        ax=axs[1, 0],
        color=colors[2] if len(colors) > 2 else colors[0],
    )
    add_text_annotations(axs[1, 0], round=False, elevation=elevations[idx])

    if ref is not None:
        if idx == 1:  # MALA steps plot
            axs[1, 0].axhline(
                y=ref["Garbage Collection (%)"],
                color="red",
                linestyle="--",
                label=(
                    "Reference MLE criterion\n"
                    + "with single posterior\n"
                    + "ULA sampling"
                ),
            )
        else:
            axs[1, 0].axhline(
                y=ref["Garbage Collection (%)"],
                color="red",
                linestyle="--",
                label=("Reference ITS\n" + "with Mixture Prior"),
            )
        axs[1, 0].legend()

    axs[1, 0].set_xlabel(keys[idx], fontsize=14)
    axs[1, 0].set_ylabel(r"Garbage Collection (\%)", fontsize=14)
    axs[1, 0].set_title(f"{title} - GC", fontsize=14)

    sns.barplot(
        x=keys[idx],
        y="Allocations",
        data=df,
        ax=axs[1, 1],
        color=colors[3] if len(colors) > 3 else colors[0],
    )

    if ref is not None:
        if idx == 1:  # MALA steps plot
            axs[1, 1].axhline(
                y=ref["Allocations"],
                color="red",
                linestyle="--",
                label=("Reference ITS\n" + "with Mixture Prior"),
            )
        else:
            axs[1, 1].axhline(
                y=ref["Allocations"],
                color="red",
                linestyle="--",
                label=("Reference ITS\n" + "with Mixture Prior"),
            )
        axs[1, 1].legend()

    axs[1, 1].set_xlabel(keys[idx], fontsize=14)
    axs[1, 1].set_ylabel("Allocations", fontsize=14)
    axs[1, 1].set_title(f"{title} - Allocations", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        f"figures/benchmark/plot_cost_{idx}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

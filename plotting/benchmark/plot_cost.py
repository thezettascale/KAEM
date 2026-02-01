from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.unicode_minus": False,
        "text.latex.preamble": (
            r"\usepackage{amsmath} "
            r"\usepackage{amsfonts} "
            r"\usepackage{amssymb} "
            r"\usepackage{bm} "
        ),
    }
)

RESULTS_DIR = Path("benches/results")
FIGURES_DIR = Path("figures/benchmark")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Time (s)", "Memory Estimate (GiB)", "Garbage Collection (%)", "Allocations"]
METRIC_TITLES = ["Time", "Memory", "GC", "Allocations"]


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load CSV file if it exists, return None otherwise."""
    if path.exists():
        return pd.read_csv(path)
    print(f"Warning: {path} not found, skipping.")
    return None


def plot_single_benchmark(
    df: pd.DataFrame,
    x_key: str,
    title: str,
    output_name: str,
    reference: dict | None = None,
    reference_label: str = "",
    colour_palette: str = "viridis",
    time_std_col: str | None = "Time Std (s)",
):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    colors = sns.color_palette(colour_palette, n_colors=max(4, len(df)))

    for idx, (metric, metric_title) in enumerate(zip(METRICS, METRIC_TITLES)):
        ax = axs[idx]
        color = colors[min(idx, len(colors) - 1)]

        # Add error bars if std is available
        if (
            metric == "Time (s)"
            and time_std_col is not None
            and time_std_col in df.columns
        ):
            ax.bar(
                range(len(df)),
                df[metric],
                yerr=df[time_std_col],
                color=color,
                capsize=5,
                error_kw={"elinewidth": 2, "capthick": 2},
            )
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x_key], fontsize=14)
        else:
            sns.barplot(x=x_key, y=metric, data=df, ax=ax, color=color)

        if reference is not None and metric in reference:
            ax.axhline(
                y=reference[metric],
                color="red",
                linestyle="--",
                linewidth=2,
                label=reference_label,
            )
            ax.legend(fontsize=12, loc="upper left")

        ax.set_xlabel(x_key, fontsize=18)
        ax.set_ylabel(metric, fontsize=18)
        ax.set_title(f"{title} - {metric_title}", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)

        if metric == "Garbage Collection (%)":
            ax.set_ylim(0, max(1, df[metric].max() * 1.1))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison(
    kaem_df: pd.DataFrame,
    vae_df: pd.DataFrame,
    kaem_key: str,
    vae_key: str,
    title: str,
    output_name: str,
):
    """Plot KAEM vs VAE comparison with grouped bars side by side."""
    fig, axs = plt.subplots(1, 4, figsize=(22, 6))

    col_rename = {
        "time_mean": "Time (s)",
        "time_std": "Time Std (s)",
        "memory_estimate": "Memory Estimate (GiB)",
        "gc_percent": "Garbage Collection (%)",
        "allocations": "Allocations",
    }

    kaem_data = kaem_df.rename(columns=col_rename).copy()
    kaem_data["Model"] = "KAEM"

    # Convert n_z to Q = 2*n_z + 1 (the actual latent dimension)
    kaem_data["Latent Dim"] = 2 * kaem_df[kaem_key] + 1

    vae_data = vae_df.rename(columns=col_rename).copy()
    vae_data["Model"] = "VAE"

    # VAE already uses Q directly as latent_dim
    vae_data["Latent Dim"] = vae_df[vae_key]

    combined = pd.concat([kaem_data, vae_data], ignore_index=True)

    palette = {"KAEM": "#2ecc71", "VAE": "#3498db"}
    latent_dims = sorted(kaem_data["Latent Dim"].unique())

    for idx, (metric, metric_title) in enumerate(zip(METRICS, METRIC_TITLES)):
        ax = axs[idx]

        if metric == "Time (s)" and "Time Std (s)" in kaem_data.columns:
            x = np.arange(len(latent_dims))
            width = 0.35

            kaem_times = [
                kaem_data[kaem_data["Latent Dim"] == ld]["Time (s)"].values[0]
                for ld in latent_dims
            ]
            kaem_stds = [
                kaem_data[kaem_data["Latent Dim"] == ld]["Time Std (s)"].values[0]
                for ld in latent_dims
            ]
            vae_times = [
                vae_data[vae_data["Latent Dim"] == ld]["Time (s)"].values[0]
                for ld in latent_dims
            ]
            vae_stds = [
                vae_data[vae_data["Latent Dim"] == ld]["Time Std (s)"].values[0]
                for ld in latent_dims
            ]

            ax.bar(
                x - width / 2,
                kaem_times,
                width,
                yerr=kaem_stds,
                label="KAEM",
                color=palette["KAEM"],
                capsize=4,
                error_kw={"elinewidth": 2, "capthick": 2},
            )
            ax.bar(
                x + width / 2,
                vae_times,
                width,
                yerr=vae_stds,
                label="VAE",
                color=palette["VAE"],
                capsize=4,
                error_kw={"elinewidth": 2, "capthick": 2},
            )

            ax.set_xticks(x)
            ax.set_xticklabels(latent_dims, fontsize=14)
        else:
            sns.barplot(
                x="Latent Dim",
                y=metric,
                hue="Model",
                data=combined,
                ax=ax,
                palette=palette,
            )

        ax.set_xlabel(r"Latent Dim, $Q$", fontsize=18)
        ax.set_ylabel(metric, fontsize=18)
        ax.set_title(f"{title} - {metric_title}", fontsize=18)
        ax.legend(fontsize=14)
        ax.tick_params(axis="both", labelsize=14)

        if metric == "Garbage Collection (%)":
            ax.set_ylim(0, max(1, combined[metric].max() * 1.1))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sampling_time_only(
    kaem_df: pd.DataFrame,
    vae_df: pd.DataFrame,
    kaem_key: str,
    vae_key: str,
    output_name: str,
):
    """Plot just the sampling time comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    col_rename = {
        "time_mean": "Time (s)",
        "time_std": "Time Std (s)",
    }

    kaem_data = kaem_df.rename(columns=col_rename).copy()
    kaem_data["Latent Dim"] = 2 * kaem_df[kaem_key] + 1

    vae_data = vae_df.rename(columns=col_rename).copy()
    vae_data["Latent Dim"] = vae_df[vae_key]

    palette = {"KAEM": "#2ecc71", "VAE": "#3498db"}
    latent_dims = sorted(kaem_data["Latent Dim"].unique())

    x = np.arange(len(latent_dims))
    width = 0.35

    kaem_times = [
        kaem_data[kaem_data["Latent Dim"] == ld]["Time (s)"].values[0]
        for ld in latent_dims
    ]
    kaem_stds = [
        kaem_data[kaem_data["Latent Dim"] == ld]["Time Std (s)"].values[0]
        for ld in latent_dims
    ]
    vae_times = [
        vae_data[vae_data["Latent Dim"] == ld]["Time (s)"].values[0]
        for ld in latent_dims
    ]
    vae_stds = [
        vae_data[vae_data["Latent Dim"] == ld]["Time Std (s)"].values[0]
        for ld in latent_dims
    ]

    ax.bar(
        x - width / 2,
        kaem_times,
        width,
        yerr=kaem_stds,
        label="KAEM",
        color=palette["KAEM"],
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )
    ax.bar(
        x + width / 2,
        vae_times,
        width,
        yerr=vae_stds,
        label="VAE",
        color=palette["VAE"],
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(latent_dims, fontsize=16)
    ax.set_xlabel(r"Latent Dim, $Q$", fontsize=20)
    ax.set_ylabel("Time (s)", fontsize=20)
    ax.set_title("Sampling Time: KAEM vs VAE", fontsize=22)
    ax.legend(fontsize=16)
    ax.tick_params(axis="both", labelsize=16)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    latent_dim_df = load_csv_safe(RESULTS_DIR / "latent_dim.csv")
    temperatures_df = load_csv_safe(RESULTS_DIR / "temperatures.csv")
    its_sampling_df = load_csv_safe(RESULTS_DIR / "ITS_generation.csv")

    vae_latent_dim_df = load_csv_safe(RESULTS_DIR / "vae_latent_dim.csv")
    vae_sampling_df = load_csv_safe(RESULTS_DIR / "vae_sampling.csv")

    ref_nz40 = None
    if latent_dim_df is not None:
        nz40_row = latent_dim_df[latent_dim_df["n_z"] == 40]
        if not nz40_row.empty:
            ref_nz40 = {
                "Time (s)": nz40_row.iloc[0]["time_mean"],
                "Memory Estimate (GiB)": nz40_row.iloc[0]["memory_estimate"],
                "Garbage Collection (%)": nz40_row.iloc[0]["gc_percent"],
                "Allocations": nz40_row.iloc[0]["allocations"],
            }

    # Plot 1: KAEM vs VAE Training Comparison (side by side)
    if latent_dim_df is not None and vae_latent_dim_df is not None:
        plot_comparison(
            latent_dim_df,
            vae_latent_dim_df,
            "n_z",
            "latent_dim",
            "Training Cost: KAEM vs VAE",
            "01_kaem_vs_vae_training_comparison.png",
        )
        print("Saved: 01_kaem_vs_vae_training_comparison.png")

    # Plot 2: KAEM vs VAE Sampling Comparison (side by side)
    if its_sampling_df is not None and vae_sampling_df is not None:
        plot_comparison(
            its_sampling_df,
            vae_sampling_df,
            "n_z",
            "latent_dim",
            "Sampling Cost: KAEM vs VAE",
            "02_kaem_vs_vae_sampling_comparison.png",
        )
        print("Saved: 02_kaem_vs_vae_sampling_comparison.png")

        # Plot 2b: Just the sampling time comparison
        plot_sampling_time_only(
            its_sampling_df,
            vae_sampling_df,
            "n_z",
            "latent_dim",
            "02b_sampling_time_only.png",
        )
        print("Saved: 02b_sampling_time_only.png")

    # Plot 3: Thermodynamic Integration (Power Posteriors)
    if temperatures_df is not None:
        temps = pd.DataFrame(
            {
                r"$N_{t}$": temperatures_df["N_t"],
                "Time (s)": temperatures_df["time_mean"],
                "Time Std (s)": temperatures_df["time_std"],
                "Memory Estimate (GiB)": temperatures_df["memory_estimate"],
                "Garbage Collection (%)": temperatures_df["gc_percent"],
                "Allocations": temperatures_df["allocations"],
            }
        )
        ref_label = r"Reference KAEM ($Q=81$, $N_t=1$)"
        plot_single_benchmark(
            temps,
            r"$N_{t}$",
            "Power Posteriors",
            "03_kaem_training_vs_num_temperatures.png",
            reference=ref_nz40,
            reference_label=ref_label,
            colour_palette="viridis",
        )
        print("Saved: 03_kaem_training_vs_num_temperatures.png")


if __name__ == "__main__":
    main()

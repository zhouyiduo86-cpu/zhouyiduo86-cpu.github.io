"""
Biodiversity risk pipeline for reptile body-mass vs. IUCN conservation status analysis.

Ingests Amniote life-history and IUCN Red List datasets, merges on standardized
scientific names, classifies specimens into binary hazard tiers, runs bootstrap
simulation of mean body-mass deltas, and exports diagnostic figures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISSING_VALUE: Final[int] = -999
REPTILIA_CLASS: Final[str] = "reptilia"
BODY_MASS_COLUMN: Final[str] = "adult_body_mass_g"
SCIENTIFIC_NAME_KEY: Final[str] = "scientific_name"
HAZARD_TIER_COLUMN: Final[str] = "hazard_tier"

HIGHER_RISK_LABEL: Final[str] = "Higher Risk"
LOWER_RISK_LABEL: Final[str] = "Lower Risk"

# Granular IUCN indices mapped to binary hazard tiers (Vulnerable and above = Higher Risk).
HIGHER_RISK_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"ex", "ew", "cr", "en", "vu"}
)
LOWER_RISK_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"lc", "nt", "lr/lc", "lr/nt", "lr/cd", "lr"}
)

# Corporate palette aligned with portfolio styling.
COLOR_NAVY: Final[str] = "#0a1628"
COLOR_NAVY_MID: Final[str] = "#1e3a5f"
COLOR_SLATE: Final[str] = "#64748b"
COLOR_SLATE_LIGHT: Final[str] = "#94a3b8"
COLOR_OFF_WHITE: Final[str] = "#f8f9fb"
COLOR_WHITE: Final[str] = "#ffffff"
COLOR_ACCENT: Final[str] = "#2563eb"


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for file paths and simulation parameters."""

    amniote_path: Path
    iucn_path: Path
    output_dir: Path
    bootstrap_iterations: int = 5_000
    random_seed: int = 42
    figure_dpi: int = 300


# ---------------------------------------------------------------------------
# Data ingestion & transformation
# ---------------------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with lowercased, stripped column names."""
    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip().str.lower()
    return normalized


def _build_scientific_name(genus: pd.Series, species: pd.Series) -> pd.Series:
    """
    Construct standardized merge keys from genus and species strings.

    Keys are lowercase, whitespace-trimmed, and formatted as ``'<genus> <species>'``.
    """
    genus_clean = genus.astype(str).str.strip().str.lower()
    species_clean = species.astype(str).str.strip().str.lower()
    return genus_clean + " " + species_clean


def _replace_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Map sentinel missing values (-999) to ``pd.NA`` across the frame."""
    return df.replace(MISSING_VALUE, pd.NA)


def _normalize_iucn_category(category: pd.Series) -> pd.Series:
    """
    Normalize IUCN category strings, condensing legacy ``Lower Risk/*`` variants.

    Examples
    --------
    ``'Lower Risk/least concern'`` → ``'lc'``
    ``'LR/nt'`` → ``'nt'``
    """
    normalized = category.astype(str).str.strip().str.lower()

    lr_mask = normalized.str.startswith("lower risk")
    normalized = normalized.where(
        ~lr_mask,
        normalized.str.replace("lower risk/", "", regex=False),
    )
    normalized = normalized.str.replace("least concern", "lc", regex=False)
    normalized = normalized.str.replace("near threatened", "nt", regex=False)
    normalized = normalized.str.replace("conservation dependent", "lc", regex=False)
    normalized = normalized.str.replace(" ", "", regex=False)

    return normalized


def load_and_transform_data(
    amniote_path: Path | str,
    iucn_path: Path | str,
) -> pd.DataFrame:
    """
    Load, clean, merge, and filter Amniote and IUCN reptile datasets.

    Parameters
    ----------
    amniote_path:
        Path to ``Amniote_Database_Aug_2015.csv``.
    iucn_path:
        Path to ``IUCN_animals.csv``.

    Returns
    -------
    pd.DataFrame
        Inner-joined reptile records with body mass, normalized IUCN category,
        and a binary ``hazard_tier`` column.

    Raises
    ------
    FileNotFoundError
        If either source file does not exist.
    ValueError
        If required columns are absent after normalization.
    """
    amniote_path = Path(amniote_path)
    iucn_path = Path(iucn_path)

    if not amniote_path.is_file():
        raise FileNotFoundError(f"Amniote database not found: {amniote_path}")
    if not iucn_path.is_file():
        raise FileNotFoundError(f"IUCN dataset not found: {iucn_path}")

    amniote_raw = pd.read_csv(amniote_path, low_memory=False)
    iucn_raw = pd.read_csv(iucn_path, low_memory=False)

    amniote = _replace_missing_values(_normalize_columns(amniote_raw))
    iucn = _normalize_columns(iucn_raw)

    required_amniote = {"class", "genus", "species", BODY_MASS_COLUMN}
    missing_amniote = required_amniote - set(amniote.columns)
    if missing_amniote:
        raise ValueError(f"Amniote file missing columns: {sorted(missing_amniote)}")

    category_col = "category" if "category" in iucn.columns else "iucn_category"
    if category_col not in iucn.columns:
        raise ValueError("IUCN file must contain a 'category' or 'iucn_category' column.")

    genus_col = "genus" if "genus" in iucn.columns else "genus_name"
    species_col = "species" if "species" in iucn.columns else "species_name"
    if genus_col not in iucn.columns or species_col not in iucn.columns:
        raise ValueError("IUCN file must contain genus and species identifier columns.")

    amniote_reptiles = amniote[
        amniote["class"].astype(str).str.strip().str.lower() == REPTILIA_CLASS
    ].copy()

    amniote_reptiles[SCIENTIFIC_NAME_KEY] = _build_scientific_name(
        amniote_reptiles["genus"],
        amniote_reptiles["species"],
    )

    iucn = iucn.copy()
    iucn[SCIENTIFIC_NAME_KEY] = _build_scientific_name(iucn[genus_col], iucn[species_col])
    iucn["category_normalized"] = _normalize_iucn_category(iucn[category_col])

    merged = amniote_reptiles.merge(
        iucn[[SCIENTIFIC_NAME_KEY, "category_normalized"]],
        on=SCIENTIFIC_NAME_KEY,
        how="inner",
    )

    merged = merged[merged["category_normalized"] != "dd"].copy()
    merged[BODY_MASS_COLUMN] = pd.to_numeric(merged[BODY_MASS_COLUMN], errors="coerce")
    merged = merged.dropna(subset=[BODY_MASS_COLUMN, "category_normalized"])

    merged[HAZARD_TIER_COLUMN] = merged["category_normalized"].map(_map_to_hazard_tier)
    merged = merged.dropna(subset=[HAZARD_TIER_COLUMN]).reset_index(drop=True)

    return merged


def _map_to_hazard_tier(category: str) -> str | pd.NA:
    """
    Map a normalized IUCN category code to a binary hazard tier label.

    Parameters
    ----------
    category:
        Normalized IUCN category (e.g. ``'vu'``, ``'lc'``, ``'lr/nt'``).

    Returns
    -------
    str | pd.NA
        ``'Higher Risk'``, ``'Lower Risk'``, or ``pd.NA`` for unmapped codes.
    """
    code = str(category).strip().lower()
    if code in HIGHER_RISK_CATEGORIES:
        return HIGHER_RISK_LABEL
    if code in LOWER_RISK_CATEGORIES:
        return LOWER_RISK_LABEL
    return pd.NA


# ---------------------------------------------------------------------------
# Bootstrap simulation
# ---------------------------------------------------------------------------


def run_bootstrap_simulation(df: pd.DataFrame, iterations: int = 5_000) -> list[float]:
    """
    Bootstrap resampled mean body-mass deltas between hazard tiers.

    Maps granular IUCN indices into binary tiers, then repeatedly resamples
    (with replacement) body-mass observations within each tier. Each iteration
    records the delta:

        mean(Higher Risk body mass) − mean(Lower Risk body mass)

    Parameters
    ----------
    df:
        Transformed dataset from :func:`load_and_transform_data`.
    iterations:
        Number of bootstrap resamples (default 5,000).

    Returns
    -------
    list[float]
        Resampled mean-difference outcomes across all iterations.

    Raises
    ------
    ValueError
        If either hazard tier has no usable body-mass observations.
    """
    higher_risk = (
        df.loc[df[HAZARD_TIER_COLUMN] == HIGHER_RISK_LABEL, BODY_MASS_COLUMN]
        .dropna()
        .to_numpy(dtype=float)
    )
    lower_risk = (
        df.loc[df[HAZARD_TIER_COLUMN] == LOWER_RISK_LABEL, BODY_MASS_COLUMN]
        .dropna()
        .to_numpy(dtype=float)
    )

    if higher_risk.size == 0 or lower_risk.size == 0:
        raise ValueError("Both hazard tiers must contain body-mass observations.")

    rng = np.random.default_rng()
    deltas = np.empty(iterations, dtype=float)

    for i in range(iterations):
        higher_sample = rng.choice(higher_risk, size=higher_risk.size, replace=True)
        lower_sample = rng.choice(lower_risk, size=lower_risk.size, replace=True)
        deltas[i] = higher_sample.mean() - lower_sample.mean()

    return deltas.tolist()


# ---------------------------------------------------------------------------
# Visualization & analytics
# ---------------------------------------------------------------------------


def _apply_corporate_style() -> None:
    """Configure matplotlib defaults for clean corporate chart styling."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_OFF_WHITE,
            "axes.facecolor": COLOR_WHITE,
            "axes.edgecolor": COLOR_SLATE_LIGHT,
            "axes.labelcolor": COLOR_NAVY,
            "axes.titlecolor": COLOR_NAVY,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": COLOR_SLATE,
            "ytick.color": COLOR_SLATE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Roboto", "Arial", "DejaVu Sans"],
            "grid.color": COLOR_SLATE_LIGHT,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )


def _mean_body_mass_by_tier(df: pd.DataFrame) -> pd.Series:
    """Compute mean adult body mass (grams) for each hazard tier."""
    return (
        df.groupby(HAZARD_TIER_COLUMN)[BODY_MASS_COLUMN]
        .mean()
        .reindex([HIGHER_RISK_LABEL, LOWER_RISK_LABEL])
        .dropna()
    )


def plot_mean_body_mass_by_tier(df: pd.DataFrame, output_path: Path) -> None:
    """
    Figure A — vertical categorical chart of mean body mass by hazard tier.

    Parameters
    ----------
    df:
        Transformed reptile dataset.
    output_path:
        Destination path for the saved PNG figure.
    """
    tier_means = _mean_body_mass_by_tier(df)
    colors = [COLOR_NAVY_MID, COLOR_SLATE]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        tier_means.index,
        tier_means.values,
        color=colors[: len(tier_means)],
        width=0.55,
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_ylabel("Mean adult body mass (g)")
    ax.set_xlabel("Standardized hazard tier")
    ax.set_title("Mean Body Mass by Conservation Hazard Tier")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    for bar, value in zip(bars, tier_means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLOR_NAVY,
            fontweight="600",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_distribution(
    bootstrap_deltas: list[float],
    output_path: Path,
) -> None:
    """
    Figure B — histogram of bootstrap mean-difference outcomes with 95% CI bounds.

    Overlays vertical reference lines at the 2.5th and 97.5th percentiles.

    Parameters
    ----------
    bootstrap_deltas:
        Output of :func:`run_bootstrap_simulation`.
    output_path:
        Destination path for the saved PNG figure.
    """
    deltas = np.asarray(bootstrap_deltas, dtype=float)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        deltas,
        bins=45,
        color=COLOR_NAVY_MID,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axvline(ci_lower, color=COLOR_ACCENT, linewidth=2, linestyle="--", label=f"2.5th pct ({ci_lower:,.1f})")
    ax.axvline(ci_upper, color=COLOR_ACCENT, linewidth=2, linestyle="--", label=f"97.5th pct ({ci_upper:,.1f})")
    ax.axvline(deltas.mean(), color=COLOR_NAVY, linewidth=1.8, linestyle="-", label=f"Mean ({deltas.mean():,.1f})")

    ax.set_xlabel("Resampled mean body mass delta (g)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Distribution of Mean Body Mass Difference")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_visualizations(
    df: pd.DataFrame,
    bootstrap_deltas: list[float],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Execute the visualization layer and persist both diagnostic figures.

    Parameters
    ----------
    df:
        Transformed reptile dataset.
    bootstrap_deltas:
        Bootstrap resampling outcomes.
    output_dir:
        Directory for saved PNG files.

    Returns
    -------
    tuple[Path, Path]
        Paths to Figure A (tier means) and Figure B (bootstrap histogram).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_corporate_style()

    figure_a_path = output_dir / "figure_a_mean_body_mass_by_tier.png"
    figure_b_path = output_dir / "figure_b_bootstrap_distribution.png"

    plot_mean_body_mass_by_tier(df, figure_a_path)
    plot_bootstrap_distribution(bootstrap_deltas, figure_b_path)

    return figure_a_path, figure_b_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    """
    Run the end-to-end biodiversity analysis pipeline.

    Parameters
    ----------
    config:
        File paths, iteration count, and output settings.

    Returns
    -------
    dict
        Summary metrics including observation counts, CI bounds, and figure paths.
    """
    dataset = load_and_transform_data(config.amniote_path, config.iucn_path)
    bootstrap_deltas = run_bootstrap_simulation(
        dataset,
        iterations=config.bootstrap_iterations,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.output_dir / "cleaned_reptile_dataset.csv", index=False)

    figure_a, figure_b = generate_visualizations(
        dataset,
        bootstrap_deltas,
        config.output_dir,
    )

    deltas = np.asarray(bootstrap_deltas, dtype=float)
    summary = {
        "observations": len(dataset),
        "higher_risk_count": int((dataset[HAZARD_TIER_COLUMN] == HIGHER_RISK_LABEL).sum()),
        "lower_risk_count": int((dataset[HAZARD_TIER_COLUMN] == LOWER_RISK_LABEL).sum()),
        "bootstrap_iterations": config.bootstrap_iterations,
        "ci_lower_2_5": float(np.percentile(deltas, 2.5)),
        "ci_upper_97_5": float(np.percentile(deltas, 97.5)),
        "mean_delta": float(deltas.mean()),
        "figure_a": figure_a,
        "figure_b": figure_b,
    }
    return summary


def _parse_args() -> PipelineConfig:
    """Parse command-line arguments into a :class:`PipelineConfig`."""
    project_root = Path(__file__).resolve().parent
    default_data = project_root / "data"

    parser = argparse.ArgumentParser(
        description="Reptile biodiversity risk pipeline — ingest, simulate, visualize.",
    )
    parser.add_argument(
        "--amniote",
        type=Path,
        default=default_data / "Amniote_Database_Aug_2015.csv",
        help="Path to Amniote_Database_Aug_2015.csv",
    )
    parser.add_argument(
        "--iucn",
        type=Path,
        default=default_data / "IUCN_animals.csv",
        help="Path to IUCN_animals.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output",
        help="Directory for cleaned data and figures",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5_000,
        help="Bootstrap iteration count (default: 5000)",
    )
    args = parser.parse_args()

    return PipelineConfig(
        amniote_path=args.amniote,
        iucn_path=args.iucn,
        output_dir=args.output,
        bootstrap_iterations=args.iterations,
    )


def main() -> None:
    """CLI entry point for the biodiversity pipeline."""
    config = _parse_args()
    summary = run_pipeline(config)

    print("Biodiversity pipeline complete.")
    print(f"  Observations (post-merge): {summary['observations']:,}")
    print(f"  Higher Risk specimens:     {summary['higher_risk_count']:,}")
    print(f"  Lower Risk specimens:      {summary['lower_risk_count']:,}")
    print(f"  Bootstrap iterations:      {summary['bootstrap_iterations']:,}")
    print(
        f"  Mean delta (g):            {summary['mean_delta']:,.2f} "
        f"[{summary['ci_lower_2_5']:,.2f}, {summary['ci_upper_97_5']:,.2f}]"
    )
    print(f"  Figure A: {summary['figure_a']}")
    print(f"  Figure B: {summary['figure_b']}")


if __name__ == "__main__":
    main()

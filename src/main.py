# =============================================================================
# Carbon Tax Policy Support Analysis
# =============================================================================
# Replicates and extends the analysis from Hope et al. (2026) on carbon tax
# support across demographic and attitudinal groups in the United Kingdom.
#
# Originally implemented in Excel; fully migrated to Python for reproducibility,
# scalability, and integration with downstream statistical workflows.
#
# Author: Shivam Gujral
# Data: Hope et al. (2026) - Simplified dataset
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the main survey dataset and variable dictionary from Excel.

    Parameters
    ----------
    filepath : str
        Path to the Excel file containing the dataset.

    Returns
    -------
    dat : pd.DataFrame
        Main survey response dataset.
    var_info : pd.DataFrame
        Data dictionary with variable descriptions and coding.
    """
    var_info = pd.read_excel(filepath, sheet_name="Data dictionary")
    dat = pd.read_excel(filepath, sheet_name="Data")
    return dat, var_info


# =============================================================================
# 2. FEATURE ENGINEERING: DUMMY VARIABLES
# =============================================================================

def create_dummy_variables(dat: pd.DataFrame) -> pd.DataFrame:
    """
    Construct binary dummy variables from raw survey responses.

    Dummies created:
    - carbon_tax_support_dummy : 1 if respondent supports carbon tax (codes 1-2)
    - age_dummy                : 1 if respondent is older than 40
    - car_dummy                : 1 if respondent commutes by car
    - rural                    : 1 if respondent lives in a rural neighbourhood
    - unequal_treatment_dummy  : 1 if perceived unequal treatment score >= 8

    Missing values are preserved (not coded as 0).

    Parameters
    ----------
    dat : pd.DataFrame
        Raw survey dataset.

    Returns
    -------
    dat : pd.DataFrame
        Dataset with dummy variables appended.
    """
    dat = dat.copy()

    # Carbon tax support dummy
    dat["carbon_tax_support_dummy"] = np.nan
    dat.loc[dat["carbon_tax_support"].isin([1, 2]), "carbon_tax_support_dummy"] = 1
    dat.loc[dat["carbon_tax_support"].isin([3, 4, 5]), "carbon_tax_support_dummy"] = 0

    # Age dummy (> 40)
    dat["age_dummy"] = np.nan
    dat.loc[dat["age"] > 40, "age_dummy"] = 1
    dat.loc[(dat["age"].notna()) & (dat["age"] <= 40), "age_dummy"] = 0

    # Car commuting dummy
    dat["car_dummy"] = np.nan
    dat.loc[dat["commute"] == "Car", "car_dummy"] = 1
    dat.loc[(dat["commute"].notna()) & (dat["commute"] != "Car"), "car_dummy"] = 0

    # Rural dummy
    dat["rural"] = np.nan
    dat.loc[dat["neighbourhood"] == "Rural", "rural"] = 1
    dat.loc[(dat["neighbourhood"].notna()) & (dat["neighbourhood"] != "Rural"), "rural"] = 0

    # Unequal treatment perception dummy (score >= 8)
    dat["unequal_treatment_dummy"] = np.nan
    dat.loc[dat["unequal_treatment"] >= 8, "unequal_treatment_dummy"] = 1
    dat.loc[
        (dat["unequal_treatment"].notna()) & (dat["unequal_treatment"] < 8),
        "unequal_treatment_dummy"
    ] = 0

    return dat


# =============================================================================
# 3. DESCRIPTIVE ANALYSIS
# =============================================================================

def treatment_group_counts(dat: pd.DataFrame) -> pd.DataFrame:
    """Count respondents by treatment and control group."""
    return (
        dat.groupby("treatment")
        .size()
        .reset_index(name="n")
    )


def support_by_group(dat: pd.DataFrame, group_var: str, outcome: str = "carbon_tax_support_dummy") -> pd.DataFrame:
    """
    Compute mean outcome by subgroup within the control sample.

    Parameters
    ----------
    dat : pd.DataFrame
        Full dataset (will be filtered to control group internally).
    group_var : str
        Column name to group by (e.g. 'age_dummy', 'car_dummy').
    outcome : str
        Outcome variable to average (default: carbon_tax_support_dummy).

    Returns
    -------
    pd.DataFrame
        Mean outcome by group.
    """
    control = dat[dat["treatment"] == 0]
    return (
        control
        .dropna(subset=[group_var, outcome])
        .groupby(group_var)[outcome]
        .mean()
        .reset_index()
    )


def carbon_tax_frequency_table(dat: pd.DataFrame) -> pd.DataFrame:
    """
    Build a frequency table for carbon tax support in the control group.

    Returns
    -------
    pd.DataFrame
        Frequency table with labels, counts, and percentages.
    """
    labels = {
        1: "Strongly support",
        2: "Support",
        3: "Neither support nor oppose",
        4: "Oppose",
        5: "Strongly oppose"
    }
    control = dat[dat["treatment"] == 0]
    freq = (
        control[control["carbon_tax_support"].notna()]
        ["carbon_tax_support"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    freq.columns = ["carbon_tax_support", "n"]
    freq["Support label"] = freq["carbon_tax_support"].map(labels)
    freq["Percentage"] = freq["n"] / freq["n"].sum() * 100
    return freq[["Support label", "carbon_tax_support", "n", "Percentage"]]


# =============================================================================
# 4. STATISTICAL TESTS
# =============================================================================

def run_ttests(dat: pd.DataFrame) -> pd.DataFrame:
    """
    Run two-sample t-tests comparing treatment and control groups
    among rural respondents.

    Outcomes tested:
    - Unequal treatment perception
    - Carbon tax unfairness perception
    - Carbon tax support (1-5 scale)

    Returns
    -------
    pd.DataFrame
        Table of means, differences, and p-values.
    """
    rural = dat[dat["rural"] == 1]

    outcomes = {
        "Unequal treatment perception": "unequal_treatment",
        "Carbon tax unfairness perception": "carbon_tax_unfairness",
        "Carbon tax support (1-5 scale)": "carbon_tax_support"
    }

    results = []
    for label, var in outcomes.items():
        ctrl = rural.loc[rural["treatment"] == 0, var].dropna()
        trt = rural.loc[rural["treatment"] == 1, var].dropna()
        ttest = stats.ttest_ind(ctrl, trt, equal_var=True)
        results.append({
            "Outcome": label,
            "Control mean": ctrl.mean(),
            "Treatment mean": trt.mean(),
            "Difference (T - C)": trt.mean() - ctrl.mean(),
            "p-value": ttest.pvalue
        })

    return pd.DataFrame(results).set_index("Outcome")


def compute_confidence_intervals(dat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 95% confidence intervals for each outcome by treatment group
    among rural respondents.

    Returns
    -------
    pd.DataFrame
        Table of means and confidence interval bounds.
    """
    rural = dat[dat["rural"] == 1]

    outcomes = {
        "Unequal treatment perception": "unequal_treatment",
        "Carbon tax unfairness perception": "carbon_tax_unfairness",
        "Carbon tax support (1-5 scale)": "carbon_tax_support"
    }

    results = []
    for label, var in outcomes.items():
        for group_label, group_code in [("Control", 0), ("Treatment", 1)]:
            series = rural.loc[rural["treatment"] == group_code, var].dropna()
            mean = series.mean()
            se = series.std(ddof=1) / np.sqrt(len(series))
            results.append({
                "Variable": label,
                "Group": group_label,
                "Mean": mean,
                "CI lower (95%)": mean - 1.96 * se,
                "CI upper (95%)": mean + 1.96 * se
            })

    return pd.DataFrame(results)


# =============================================================================
# 5. VISUALISATION
# =============================================================================

def plot_carbon_tax_support(freq_table: pd.DataFrame) -> None:
    """Bar chart of carbon tax support distribution in the control group."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(freq_table["Support label"], freq_table["Percentage"])
    plt.title("Distribution of Support for Carbon Taxation in the United Kingdom")
    plt.xlabel("Carbon Tax Support")
    plt.ylabel("Percentage of Respondents")
    plt.xticks(fontsize=8)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center", va="bottom", fontsize=8
        )
    plt.tight_layout()
    plt.show()


def plot_treatment_comparison(dat: pd.DataFrame, var: str, title: str) -> None:
    """Bar chart comparing treatment and control group means for a given variable."""
    rural = dat[dat["rural"] == 1]
    labels = {0: "Control", 1: "Treatment"}
    summary = (
        rural.dropna(subset=[var])
        .groupby("treatment")[var]
        .mean()
        .reset_index()
    )
    summary["treatment"] = summary["treatment"].map(labels)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(summary["treatment"], summary[var])
    plt.title(title)
    plt.xlabel("Group")
    plt.ylabel(f"Mean: {var}")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center", va="bottom", fontsize=8
        )
    plt.tight_layout()
    plt.show()


def plot_confidence_intervals(ci_table: pd.DataFrame) -> None:
    """Horizontal error bar charts of 95% CIs by treatment group."""
    for variable in ci_table["Variable"].unique():
        plot_data = ci_table[ci_table["Variable"] == variable]
        plt.figure(figsize=(6, 4))
        plt.barh(
            plot_data["Group"],
            plot_data["Mean"],
            xerr=[
                plot_data["Mean"] - plot_data["CI lower (95%)"],
                plot_data["CI upper (95%)"] - plot_data["Mean"]
            ],
            capsize=6
        )
        plt.title(variable)
        plt.xlabel("Mean value")
        plt.tight_layout()
        plt.show()


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":

    # Load data
    FILEPATH = "data/Hope et al. (2026) - Simplified dataset.xlsx"
    dat, var_info = load_data(FILEPATH)

    # Feature engineering
    dat = create_dummy_variables(dat)

    # Descriptive analysis
    print("=== Treatment Group Counts ===")
    print(treatment_group_counts(dat))

    freq_table = carbon_tax_frequency_table(dat)
    print("\n=== Carbon Tax Support Frequency Table (Control Group) ===")
    print(freq_table)

    # Visualisations
    plot_carbon_tax_support(freq_table)

    for var, title in [
        ("unequal_treatment", "Average Unequal Treatment Perceptions"),
        ("carbon_tax_unfairness", "Average Carbon Tax Unfairness Perception"),
        ("carbon_tax_support_dummy", "Average Carbon Tax Support")
    ]:
        plot_treatment_comparison(dat, var, title)

    # Statistical tests
    print("\n=== Two-Sample T-Tests: Rural Respondents ===")
    print(run_ttests(dat))

    # Confidence intervals
    ci_table = compute_confidence_intervals(dat)
    print("\n=== 95% Confidence Intervals ===")
    print(ci_table)
    plot_confidence_intervals(ci_table)

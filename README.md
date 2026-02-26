# Carbon Tax Policy Model (Python)

## Overview
This project implements a carbon tax policy modelling framework in Python, converted from an Excel-based economic model.

## Key Contributions

- Fully migrated an Excel-based carbon tax support model to a modular Python pipeline.
- Implemented reproducible data loading, feature engineering, and statistical testing workflow.
- Automated subgroup analysis and treatment effect estimation using vectorised operations.
- Produced publication-style summary tables and visualisations for policy interpretation.

## Objectives
- Replicate Excel logic in Python
- Improve scalability and reproducibility
- Enable scenario analysis

## Methods
- Pandas / NumPy
- Vectorised calculations
- Policy simulation scenarios

## Output
Quantitative estimates of tax impact under alternative carbon pricing assumptions.

## How to Run

1. Clone the repository.
2. Install dependencies:
   pip install -r requirements.txt
3. Place the Excel dataset inside the `data/` folder.
4. Run:
   python src/main.py

## Methods

- Construction of binary dummy variables from categorical survey responses.
- Descriptive subgroup analysis within control samples.
- Two-sample t-tests (equal variance assumption) comparing treatment vs control.
- Manual computation of 95% confidence intervals.
- Visualisation of distributional and treatment effects using bar charts.

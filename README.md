# Skeleton-Posterior-Causal-Discovery
CSE516 Probabilistic Graphical Models group project on skeleton posterior-guided causal discovery under latent confounding, with implementation, synthetic data generation, baseline comparison, and experimental evaluation.
# SPOT-The-Cause: Skeleton Posterior-Guided Causal Discovery Under Latent Confounding

**CSE516 Probabilistic Graphical Models | Ahmedabad University**

### Team Members

* Anant Narayan
* Milan Shah
* Ankit Kumar

## Project Overview

This project implements and evaluates **SPOT (Skeleton Posterior-guided OpTimization)** for causal discovery in the presence of latent confounders. The approach uses a probabilistic **skeleton posterior** to guide differentiable causal graph optimization.

The project evaluates SPOT on synthetic datasets with known ground-truth causal graphs and the real-world **ERA5 climate dataset**, comparing it with methods such as **ABIC, FCI, and RFCI**.

## Dataset

### Synthetic Data

Synthetic causal datasets will be generated with controllable:

* Number of variables
* Sample size
* Graph density
* Latent confounding
* Edge strengths

Ground-truth graphs will be retained for quantitative evaluation.

### ERA5 Climate Dataset

The project will use the **ERA5 dataset from the EU Copernicus Climate Change Service** to evaluate causal discovery in a real-world climate system containing potentially unobserved variables.

## Key Objectives

* Develop a skeleton-posterior estimator using conditional-independence features and XGBoost classifiers.
* Implement and extend posterior-guided optimization for differentiable MAG learning.
* Evaluate the approach on synthetic datasets with known ground truth and the ERA5 dataset.
* Compare SPOT with **ABIC, FCI, and RFCI** under varying graph sizes, sample sizes, and confounding levels.
* Evaluate accuracy, scalability, and computational efficiency.

## Evaluation Metrics

* Skeleton F1
* Arrowhead F1
* Tail F1
* Structural Hamming Distance (SHD)
* True Positive Rate (TPR)
* False Discovery Rate (FDR)
* Execution Time

## Original SPOT Implementation

The original research artifact is available in Microsoft's `reliableAI` repository:

https://github.com/microsoft/reliableAI/tree/main/causal-kit/Spot

Data-generation utilities are available in the original artifact under:

`utils/datagen.py`

## Base Paper

**Scalable Differentiable Causal Discovery in the Presence of Latent Confounders with Skeleton Posterior**
Pingchuan Ma, Rui Ding, Qiang Fu, Jiaru Zhang, Shuai Wang, Shi Han, and Dongmei Zhang.
**KDD 2024**

**Paper:**
https://arxiv.org/abs/2406.10537

**PDF:**
https://arxiv.org/pdf/2406.10537

**DOI:**
https://doi.org/10.1145/3637528.3672031

## Status

**In Progress**

This repository contains the implementation, experiments, datasets/data-generation scripts, and documentation developed for the **CSE516 – Probabilistic Graphical Models** project.

# OpenFE RBFE Dashboard
An intuitive, web-based dashboard for performing Relative Binding Free Energy (RBFE) and alchemical Free Energy Perturbation (FEP) calculations using the [OpenFE framework]("https://github.com/IIIS-Li-Group/OpenFE").

⚠️ **Work in Progress: This dashboard is currently under heavy development.**

## 📖 Background & Concepts
If you are new to the world of alchemical transformations, start with my introductory guide: 
👉 Read the Blog Post: [Introduction to Alchemical Free Energy Calculation: FEP and TI for RBFE](https://yonglanliu.github.io/2025/12/19/RBFE.html). 

This guide breaks down the core concepts without overwhelming you with complex mathematical formulas.

---

## 🚀 Key Features
This package streamlines the standard OpenFE workflow into five user-friendly steps:
1. Molecular Visualization: Interactively view your protein target and ligand structures within the dashboard.
2. Network Planning: Automatically generate and inspect the Ligand Network (transformation edges).
3. System Setup: Configure your alchemical system, including solvation, force fields, and box parameters.
4. Job Execution: Simplified controls to prepare and run simulation scripts.
5. Results Analysis: Parse raw output data, visualize convergence, and extract final $\Delta \Delta G$ values.

---

## 🛠️ Installation & Usage
**Prerequisites**
Ensure you have OpenFE and the dashboard dependencies installed:
```bash
# Example installation (update with your actual package name)
pip install openfe-rbfe-dashboard
```

**Running the Dashboard**
```bash
python app.py
```

# Publication-Ready Visualizations Guide

## Overview

This project now includes two complementary visualization resources for generating thesis-quality figures:

1. **`src/visualizations.py`** - Reusable module with all visualization functions
2. **`notebooks/04-visualizations.ipynb`** - Interactive demonstration notebook

Both provide the same core functionality with examples and customization options.

---

## Available Visualization Functions

### 1. Lipinski Distribution (EDA)

**Function**: `plot_lipinski_distributions(dataframe, output_path, title)`

**Purpose**: Generate a 2×2 grid of histograms for Lipinski's Rule of Five descriptors.

**Inputs**:
- `dataframe` (pd.DataFrame): Must contain columns: `['Molecular Weight', 'LogP', 'NumHDonors', 'NumHAcceptors']`
- `output_path` (Path, optional): Where to save the figure (default: `results/figures/lipinski_distributions.png`)
- `title` (str, optional): Figure title

**Outputs**:
- PNG image with 4 subplots showing distributions
- Red dashed lines indicate Lipinski limits
- Frequency histograms with clear labels

**Example**:
```python
from visualizations import compute_lipinski_descriptors, plot_lipinski_distributions
import pandas as pd

# Option A: Compute from SMILES strings
df = compute_lipinski_descriptors(smiles_list)
plot_lipinski_distributions(df)

# Option B: Use existing dataframe with descriptor columns
df = pd.read_csv('data/raw/bioactivity/egfr_full_standardized.csv')
plot_lipinski_distributions(df)
```

**Figure Specifications**:
- Size: 12×10 inches
- Resolution: 300 DPI
- Subplots: MW (g/mol), LogP, H-Donors, H-Acceptors
- Reference lines: Lipinski limits for each descriptor

---

### 2. GNN Parity Plot

**Function**: `plot_gnn_parity(y_true, y_pred, output_path, title)`

**Purpose**: Compare predicted vs. true pIC50 values with regression metrics.

**Inputs**:
- `y_true` (np.ndarray or list): Ground truth pIC50 values
- `y_pred` (np.ndarray or list): Model predictions
- `output_path` (Path, optional): Save location (default: `results/figures/gnn_parity_plot.png`)
- `title` (str, optional): Figure title

**Outputs**:
- Scatter plot with perfect prediction diagonal line
- Annotated metrics: R², RMSE, MAE, sample count
- Returns: Tuple of (r2_score, rmse, mae)

**Example**:
```python
from visualizations import plot_gnn_parity
import numpy as np

# Generate synthetic predictions for demonstration
y_true = np.random.normal(6.5, 1.5, 100)
y_pred = y_true + np.random.normal(0, 0.4, 100)

r2, rmse, mae = plot_gnn_parity(y_true, y_pred)
print(f"Model Performance - R²: {r2:.4f}, RMSE: {rmse:.4f}")
```

**Figure Specifications**:
- Size: 10×9 inches (square aspect ratio)
- Reference line: Perfect predictions (y=x) in red
- Metrics box: Top-left corner with statistics
- Grid: Light transparency for readability

---

### 3. VAE Molecule Grid

**Function**: `plot_vae_molecule_grid(smiles_list, predicted_scores, output_path, mols_per_row, title)`

**Purpose**: Display generated molecules as a grid with predicted affinity scores.

**Inputs**:
- `smiles_list` (List[str]): SMILES strings for molecules to display
- `predicted_scores` (List[float], optional): pIC50 predictions for each molecule
- `output_path` (Path, optional): Save location (default: `results/figures/vae_generated_molecules.png`)
- `mols_per_row` (int, optional): Molecules per row in grid (default: 5)
- `title` (str, optional): Figure title

**Outputs**:
- PNG grid image with molecule structures
- Labels: "pIC50: X.XX" for each molecule
- Automatically handles invalid SMILES

**Example**:
```python
from visualizations import plot_vae_molecule_grid

# Generated molecules from your VAE model
smiles_list = [
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Example: Ibuprofen
    'CC(=O)Oc1ccccc1C(=O)O',       # Example: Aspirin
]

# Predicted affinity scores
scores = [7.2, 6.8]

plot_vae_molecule_grid(smiles_list, predicted_scores=scores, mols_per_row=5)
```

**Figure Specifications**:
- Image size per molecule: 300×300 pixels
- Grid layout: Automatic based on total molecules
- Labels: Molecule index or pIC50 score
- Format: PNG at 300 DPI

---

### 4. Training Curves

**Function**: `plot_training_curves(epochs, train_loss, val_loss, output_path, title, ylabel)`

**Purpose**: Visualize training and validation loss over epochs.

**Inputs**:
- `epochs` (List[int]): Epoch numbers
- `train_loss` (List[float]): Training loss values
- `val_loss` (List[float]): Validation loss values
- `output_path` (Path, optional): Save location (default: `results/figures/training_curves.png`)
- `title` (str, optional): Figure title
- `ylabel` (str, optional): Y-axis label

**Outputs**:
- Line plot with training and validation curves
- Legend distinguishing train/val phases

**Example**:
```python
from visualizations import plot_training_curves

# From your training loop
epochs = list(range(1, 21))
train_losses = [10.0, 9.5, 9.2, ...]  # 20 values
val_losses = [9.8, 9.4, 9.1, ...]     # 20 values

plot_training_curves(epochs, train_losses, val_losses)
```

---

## Installation & Dependencies

All functions require standard ML/chemistry libraries:

```bash
pip install matplotlib seaborn rdkit torch pandas numpy scikit-learn
```

Tested with:
- matplotlib ≥ 3.5
- seaborn ≥ 0.12
- rdkit ≥ 2024.09
- scikit-learn ≥ 1.3

---

## Usage Workflow

### Quick Start (Notebook)

```python
# 1. Import and setup
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from visualizations import (
    compute_lipinski_descriptors,
    plot_lipinski_distributions,
    plot_gnn_parity,
    plot_vae_molecule_grid,
    plot_training_curves
)

# 2. Load your data
raw_df = pd.read_csv('data/raw/bioactivity/egfr_full_standardized.csv')

# 3. Generate figures (all saved to results/figures/)
plot_lipinski_distributions(raw_df)
plot_gnn_parity(y_test, y_pred_gnn)
plot_vae_molecule_grid(generated_smiles, gnn_predictions)
```

### Integration with Master Workflow

Add to `00-master-workflow.ipynb`:

```python
# Section: Generate Publication Figures
%run notebooks/04-visualizations.ipynb
```

Or run individually:

```python
# After model training
exec(open('notebooks/04-visualizations.ipynb').read())
```

---

## Customization

### Modify Plot Appearance

**Change Colors**:
```python
# In the function, update hex codes:
color='#3498db'  # Blue
color='#e74c3c'  # Red
color='#2ecc71'  # Green
```

Recommended colorblind-friendly palette:
- `#1f77b4` (blue)
- `#ff7f0e` (orange)
- `#2ca02c` (green)
- `#d62728` (red)

**Adjust Figure Size**:
```python
fig, ax = plt.subplots(figsize=(12, 8))  # width, height in inches
```

**Change Resolution**:
```python
plt.savefig(output_path, dpi=600)  # For ultra-high quality
```

**Remove Grid**:
```python
ax.grid(False)  # or ax.grid(True, alpha=0)
```

---

## Output Directory Structure

All figures are saved to `results/figures/`:

```
results/figures/
├── lipinski_distributions.png          # EDA (2×2 grid)
├── gnn_parity_plot.png                 # Regression metrics
├── vae_generated_molecules.png         # Molecule grid
├── training_curves.png                 # Loss curves
└── [other custom figures]
```

Each file is **300 DPI** PNG format, suitable for:
- LaTeX documents (`\includegraphics{...}`)
- Microsoft Word/PowerPoint
- PDF reports
- Peer-reviewed journals

---

## Troubleshooting

### "Invalid SMILES" warnings
- Some generated molecules may be chemically unfeasible
- The function skips invalid SMILES and logs warnings
- Check molecules with RDKit's `Chem.MolFromSmiles(smiles)`

### Missing descriptor columns
- Ensure dataframe has: `['Molecular Weight', 'LogP', 'NumHDonors', 'NumHAcceptors']`
- Use `compute_lipinski_descriptors()` to calculate from SMILES

### Model not loading
- Check `models/best_gnn_regressor.pth` exists
- Verify checkpoint compatibility with model architecture
- Use synthetic predictions as fallback

### Figure not saving
- Ensure `results/figures/` directory exists (auto-created by functions)
- Check write permissions in output directory
- Verify disk space available

---

## Examples for Thesis

### Results Section
```
Figure 3: GNN regression performance on EGFR bioactivity
[insert gnn_parity_plot.png]
The model achieved R² = 0.82 with RMSE = 0.45 pIC50 units.
```

### Methods Section
```
Figure 1: Lipinski profile of chemical dataset
[insert lipinski_distributions.png]
The dataset comprises molecules following drug-likeness criteria.
```

### Appendix/Supplementary
```
Figure A1: Generated molecules with predicted affinity
[insert vae_generated_molecules.png]
The generative model produced molecules spanning
the affinity range from 5.2 to 8.1 pIC50.
```

---

## Version History

- **v1.0** (May 2026): Initial release
  - Lipinski distributions, GNN parity, VAE grid, training curves
  - Tested with Python 3.12, PyTorch 2.1, RDKit 2026.03

---

## Citation

If you use these visualizations in your research:

```bibtex
@software{clean_data_diss2026,
  title={Publication-Ready ML Visualizations},
  author={Your Name},
  year={2026},
  url={https://github.com/ameliasburton/clean_data_diss}
}
```

---

## License

Use freely for your Master's thesis and research publications.

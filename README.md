# EGFR GNN Project

Project scaffold for GNN-based regression on EGFR binding affinity (pChEMBL).

Structure
```
egfr-gnn-project/

├── data/
│   ├── raw/                 # original, immutable datasets
│   ├── processed/           # processed PyG graph data (graph_data.pt)
│   └── external/            # decoys, structures, benchmarks
│
├── models/                  # saved model weights
│
├── notebooks/               # Jupyter notebooks for exploration
│
├── src/                     # Python source code
│   ├── __init__.py
│   ├── data_prep.py         # SMILES -> PyG graphs
│   ├── model_arch.py        # GNN architecture
│   ├── train.py             # training loop
│   └── evaluate.py          # inference/evaluation
│
├── logs/                    # TensorBoard logs
│
├── results/                 # outputs from pipeline
│   ├── figures/             # loss curves, RDKit drawings
│   └── generated_mols/      # generated SMILES and properties
│
├── requirements.txt
└── README.md
```

Quickstart

1. Create a conda env and install RDKit, PyTorch, and PyG (conda recommended for RDKit).

2. Place your raw CSV (`raw_chembl_egfr.csv`) under `data/raw/`.

3. Run preprocessing:

```bash
python src/data_prep.py --input data/raw/raw_chembl_egfr.csv --output data/processed/graph_data.pt
```

4. Train (example):

```bash
python src/train.py --train-file data/processed/graph_data.pt --epochs 50 --batch 32 --out models/gnn_regressor.pth
```

5. Evaluate:

```bash
python src/evaluate.py --model-state models/gnn_regressor.pth --data-file data/processed/graph_data.pt
```

Notes
- See `notebooks/` for exploration templates.
- Tune model, features, and data handling for production use.

# CHIME

## Overview

This is the data and source code for our paper: 

> **Beyond Global Pooling: Fine-Grained Drug-Drug Interaction Prediction via Knowledge-Guided Hypergraph Motif Matching**

CHIME combines molecular motif encoding, biomedical knowledge-graph (KG) context, and cross-drug interaction reasoning for multi-class drug-drug interaction (DDI) prediction.

## Included files

```
├── main.py              # Experiment entry point and hyperparameters
├── train_test.py        # Training and evaluation loops
├── model/
│  ├── CHIME.py          # Top-level model wrapper
│  └── decoder.py        # MLP DDI classifier
├── utils/
│  ├── EarlyStopping.py  # Macro-F1-based early stopping helper
│  └── metrics.py        # ACC, Macro-F1, Macro-Precision, Macro-Recall
└── data/
  ├── DRKG+Deng/
  │  ├── ddi.tsv
  │  └── smiles.tsv
  └── DRKG+Ryu/
     ├── ddi.tsv
     └── smiles.tsv
```

## Data snapshot

| Dataset | Drugs | DDI pairs | Classes | Drug ID range | Label range |
| ------- | ----- | --------- | ------- | ------------- | ----------- |
| Deng    | 570   | 37,264    | 65      | 0-569         | 0-64        |
| Ryu     | 1700  | 191,570   | 86      | 0-1699        | 0-85        |

Each `ddi.tsv` row is tab-separated and has the form:

```text
left_drug_id    right_drug_id    ddi_class_id
```

Each `smiles.tsv` row is tab-separated and has the form:

```text
drug_id    canonical_or_source_smiles
```

## Data sources

| Resource | Role                       | Source                                                       |
| -------- | -------------------------- | ------------------------------------------------------------ |
| Deng     | 65-class DDI event dataset | [Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/15/4316/5837109) |
| Ryu      | 86-class DDI event dataset | [PNAS 2018](https://www.pnas.org/doi/10.1073/pnas.1803294115) |
| DRKG     | Biomedical knowledge graph | [Official repository](https://github.com/gnn4dr/DRKG)        |

The intended drug-level split protocol follows the general setup used by [HetDDI](https://github.com/LinsLab/HetDDI).

## Package Dependencies

Please ensure your local compilation environment includes the following packages:

```
python        == 3.10
torch         == 2.2.1+cu121
torch_scatter == 2.1.2+pt22cu121
dgl           == 2.2.1
numpy         == 1.26.4
networkx      == 2.6
rdkit         == 2024.9.6
scikit-learn  == 1.6.1
```

Reported experiments were run on an Intel Xeon Gold 6330 CPU and an NVIDIA RTX 5880 Ada GPU.

## Intended experiment interface

`main.py` defines the following main options:

| Argument         | Default | Description                                   |
| ---------------- | ------: | --------------------------------------------- |
| `--ddi_name`     |  `Deng` | Dataset: `Deng` or `Ryu`                      |
| `--condition`    |    `S1` | Split setting: `S1`, `S2`, or `S3`            |
| `--fold_num`     |     `5` | Number of folds expected from the data loader |
| `--hidden_dim`   |   `256` | Hidden dimension                              |
| `--num_layer`    |     `3` | Number of encoder layers                      |
| `--batch_size`   |  `2048` | Mini-batch size                               |
| `--lr`           |  `5e-4` | AdamW learning rate                           |
| `--weight_decay` |  `1e-4` | Weight decay                                  |
| `--patience`     |    `40` | Early-stopping patience                       |

After the complete data loader, encoders, KG files, and split files are released, experiments will follow an interface such as:

```bash
python main.py --ddi_name Deng --condition S1 --ablation full
python main.py --ddi_name Ryu  --condition S1 --ablation full
```

## Citation and license

Citation metadata and a software license have not yet been included in this snapshot. Add both before public release.

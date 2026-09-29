# TSB-AD Dataset

This directory stores the example data and file lists used by VETime for evaluation on TSB-AD.

## Directory Layout

```text
Datasets/
├── TSB-AD-U/       # Univariate time series
├── TSB-AD-M/       # Multivariate time series
└── File_List/      # Evaluation and tuning file lists
```

## Download

- [TSB-AD-U](https://www.thedatum.org/datasets/TSB-AD-U.zip)
- [TSB-AD-M](https://www.thedatum.org/datasets/TSB-AD-M.zip)

After downloading and extracting the archives, place the two dataset folders under this directory. The repository also includes example time series in `TSB-AD-U/` and `TSB-AD-M/`.

## File Naming Convention

```text
[index]_[dataset_name]_id_[id]_[domain]_tr_[train_index]_1st_[first_anomaly_index].csv
```

Supported domains include `Web Service`, `Sensor`, `Environment`, `Traffic`, `Finance`, `Facility`, `Medical`, and `Synthetic`.

The `File_List/` directory contains the file lists used for evaluation and hyperparameter tuning.

## Data License

The datasets are provided for reproducibility. Please consult the [TSB-AD documentation](https://thedatumorg.github.io/TSB-AD/) and the original data sources for the license of each included dataset.

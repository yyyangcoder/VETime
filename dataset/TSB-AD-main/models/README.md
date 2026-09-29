# Optional Baseline Dependencies

The following dependencies are only required when evaluating the corresponding baseline models. Install them from the repository root or from the environment used for TSB-AD evaluation.

## Chronos

```bash
git clone https://github.com/autogluon/autogluon
cd autogluon
pip install -e 'timeseries/[TimeSeriesDataFrame,TimeSeriesPredictor]'
```

## MOMENT

```bash
pip install momentfm
```

The current MOMENT implementation requires Python 3.11.

## TimesFM

```bash
pip install 'timesfm[torch]'
```

## Lag-Llama

```bash
pip install 'gluonts[torch]<=0.14.4'
```

Download the Lag-Llama checkpoint from the [official repository](https://github.com/time-series-foundation-models/lag-llama), then configure its path in [`Lag_Llama.py`](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/models/Lag_Llama.py).

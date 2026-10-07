# Jet-engine-failure-prediction
predicts the Remaining Useful Life (RUL) of a turbofan engine using machine learning.

## Technology Used
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Random Forest Regression

## Dataset
NASA C-MAPSS FD001 dataset.
* 100 training engines
* 100 test engines
* 21 sensor readings
* 3 operating settings
* Engine cycle information
* RUL values for the test engines
The dataset files used in this project are inside the `data` folder.

## Model
A Random Forest Regressor to predict the remaining useful life.
The RUL is measured in operating cycles.

The model achieved:
```text
Validation MAE  : 12.45 cycles
Validation RMSE : 17.11 cycles
Test MAE        : 13.11 cycles
```
A lower MAE means the predictions are closer to the actual RUL values.

## Project Files
```text
jet_engine_failure_prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
└── data/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```
## How to Run
First install the required libraries:
```bash
pip install -r requirements.txt
```

Then train the model:
```bash
python train_model.py
```

After training is finished, start the dashboard:
```bash
python -m streamlit run app.py
```

## Dashboard
shows:
* Predicted remaining useful life
* Last observed engine cycle
* Prediction error
* Sensor graphs
* Predicted vs actual RUL
* Overall test MAE
* Predictions for all test engines

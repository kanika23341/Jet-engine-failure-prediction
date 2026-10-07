from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"
MODEL_DIR=ROOT/"models"
MODEL_DIR.mkdir(exist_ok=True)
columns=(["engine_id","cycle"]+[f"setting_{i}" for i in range(1,4)]+[f"sensor_{i}" for i in range(1,22)])
features=columns[2:]
RUL_CAP=125

train_path=DATA_DIR/"train_FD001.txt"
test_path=DATA_DIR/"test_FD001.txt"
rul_path=DATA_DIR/"RUL_FD001.txt"

for path in [train_path, test_path, rul_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing File: {path}\n")
train=pd.read_csv(train_path, sep=r"\s+", header=None, names=columns).dropna(axis=1, how="all")
test=pd.read_csv(test_path, sep=r"\s+", header=None, names=columns).dropna(axis=1, how="all")
true_rul=pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:,0]

print("Training rows:", len(train))
print("Training engines: ", train["engine_id"].nunique())
print("Test engines: ",test["engine_id"].nunique())

last_cycle=train.groupby("engine_id")["cycle"].max()
train["RUL"]=(train["engine_id"].map(last_cycle)- train["cycle"])
train["RUL"]=train["RUL"].clip(upper=RUL_CAP)

splitter=GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx,val_idx=next(splitter.split(train[features], train["RUL"], groups=train["engine_id"],))

x_train=train.iloc[train_idx][features]
y_train=train.iloc[train_idx]["RUL"]
x_val=train.iloc[val_idx][features]
y_val=train.iloc[val_idx]["RUL"]

#random forest training
model=RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1,)
print("Training model...\n")
model.fit(x_train, y_train)
val_prediction=np.clip(model.predict(x_val),0,None)

mae=mean_absolute_error(y_val, val_prediction)
rmse=np.sqrt(mean_squared_error(y_val, val_prediction))

print("Validation results")
print(f"MAE: {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")

model.fit(train[features], train["RUL"])
joblib.dump(
    {"model": model, "features": features, "rul_cap": RUL_CAP},
    MODEL_DIR/"rul_model.joblib",
)

test_last=(test.sort_values(["engine_id","cycle"]).groupby("engine_id").tail(1).sort_values("engine_id").reset_index(drop=True))
test_predictions=np.clip(model.predict(test_last[features]),0,None)

if(len(true_rul)!=len(test_last)):
    raise ValueError("Test engines and RUL labels do not match")
test_results=pd.DataFrame({
    "engine_id": test_last["engine_id"],
    "last_cycle": test_last["cycle"],
    "actual_RUL": true_rul.to_numpy(),
    "predicted_RUL": test_predictions,
})

test_mae=mean_absolute_error(test_results["actual_RUL"], test_results["predicted_RUL"],)
test_results.to_csv(ROOT /"test_predictions.csv", index=False)
print(f"\nOfficial test-set MAE: {test_mae:.2f} cycles")
print("Saved model to models/rul_model.joblib and predictions to test_predictions.csv")
print("\nTraining complete!")
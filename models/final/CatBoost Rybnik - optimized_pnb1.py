import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


from sklearn.metrics import mean_absolute_error, mean_squared_error



from przepompowanie_rybnik_dane import przygotuj_dane_przepompowanie

_,_,X,y = przygotuj_dane_przepompowanie()

train_end = int(len(X) * 0.7)

X_train = X.iloc[:train_end]
X_test = X.iloc[train_end:]

y_train = y.iloc[:train_end]
y_test = y.iloc[train_end:]

#print("TRAIN:", X_train.shape, X_train.index.min(), X_train.index.max())
#print("VAL:", X_val.shape, X_val.index.min(), X_val.index.max())
#print("TEST:", X_test.shape, X_test.index.min(), X_test.index.max())



best_params_median = {'learning_rate': 0.04203513948641982, 'depth': 6, 'l2_leaf_reg': 19.838135683506707, 'random_strength': 1.7287450778554834}
# CATBOOST
from catboost import CatBoostRegressor

final_cb_median = CatBoostRegressor(
    **best_params_median,
    loss_function="MAE",
    iterations=270,
    task_type="CPU",
    thread_count=-1,
    random_seed=42,
    verbose=50
)

final_cb_median.fit(X_train, y_train)

y_pred_test = final_cb_median.predict(X_test)

final_mae = np.mean(np.abs(y_test - y_pred_test))
print(f"\nOstateczne MAE na zbiorze TESTOWYM: {final_mae:.4f}")




best_params_q97 = {'learning_rate': 0.013079619314708114, 'depth': 6, 'l2_leaf_reg': 11.689680625188423, 'random_strength': 0.48731610238448314}

final_cb_q97 = CatBoostRegressor(
    **best_params_q97,
    loss_function="Quantile:alpha=0.97",
    iterations=240,
    task_type="CPU",
    thread_count=-1,
    random_seed=42,
    verbose=50
)
final_cb_q97.fit(X_train, y_train)

q97_preds = final_cb_q97.predict(X_test)

test_coverage = np.mean(y_test <= q97_preds)
print(f"\nFinalne Coverage na zbiorze testowym: {test_coverage * 100:.2f}% (cel: 97.00%)")




# ZAPIS MODELOW DO PLIKOW

final_cb_median.save_model("rybnik_przepompowanie_pnb1_median.cbm")
final_cb_q97.save_model("rybnik_przepompowanie_pnb1_q97.cbm")
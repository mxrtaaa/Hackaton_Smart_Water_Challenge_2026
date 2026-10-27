import pandas as pd
import numpy as np
from pathlib import Path
import sys
from lightgbm import LGBMRegressor
from przygotowanie_danych_zuzycia_energii import przygotuj_dane_zuzycia_energii

models_path = Path.cwd().parents[0]
sys.path.append(str(models_path / "Predykcja przepompowanych ścieków"))

from przygotowanie_danych_przepompowania import przygotuj_dane_przepompowania_z_opadami

X_energia, Y_energia = przygotuj_dane_zuzycia_energii()
X_przeplyw, Y_przeplyw = przygotuj_dane_przepompowania_z_opadami()

features = [
    # Kalendarz
    "hour", "dayofweek", "month",

    # Opady
    "opad_1h", "opad_6h", "opad_12h",
    "opad_24h", "opad_48h", "opad_168h",
    "h_od_deszczu",
]

X = X_energia[features].join(Y_przeplyw)
y = Y_energia

train_end = int(len(X) * 0.7)

X_train = X.iloc[:train_end]
X_test = X.iloc[train_end:]

y_train = y.iloc[:train_end]
y_test = y.iloc[train_end:]


best_params_l1 = {'learning_rate': 0.07880420151149103, 'num_leaves': 38, 'min_child_samples': 49, 'subsample': 0.9499165887148772, 'colsample_bytree': 0.9221551338108949, 'reg_alpha': 0.00908900502044327, 'reg_lambda': 0.12097202281514402}

# 2. Finalny trening na sztywnej liczbie drzew
final_lgbm_l1 = LGBMRegressor(
    objective="l1",
    **best_params_l1,
    n_estimators=124,   # dokładnie tyle, ile dało minimum błędu
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

final_lgbm_l1.fit(X_train, y_train)

y_pred_test = final_lgbm_l1.predict(X_test)
final_mae = np.mean(np.abs(y_test - y_pred_test))
print(f"\nOstateczne MAE na zbiorze TESTOWYM: {final_mae:.4f}")


best_params_q97 = {'learning_rate': 0.07247101488789359, 'num_leaves': 30, 'min_child_samples': 66, 'subsample': 0.9293703223284387, 'colsample_bytree': 0.9035414290503055, 'reg_alpha': 3.1168759749192407, 'reg_lambda': 1.9594681074471874}

final_lgbm_q97 = LGBMRegressor(
    objective="quantile",
    alpha=0.97,
    **best_params_q97,
    n_estimators=120,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
final_lgbm_q97.fit(X_train, y_train,)

q97_preds = final_lgbm_q97.predict(X_test)

test_coverage = np.mean(y_test <= q97_preds)
print(f"\nFinalne Coverage na zbiorze testowym: {test_coverage * 100:.2f}% (cel: 97.00%)")


final_lgbm_l1.booster_.save_model("chorzow_energia_median.txt")
final_lgbm_q97.booster_.save_model("chorzow_energia_q97.txt")


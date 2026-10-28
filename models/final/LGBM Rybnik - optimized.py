import pandas as pd
import numpy as np
from pathlib import Path
import sys
from lightgbm import LGBMRegressor

from energia_rybnik_dane import przygotuj_dane as przygotuj_dane_energia

X_energia, Y_energia,_,_ = przygotuj_dane_energia()
models_path = Path.cwd().parents[0]
sys.path.append(str(models_path / "Predykcja przepompowanych ścieków"))
from przepompowanie_rybnik_dane import przygotuj_dane_przepompowanie

X_przeplyw, Y_przeplyw,_,_ = przygotuj_dane_przepompowanie()

features = [
    # Kalendarz
    "hour", "dayofweek", "month",

    # Opady
    "opad_1h", "opad_6h", "opad_12h",
    "opad_24h", "opad_48h", "opad_168h",
    "h_od_deszczu",
]

df_all = X_energia[features].join([Y_przeplyw, Y_energia], how="inner")

# Odrzucenie ewentualnych wierszy z brakami po złączeniu
df_all = df_all.dropna()

# Rozdzielenie na X i y z gwarancją zgodności wiersz po wierszu
target_col = Y_energia.name  # np. 'zuzycie_energii'
X = df_all.drop(columns=[target_col])
y = df_all[target_col]

# Weryfikacja: teraz len(X) == len(y) bez wyjątku
print(f"Liczba próbek po dopasowaniu: {len(X)}")

train_end = int(len(X) * 0.7)

X_train = X.iloc[:train_end]
X_test = X.iloc[train_end:]

y_train = y.iloc[:train_end]
y_test = y.iloc[train_end:]

best_params_l1 = {'learning_rate': 0.04061780271024422, 'num_leaves': 38, 'min_child_samples': 24, 'subsample': 0.743921700216187, 'colsample_bytree': 0.7732833512896126, 'reg_alpha': 0.0036288325466427265, 'reg_lambda': 0.014766552392796261}

# 2. Finalny trening na sztywnej liczbie drzew
final_lgbm_l1 = LGBMRegressor(
    objective="l1",
    **best_params_l1,
    n_estimators=140,   # dokładnie tyle, ile dało minimum błędu
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

final_lgbm_l1.fit(X_train, y_train)

y_pred_test = final_lgbm_l1.predict(X_test)
final_mae = np.mean(np.abs(y_test - y_pred_test))
print(f"\nOstateczne MAE na zbiorze TESTOWYM: {final_mae:.4f}")


best_params_q97 = {'learning_rate': 0.08020337033817654, 'num_leaves': 22, 'min_child_samples': 48, 'subsample': 0.6434378819004541, 'colsample_bytree': 0.6590293440773054, 'reg_alpha': 3.7080310739472964, 'reg_lambda': 1.6219681845800913}

final_lgbm_q97 = LGBMRegressor(
    objective="quantile",
    alpha=0.993,
    **best_params_q97,
    n_estimators=170,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
final_lgbm_q97.fit(X_train, y_train)

q97_preds = final_lgbm_q97.predict(X_test)

test_coverage = np.mean(y_test <= q97_preds)
print(f"\nFinalne Coverage na zbiorze testowym: {test_coverage * 100:.2f}% (cel: 97.00%)")


final_lgbm_l1.booster_.save_model("rybnik_energia_pz1_median.txt")
final_lgbm_q97.booster_.save_model("rybnik_energia_pz1_q97.txt")
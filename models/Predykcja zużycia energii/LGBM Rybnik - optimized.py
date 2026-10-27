import pandas as pd
import numpy as np
from pathlib import Path
import sys
from lightgbm import LGBMRegressor
from przygotowanie_danych_zuzycia_energii import przygotuj_dane_zuzycia_energii

from energia_ruda_dane import przygotuj_dane as przygotuj_dane_energia

X_energia, Y_energia = przygotuj_dane_energia()
models_path = Path.cwd().parents[0]
sys.path.append(str(models_path / "Predykcja przepompowanych ścieków"))
from przepompowanie_rybnik_dane import przygotuj_dane_przepompowanie_pz1

X_przeplyw, Y_przeplyw = przygotuj_dane_przepompowanie_pz1()

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


best_params_l1 = {'learning_rate': 0.020199610304989114, 'num_leaves': 36, 'min_child_samples': 24, 'subsample': 0.770384612026848, 'colsample_bytree': 0.9221233904338164, 'reg_alpha': 0.0011863471828765819, 'reg_lambda': 0.00303611279227012}

# 2. Finalny trening na sztywnej liczbie drzew
final_lgbm_l1 = LGBMRegressor(
    objective="l1",
    **best_params_l1,
    n_estimators=210,   # dokładnie tyle, ile dało minimum błędu
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

final_lgbm_l1.fit(X_train, y_train)

y_pred_test = final_lgbm_l1.predict(X_test)
final_mae = np.mean(np.abs(y_test - y_pred_test))
print(f"\nOstateczne MAE na zbiorze TESTOWYM: {final_mae:.4f}")


best_params_q97 = {'learning_rate': 0.05151661242024018, 'num_leaves': 32, 'min_child_samples': 8, 'subsample': 0.713562588867843, 'colsample_bytree': 0.6995608051638371, 'reg_alpha': 0.0013479702461600001, 'reg_lambda': 0.015956274493326334}

final_lgbm_q97 = LGBMRegressor(
    objective="quantile",
    alpha=0.993,
    **best_params_q97,
    n_estimators=360,
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
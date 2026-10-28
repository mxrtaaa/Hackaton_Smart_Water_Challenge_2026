import pandas as pd
import numpy as np
from pathlib import Path
import sys
from lightgbm import LGBMRegressor


from energia_rybnik_dane import przygotuj_dane_energia

X_energia, Y_energia,_,_ = przygotuj_dane_energia()

features = [
    # Kalendarz
    "hour", "dayofweek", "month",

    # Opady
    "opad_1h", "opad_6h", "opad_12h",
    "opad_24h", "opad_48h", "opad_168h",
    "h_od_deszczu", "przepompowano_m3"
]

df_all = X_energia[features].join([Y_energia], how="inner")

# Odrzucenie ewentualnych wierszy z brakami po złączeniu
df_all = df_all.dropna()

# Rozdzielenie na X i y z gwarancją zgodności wiersz po wierszu
target_col = Y_energia.name
X = df_all.drop(columns=[target_col])
y = df_all[target_col]
# Weryfikacja: teraz len(X) == len(y) bez wyjątku
print(f"Liczba próbek po dopasowaniu: {len(X)}")

train_end = int(len(X) * 0.7)

X_train = X.iloc[:train_end]
X_test = X.iloc[train_end:]

y_train = y.iloc[:train_end]
y_test = y.iloc[train_end:]


best_params_l1 = {'learning_rate': 0.03366420547023812, 'num_leaves': 39, 'min_child_samples': 22, 'subsample': 0.6982403767535683, 'colsample_bytree': 0.8935636889419895, 'reg_alpha': 0.29180496418409846, 'reg_lambda': 0.018389999841543495}

# 2. Finalny trening na sztywnej liczbie drzew
final_lgbm_l1 = LGBMRegressor(
    objective="l1",
    **best_params_l1,
    n_estimators=120,   # dokładnie tyle, ile dało minimum błędu
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

final_lgbm_l1.fit(X_train, y_train)

y_pred_test = final_lgbm_l1.predict(X_test)
final_mae = np.mean(np.abs(y_test - y_pred_test))
print(f"\nOstateczne MAE na zbiorze TESTOWYM: {final_mae:.4f}")


best_params_q97 = {'learning_rate': 0.05196030341075277, 'num_leaves': 16, 'min_child_samples': 27, 'subsample': 0.7187107056091574, 'colsample_bytree': 0.8989434630587481, 'reg_alpha': 0.0015836655559547709, 'reg_lambda': 0.003782096440904748}

final_lgbm_q97 = LGBMRegressor(
    objective="quantile",
    alpha=0.97,
    **best_params_q97,
    n_estimators=190,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
final_lgbm_q97.fit(X_train, y_train)

q97_preds = final_lgbm_q97.predict(X_test)

# DO ANALIZY BIZNESOWEJ ############

from sciezka_projektu import get_project_root

results_test = pd.DataFrame({
    "rzeczywiste": y_test,
    "prognoza": y_pred_test,
    "q97": q97_preds,
}, index=y_test.index)

results_test["alarm"] = (
    results_test["rzeczywiste"] > results_test["q97"]
)

output = get_project_root() / "reports" / "models"
output.mkdir(parents=True, exist_ok=True)

results_test.to_parquet(output / "rybnik_pz1_energia.parquet")

print("Zapisano:", output / "rybnik_pz1_energia.parquet")

#####################################33

test_coverage = np.mean(y_test <= q97_preds)
print(f"\nFinalne Coverage na zbiorze testowym: {test_coverage * 100:.2f}% (cel: 97.00%)")


final_lgbm_l1.booster_.save_model("rybnik_energia_pz1_median.txt")
final_lgbm_q97.booster_.save_model("rybnik_energia_pz1_q97.txt")
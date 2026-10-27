import pandas as pd
import numpy as np
from pathlib import Path
import sys
from lightgbm import LGBMRegressor
from przygotowanie_danych_zuzycia_energii import przygotuj_dane_zuzycia_energii

from energia_ruda_dane import przygotuj_dane as przygotuj_dane_energia

X_energia,Y_energia = przygotuj_dane_energia()

models_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(models_path / "Predykcja przepompowanych ścieków"))

from przepompowanie_ruda_dane import przygotuj_dane as przygotuj_dane_przeplyw

X_przeplyw,Y_przeplyw = przygotuj_dane_przeplyw()

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


best_params_l1 = {'learning_rate': 0.049116933567479586, 'num_leaves': 11, 'min_child_samples': 20, 'subsample': 0.8961580569651143, 'colsample_bytree': 0.9013106236449329, 'reg_alpha': 1.7868742439638265, 'reg_lambda': 0.009965203822210505}

# 2. Finalny trening na sztywnej liczbie drzew
final_lgbm_l1 = LGBMRegressor(
    objective="l1",
    **best_params_l1,
    n_estimators=100,   # dokładnie tyle, ile dało minimum błędu
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

final_lgbm_l1.fit(X_train, y_train)

y_pred_test = final_lgbm_l1.predict(X_test)
final_mae = np.mean(np.abs(y_test - y_pred_test))
print(f"\nOstateczne MAE na zbiorze TESTOWYM: {final_mae:.4f}")


best_params_q97 = {'learning_rate': 0.05486811520787273, 'num_leaves': 30, 'min_child_samples': 41, 'subsample': 0.6541648132716525, 'colsample_bytree': 0.7315101306749677, 'reg_alpha': 0.03223315978268554, 'reg_lambda': 0.002719348566685364}

final_lgbm_q97 = LGBMRegressor(
    objective="quantile",
    alpha=0.97,
    **best_params_q97,
    n_estimators=648,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
final_lgbm_q97.fit(X_train, y_train,)

q97_preds = final_lgbm_q97.predict(X_test)


# DO ANALIZY BIZNESOWEJ ##########

from sciezka_projektu import get_project_root

results_test = pd.DataFrame({
    "rzeczywiste": y_test,
    "prognoza": y_pred_test,
    "q97": q97_preds,
}, index=y_test.index)

results_test["alarm"] = results_test["rzeczywiste"] > results_test["q97"]

output = get_project_root() / "reports" / "models"
output.mkdir(parents=True, exist_ok=True)

results_test.to_parquet(output / "ruda_energia.parquet")
print("Zapisano:", output / "ruda_energia.parquet")

#################


test_coverage = np.mean(y_test <= q97_preds)
print(f"\nFinalne Coverage na zbiorze testowym: {test_coverage * 100:.2f}% (cel: 97.00%)")


final_lgbm_l1.booster_.save_model("ruda_energia_median.txt")
final_lgbm_q97.booster_.save_model("ruda_energia_q97.txt")


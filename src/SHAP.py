#SHAP
import matplotlib.pyplot as plt
import shap
def generate_shap_from_tree(model,X_val,date_with_time):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)


    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("Wpływ cech na prognozę (SHAP Beeswarm)", fontsize=13)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Globalna ważność cech (SHAP Bar plot)", fontsize=13)
    plt.tight_layout()
    plt.show()

    idx = X_val.index.get_loc(date_with_time)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
    plt.title(f"Wyjaśnienie predykcji dla punktu {X_val.index[idx]}", fontsize=13)
    plt.tight_layout()
    plt.show()
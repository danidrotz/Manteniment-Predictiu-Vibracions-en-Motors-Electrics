# -------------------------------------------------------------------------------
# Títol: Random Forest en escenari ideal o de control. Train 80 - Val 20.
# Autor: Daniel Delgado Drotz
# TFG 2026: Manteniment predictiu de motors elèctrics mitjançant l’anàlisi de vibracions 
# Escola Universitària Salesiana de Sarrià (EUSS), Barcelona. 
# ----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# Carregar el dataset en format .csv
ds = pd.read_csv(r"C:\Users\dddro\Desktop\MESURES\DATASET\dataset_tres_motors.csv")


# Definir features importants
features = [
    "Ratio41",
    "Ratio21",
    "Pct_E_160_500",
    "SpectralCentroid",
    "Kurtosis"
]

X = ds[features] #Features
y = ds["ESTAT_CONCRET"] #Etiquetes

print("Features utilitzades:")
print(X)

print("Motors:", ds["MOTOR_CONCRET"].unique())
print("Estats:", ds["ESTAT_CONCRET"].unique())

# S'entrena amb un 80% de les mostres i es valida amb un 20%
train_X, val_X, train_y, val_y = train_test_split(
    X, y,
    test_size=0.20, #% de les mostres que es destinen a validar
    random_state=23, #llavor per iniciar en el mateix punt
    stratify=y #manté proporcions
)

# Definir i entrenar el model RF
model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

model_rf.fit(train_X, train_y)


# Prediccio i rendiment
pred_y = model_rf.predict(val_X) 

acc = accuracy_score(val_y, pred_y) #Compara la predicció amb el resultat real
acc*=100

print("\n==============================")
print("RESULTATS Random Forest en ESCENARI IDEAL")
print("==============================")
print(f"Accuracy: {acc}%")

print("\nClassification report:")
print(classification_report(val_y, pred_y))


# Matriu de confusió
labels = sorted(y.unique()) 
mat_conf = confusion_matrix(val_y, pred_y, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=mat_conf, display_labels=labels)

disp.plot(cmap="Greens", values_format="d")
plt.title("Matriu de confusió Random Forest en ESCENARI IDEAL")
plt.xlabel("Predicció")
plt.ylabel("Valor real")
plt.grid(False)
plt.show()


# Extra: importància de variables 
importancies = pd.Series(model_rf.feature_importances_, index=features)
importancies = importancies.sort_values(ascending=True)

plt.figure(figsize=(10,5))
plt.barh(importancies.index, importancies.values)
plt.xlabel("Importància [0 - 1]")
plt.title("Importància de les variables en Random Forest")
plt.grid(axis="x")
plt.show()
# ---------------------------------------------------------------------------------------
# Títol: K-Nearest Neighbors en escenari ideal o de control. Train 80 - Val 20.
# Autor: Daniel Delgado Drotz
# TFG 2026: Manteniment predictiu de motors elèctrics mitjançant l’anàlisi de vibracions 
# Escola Universitària Salesiana de Sarrià (EUSS), Barcelona. 
# ----------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
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

# Definir i entrenar el model KNN
model_knn = Pipeline([
    ("scaler", StandardScaler()), # s'aplica la Z-score (mitjana = 0 i desv = 1)
    ("knn", KNeighborsClassifier(
        n_neighbors=5, # es defineix la k en 5 veïns propers
        weights="distance"
    ))
])

model_knn.fit(train_X, train_y)

# Prediccio i rendiment
pred_y = model_knn.predict(val_X) 

acc = accuracy_score(val_y, pred_y) #Compara la predicció amb el resultat real
acc*=100

print("\n==============================")
print("RESULTATS K-NN en ESCENARI IDEAL")
print("==============================")
print(f"Accuracy: {acc}%")

print("\nClassification report:")
print(classification_report(val_y, pred_y))

# Matriu de confusió
labels = sorted(y.unique())
mat_conf = confusion_matrix(val_y, pred_y, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=mat_conf, display_labels=labels)

disp.plot(cmap="Oranges", values_format="d")
plt.title("Matriu de confusió Random Forest en ESCENARI IDEAL")
plt.xlabel("Predicció")
plt.ylabel("Valor real")
plt.grid(False)
plt.show()
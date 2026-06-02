# ---------------------------------------------------------------------------------
# Títol: SVM en escenari ideal o de control. Train 80 - Val 20.
# Autor: Daniel Delgado Drotz
# TFG 2026: Manteniment predictiu de motors elèctrics mitjançant l’anàlisi de vibracions 
# Escola Universitària Salesiana de Sarrià (EUSS), Barcelona. 
# -------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Carregar el dataset
ds = pd.read_csv(r"C:\Users\dddro\Desktop\MESURES\DATASET\dataset_tres_motors.csv")

print("Dimensions dataset:", ds.shape)
print("Columnes:", ds.columns)
print("Motors:", ds["MOTOR_CONCRET"].unique())
print("Estats:", ds["ESTAT_CONCRET"].unique())


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

print("\nFeatures utilitzades:")
print(X)

# S'entrena amb un 80% de les mostres i es valida amb un 20%
train_X, val_X, train_y, val_y = train_test_split(
    X, y,
    test_size=0.20, #% de les mostres que es destinen a validar
    random_state=23, #llavor per iniciar en el mateix punt
    stratify=y #manté proporcions
)

print("\nMostres entrenament:", train_X.shape[0])
print("Mostres validació:", val_X.shape[0])

# Definir i entrenar el model
model_svm = Pipeline([
    ("scaler", StandardScaler()), # s'aplica la Z-score (mitjana = 0 i desv = 1)
    ("svm", SVC(
        kernel="rbf" #kernel radial (radial basis function)
    ))
])

model_svm.fit(train_X, train_y)

# Prediccions i rendiments
pred_y = model_svm.predict(val_X) #Prediu estats en relació a features que no coneixia

acc = accuracy_score(val_y, pred_y) #Compara la predicció amb el resultat real
acc*=100

print("RESULTATS SVM amb kernel RBF en ESCENARI IDEAL")
print(f"Accuracy: {acc}%")

print("\nClassification report:")
print(classification_report(val_y, pred_y))

# 6. Matriu de confusió
labels = sorted(y.unique()) #Important per mantenir ordres i etiquetes a la matriu
mat_conf = confusion_matrix(val_y, pred_y)

disp = ConfusionMatrixDisplay(confusion_matrix=mat_conf, display_labels=labels)

plt.figure(figsize=(5,5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriu de confusió SVM amb kernel RBF en ESCENARI IDEAL")
plt.xlabel("Predicció")
plt.ylabel("Valor real")
plt.grid(False)
plt.show()
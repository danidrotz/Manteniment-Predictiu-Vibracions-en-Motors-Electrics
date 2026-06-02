# ---------------------------------------------------------------------------------------
# Títol: K-Nearest Neighbors en escenari ideal o de control. Train 80 - Val 20.
# Autor: Daniel Delgado Drotz
# TFG 2026: Manteniment predictiu de motors elèctrics mitjançant l’anàlisi de vibracions 
# Escola Universitària Salesiana de Sarrià (EUSS), Barcelona. 
# Nota: executar en entorn de Google Colab, desde VSCode no reconeix TensorFlow si no es té versió corresponent de Python
# ----------------------------------------------------------------------------------------

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib import pyplot as plt

#Carregar el dataset 
dadesc = pd.read_csv("features_importants.csv")

c = ["S", "U", "L"]
dadesc["estat_num"] = dadesc["ESTAT_CONCRET"].map({
    "S": 0,
    "U": 1,
    "L": 2
})

# Separar X i y
X = dadesc.drop(columns=["ESTAT_CONCRET", "estat_num", "fitxer"], errors="ignore")
y = dadesc["estat_num"]


# Separar entrenament i validació
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)


# Normalitzar dades
scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)


# Crear xarxa neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=[train_X.shape[1]]),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


# Entrenar model
model.fit(train_X, train_y, epochs=2, verbose=1)

# Validar model
test_loss, test_acc = model.evaluate(val_X, val_y, verbose=2)

print("Precisió validació: {:.2f}%".format(test_acc * 100))

# Prediccions
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

prediccions = pd.DataFrame(
    probability_model.predict(val_X),
    columns=llista_classes,
    index=val_y.index
)

r = prediccions.idxmax(axis=1)
val_y_text = val_y.map({
    0: "S", #'Sa', 'Unbalance', 'Looseness'
    1: "U",
    2: "L"
})

errors = r != val_y_text

print("Nombre d'errades:", errors.sum())
print("Percentatge d'encert: {:.2f}%".format(100 * (1 - errors.sum() / len(errors))))

# Matriu de confusió
cm = confusion_matrix(val_y_text, r, labels=llista_classes)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=llista_classes,
            yticklabels=llista_classes)

plt.xlabel("Predicció")
plt.ylabel("Valor real")
plt.title("Matriu de confusió - Xarxa neuronal")
plt.show()

print(classification_report(val_y_text, r))
import numpy as np.

import pandas as pd

from sklearn.model selection import train test split

from sklearn.preprocessing import Standard Scaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy score, classification report, confusion matrix

import matplotlib.pyplot as plt

#Synthetic Pima-like dataset generation

mg= np.random.default rng(42) n=768

Xpd.DataFrame((

"Pregnancies": rng.poisson(3, n),

"Glucose": rng.normal(120, 30, n).clip(50, 200),

"Blood Pressure": rng.normal(70, 12, n).clip(40, 120),

"Skin Thickness": mg.normal(25, 8, n).clip(5, 60),

"Insulin": mg.normal(80, 50, n).clip(0, 400),

"BMI": rng.normal(32, 7, n).clip(1.5, 60),

"Diabetes Pedigree Function": rng.normal(0.5, 0.3, n).clip(0.1, 2.5),

"Age": mg.integers(21, 70, n),

1) logit (0.04*X.Pregnancies + 0.03* (X.Glucose-120)+0.01*(Χ.ΒΜ1-30) + 0.02*(X.Age-35)+0.5* (X.Diabetes Pedigree Function-0.5)-mg.normal(0, 0.7. n))

prob 1/(1+np.exp(-logit))

y=(prob>0.5).astype(int)

#Split and scale

X train, X test, y train, y test train test split(X, y, test size 0.25, random state 7, stratify-y)

scaler StandardScaler()

X train scaled scaler.fit transform(X train)

X test scaled scaler.transform(X test)

#Model training

clf=LogisticRegression(max iter-500, solver "Ibfgs")

elf. fit(X train scaled, y train) y pred clf.predict(X test scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification report(y test, y pred))

#Confusion Matrix Visualization

cm confusion_matrix(y test, y_pred)

plt.imshow(cm, cmap 'Blues')

plt.title("Confusion Matrix-Diabetes Prediction")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

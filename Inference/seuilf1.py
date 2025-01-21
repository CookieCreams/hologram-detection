import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Données
Holo = np.array([165, 99, 94, 100, 77, 175, 89, 108, 132, 148, 103, 111, 84, 100, 135, 147, 98, 161, 89, 101, 100, 149, 96, 134])
No_Holo = np.array([6, 31, 48, 3, 11, 29, 5, 16, 0, 13, 5, 30, 2, 4, 3, 0, 15, 1, 21, 0, 7, 21, 2, 1])

# Labels
y_true = np.array([1] * len(Holo) + [0] * len(No_Holo))  # 1 pour Holo, 0 pour No Holo
y_scores = np.concatenate((Holo, No_Holo))  # Les valeurs à évaluer

# Définir une série de seuils
thresholds = np.linspace(min(y_scores), max(y_scores), num=100)
f1_scores = []

# Calculer le F1 Score pour chaque seuil
for threshold in thresholds:
    y_pred = (y_scores >= threshold).astype(int)  # Prédiction binaire basée sur le seuil
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)

# Trouver le F1 Score maximum
best_f1_score = max(f1_scores)

# Trouver tous les seuils qui produisent le F1 Score maximum
best_thresholds = thresholds[np.where(f1_scores == best_f1_score)]

# Trouver le seuil médian parmi les meilleurs seuils
median_threshold = np.median(best_thresholds)

# Affichage des résultats
print(f'Seuil médian optimal basé sur le F1 Score : {median_threshold}')
print(f'F1 Score correspondant : {best_f1_score:.2f}')

# Affichage du graphique F1 Score vs Seuil
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1 Score', color='purple')
plt.axvline(median_threshold, color='green', linestyle='dashed', label='Seuil optimal')
plt.xlabel('Seuil (nombre d\'holo à détecter)')
plt.ylabel('F1 Score')
plt.title('F1 Score en fonction du seuil')
plt.legend()
plt.show()


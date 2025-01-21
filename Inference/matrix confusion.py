import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Données
Holo = np.array([96, 36, 10, 30, 95, 61, 65, 20, 131, 106, 14, 21, 19, 57, 48, 59, 58, 19, 33, 62, 123, 32, 104, 25, 99, 84, 30, 50, 69, 62])
No_Holo = np.array([8, 0, 0, 1, 0, 0, 0, 1, 2, 21, 0, 0, 0, 4, 0, 0, 4, 1, 1, 11, 7, 0, 0, 2, 0, 0, 1, 2, 0, 15])

# Seuil
seuil_holo = 63

# Prévisions
predictions_holo = Holo >= seuil_holo
predictions_no_holo = No_Holo < seuil_holo

# Calcul de la matrice de confusion
TP = np.sum(predictions_holo)  # Holo correctement classé
FP = np.sum(~predictions_no_holo)  # No_Holo incorrectement classé comme Holo
TN = np.sum(predictions_no_holo)  # No_Holo correctement classé
FN = np.sum(~predictions_holo)  # Holo incorrectement classé comme No_Holo

# Matrice de confusion
matrice_confusion = np.array([[TP, FN], 
                               [FP, TN]])

# Affichage de la matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prédit Holo', 'Prédit No_Holo'], 
            yticklabels=['Réel Holo', 'Réel No_Holo'])
plt.title('Matrice de Confusion')
plt.ylabel('Classe réelle')
plt.xlabel('Classe prédite')
plt.show()

# Calcul de la précision et du rappel
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calcul du F1 Score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Précision: {precision:.2f}")
print(f"Rappel: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

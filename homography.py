import os
import cv2
import json
import numpy as np

# Spécifiez le chemin des dossiers contenant vos images et JSON
image_folder_path = r"D:\Hologram\images\images\origins\passport\psp04_03_03"
json_folder_path = r"D:\Hologram\markup\markup\origins\passport\psp04_03_03"
mask_holo_path = r"D:\Hologram\templates\templates\hologram_masks\passport_hologram_mask.png"
mask_holo = cv2.imread(mask_holo_path)
mask_holo_height, mask_holo_width = mask_holo.shape[:2]
print(mask_holo.shape)

# Liste tous les fichiers du dossier d'images
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')]

# Pour chaque image, traiter la mosaïque
for image_index, image_file in enumerate(image_files):
    # Chemins pour l'image et le fichier JSON
    image_path = os.path.join(image_folder_path, image_file)
    json_path = os.path.join(json_folder_path, image_file + '.json')

    # Lire l'image
    image = cv2.imread(image_path)

    # Lire les données JSON si le fichier JSON existe
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

        # Récupérer les coordonnées du template du json
        coords = json_data['document']['templates']['uto.passport.type4:main']['template_quad']
        input_points = np.array(coords, dtype=np.float32)

        # Définir les points de destination (coins du masque holographique)
        output_points = np.array([[0, 0], [mask_holo_width, 0], [mask_holo_width, mask_holo_height], [0, mask_holo_height]], dtype=np.float32)

        # Calculer l'homographie
        H, status = cv2.findHomography(input_points, output_points)

        # Appliquer l'homographie 
        warped_mask = cv2.warpPerspective(image, H, (mask_holo.shape[1], mask_holo.shape[0]))

        # Obtenir les dimensions de l'image warped_mask
        height, width = warped_mask.shape[:2]

        # Définir le ratio de redimensionnement 
        resize_ratio = 0.2
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)

        # Redimensionner l'image warped_mask par ratio
        resized_warped_mask = cv2.resize(warped_mask, (new_width, new_height))

        # Afficher l'image transformée
        cv2.imshow('Resized Warped Mask', resized_warped_mask)
        cv2.waitKey(0)

# Ne pas oublier de libérer les ressources
cv2.destroyAllWindows()

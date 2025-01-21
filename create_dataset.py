import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import math

# chemin du dossier contenant les dossiers d'images et JSON
base_image_folder_path = r"D:\Hologram\images\images\origins\passport"
base_json_folder_path = r"D:\Hologram\markup\markup\origins\passport"
mask_holo_path = r"D:\Hologram\templates\templates\hologram_masks\passport_hologram_mask.png"
mask_holo = cv2.imread(mask_holo_path)
mask_holo_height, mask_holo_width = mask_holo.shape[:2]
# print(mask_holo_height," et ",mask_holo_width)
# plt.imshow(cv2.cvtColor(mask_holo, cv2.COLOR_BGR2RGB)) 
# plt.title("Mask Holo")
# plt.show()

# Dossier où les mosaïques seront sauvegardées
holo_folder = r"D:\Hologram\dataset\train\Holo"
# holo_folder = r"D:\Hologram\dataset\val\Holo"
# holo_folder = r"D:\Hologram\dataset\test\Holo"
os.makedirs(holo_folder, exist_ok=True)  # Crée le dossier s'il n'existe pas déjà
# no_holo_folder = r"D:\Hologram\dataset\train\No_Holo"
# os.makedirs(no_holo_folder, exist_ok=True)  # Crée le dossier s'il n'existe pas déjà

patch_size = 200
# Decouper les bords du passeport/masque hologramme
left_right_margin = 400 
top_bottom_margin = 200

mask_holo = mask_holo[top_bottom_margin:mask_holo_height-top_bottom_margin, 
                               left_right_margin:mask_holo_width-left_right_margin]
mask_holo_height, mask_holo_width = mask_holo.shape[:2]
# print(mask_holo_height," et ",mask_holo_width)
# plt.imshow(cv2.cvtColor(mask_holo, cv2.COLOR_BGR2RGB)) 
# plt.title("Mask Holo")
# plt.show()

liste_pixel_blanc = []
patch_indices = []  # Liste pour stocker les indices des patches avec > 3000 pixels blancs

# Afficher l'image mask_holo avec des patchs 200x200
for i in range(50, mask_holo_height, patch_size):
    for j in range(20, mask_holo_width, patch_size):
        # Compter le numéro du patch
        patch_index = len(liste_pixel_blanc)
        
        # Dessiner un rectangle rouge pour chaque patch
        cv2.rectangle(mask_holo, (j, i), (j + patch_size, i + patch_size), (0, 0, 255), 1) 

        # Extraire le patch
        patch = mask_holo[i:i + patch_size, j:j + patch_size]

        # Compter le nombre de pixels blancs
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:  # Vérifiez si le patch est complet
            pixel_white_count = np.sum(np.all(patch == 255, axis=-1))  # Comptage des pixels blancs
            liste_pixel_blanc.append(pixel_white_count)
            
            # Stocker l'indice du patch si le nombre de pixels blancs est supérieur à 3000
            if pixel_white_count > 3000:
                patch_indices.append(patch_index)

            # Ajouter le numéro du patch au centre
            cv2.putText(mask_holo, str(patch_index), (j + patch_size // 2 - 10, i + patch_size // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2) 
            
# Afficher le nombre de pixels blancs par patch
# for index, count in enumerate(liste_pixel_blanc):
#     print(f"Patch {index }: {count} pixels blancs")
# print(len(patch_indices))
# print((patch_indices))


# Afficher le masque hologramme avec la grille de patchs
# plt.imshow(cv2.cvtColor(mask_holo, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('Image du masque holographique avec patchs')
# plt.show()
# cv2.imwrite("hologram_with_patchs.png", mask_holo)

# Train
sub_folders = [
    "psp01_01_01", "psp01_01_02", "psp01_01_03",
    "psp03_02_01", "psp03_02_03",
    "psp05_03_01", "psp05_03_02", "psp05_03_03",
    "psp07_04_02", "psp07_04_03",
    "psp09_05_01", "psp09_05_02",
    "psp02_01_01", "psp02_01_02", "psp02_01_03",
    "psp04_02_01", "psp04_02_03",
    "psp06_03_01", "psp06_03_02", "psp06_03_03",
    "psp08_04_02", "psp08_04_03",
    "psp10_05_01", "psp10_05_02",
    "psp01_05_01", "psp01_05_02", "psp01_05_03",
    "psp03_01_01", "psp03_01_03",
    "psp05_02_01", "psp05_02_02", "psp05_02_03",
    "psp07_03_02", "psp07_03_03",
    "psp09_04_01", "psp09_04_02",
    "psp02_05_01", "psp02_05_02", "psp02_05_03",
    "psp04_01_01", "psp04_01_03",
    "psp06_02_01", "psp06_02_02", "psp06_02_03",
    "psp08_03_02", "psp08_03_03",
    "psp10_04_01", "psp10_04_02",
    "psp01_04_01", "psp01_04_02", "psp01_04_03",
    "psp03_05_01", "psp03_05_03",
    "psp05_01_01", "psp05_01_02", "psp05_01_03",
    "psp07_02_02", "psp07_02_03",
    "psp09_03_01", "psp09_03_02",
    "psp02_04_01", "psp02_04_02", "psp02_04_03",
    "psp04_05_01", "psp04_05_03",
    "psp06_01_01", "psp06_01_02", "psp06_01_03",
    "psp08_02_02", "psp08_02_03",
    "psp10_03_01", "psp10_03_03",
    "psp01_03_01", "psp01_03_02", "psp01_03_03",
    "psp03_04_01", "psp03_04_03",
    "psp05_05_01", "psp05_05_02", "psp05_05_03",
    "psp07_01_02", "psp07_01_03",
    "psp09_02_01", "psp09_02_02",
    "psp02_03_01", "psp02_03_02", "psp02_03_03",
    "psp04_04_01", "psp04_04_03",
    "psp06_05_01", "psp06_05_02", "psp06_05_03",
    "psp08_01_02", "psp08_01_03",
    "psp10_02_01", "psp10_02_02"
]

# Validation
# sub_folders = [
#     "psp03_02_02",
#     "psp07_04_01",
#     "psp09_05_03",
#     "psp04_02_02",
#     "psp08_04_01",
#     "psp10_05_03",
#     "psp03_01_02",
#     "psp07_03_01",
#     "psp09_04_03",
#     "psp04_01_02",
#     "psp08_03_01",
#     "psp10_04_03",
#     "psp03_05_02",
#     "psp07_02_01",
#     "psp09_03_03",
#     "psp04_05_02",
#     "psp08_02_01",
#     "psp10_03_03",
#     "psp03_04_02",
#     "psp07_01_01",
#     "psp09_02_03",
#     "psp04_04_02",
#     "psp08_01_01",
#     "psp10_02_03"
# ]

# Test
# sub_folders = [
#     "psp01_02_01",
#     "psp01_02_02",
#     "psp01_02_03",
#     "psp03_03_01",
#     "psp03_03_02",
#     "psp03_03_03",
#     "psp05_04_01",
#     "psp05_04_02",
#     "psp05_04_03",
#     "psp07_05_01",
#     "psp07_05_02",
#     "psp07_05_03",
#     "psp09_01_01",
#     "psp09_01_02",
#     "psp09_01_03",
#     "psp02_02_01",
#     "psp02_02_02",
#     "psp02_02_03",
#     "psp04_03_01",
#     "psp04_03_02",
#     "psp04_03_03",
#     "psp06_04_01",
#     "psp06_04_02",
#     "psp06_04_03",
#     "psp08_05_01",
#     "psp08_05_02",
#     "psp08_05_03",
#     "psp10_01_01",
#     "psp10_01_02",
#     "psp10_01_03",
# ]


for sub_folder in sub_folders:
    print(sub_folder)
    new_folder_path = os.path.join(holo_folder, sub_folder)
    holo_folder = new_folder_path
    os.makedirs(new_folder_path, exist_ok=True)
    number = sub_folder[3:5]
    number = int(number)
    number = str(number)
    # Chemins pour les images et les JSON de chaque sous-dossier
    image_folder_path = os.path.join(base_image_folder_path, sub_folder)
    json_folder_path = os.path.join(base_json_folder_path, sub_folder)

    # Liste tous les fichiers du dossier d'images
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg') ]

    tous_les_patchs = []

    # Dimensions du patch
    patch_size = 200

    # Pour chaque image, traiter la mosaïque
    for image_index, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder_path, image_file)
        json_path = os.path.join(json_folder_path, image_file + '.json')

        image = cv2.imread(image_path)

        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
            type_passeport = "uto.passport.type"+number+":main"
            coords = json_data['document']['templates'][type_passeport]['template_quad']
            input_points = np.array(coords, dtype=np.float32)

            output_points = np.array([[0, 0], [mask_holo_width, 0], [mask_holo_width, mask_holo_height], [0, mask_holo_height]], dtype=np.float32)
            
            H, status = cv2.findHomography(input_points, output_points)

            warped_mask = cv2.warpPerspective(image, H, (mask_holo.shape[1], mask_holo.shape[0]))

            warped_mask = warped_mask[top_bottom_margin:mask_holo_height-top_bottom_margin, left_right_margin:mask_holo_width-left_right_margin]
            
            warped_mask = cv2.resize(warped_mask, (mask_holo.shape[1], mask_holo.shape[0]))
            
            
            # Afficher l'homographie avec les patchs
            # ind = 0
            # for i in range(0, warped_mask.shape[0], patch_size):
            #     for j in range(0, warped_mask.shape[1], patch_size):
            #         if i + patch_size <= warped_mask.shape[0] and j + patch_size <= warped_mask.shape[1]:
            #             # Dessiner un rectangle rouge pour chaque patch
            #             cv2.rectangle(warped_mask, (j, i), (j + patch_size, i + patch_size), (0, 0, 255), 1)  # Rouge
                        
            #             # Ajouter le numéro du patch
            #             cv2.putText(warped_mask, str(ind), (j + 10, i + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            #             ind += 1

            # plt.imshow(cv2.cvtColor(warped_mask, cv2.COLOR_BGR2RGB))
            # plt.axis('off')  # Ne pas afficher les axes
            # plt.title(f'Image warped avec grilles - {image_file}')
            # plt.show()


            patches = []
            for i in range(50, warped_mask.shape[0], patch_size):
                for j in range(20, warped_mask.shape[1], patch_size):
                    patch = warped_mask[i:i + patch_size, j:j + patch_size]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        patches.append(patch)

            tous_les_patchs.append(patches)

    mosaiques = []

    # Transpose the data
    for i in range(len(tous_les_patchs[0])):
        new_entry = []
        for sublist in tous_les_patchs:
            new_entry.append(sublist[i])
        mosaiques.append(new_entry)

    # CREATION MOSAIQUES 2D
    if len(mosaiques) > 0:
        for mosaic_index, patches in enumerate(mosaiques):
            num_patches = len(patches)
            #print(num_patches)
            grid_size = int(math.ceil(math.sqrt(num_patches)))
            #print(grid_size)
            mosaic_image_size = grid_size * patch_size
            mosaic_image = np.zeros((mosaic_image_size, mosaic_image_size, 3), dtype=np.uint8)

            # Remplir la mosaïque
            for idx in range(grid_size * grid_size):
                row = idx // grid_size
                col = idx % grid_size
                if idx < num_patches:  # Si on a un patch à placer
                    mosaic_image[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patches[idx]
                else:  # Sinon on comble avec les premiers patchs
                    mosaic_image[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patches[idx % num_patches]

            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.title(f'Mosaïque {mosaic_index} - {sub_folder}')
            # plt.show()

            if mosaic_index in patch_indices:
            # Sauvegarder la mosaïque
                mosaic_filename = os.path.join(holo_folder, f'{sub_folder}_mosaic_{mosaic_index }_origins_holo.png')
                mosaic_image = cv2.resize(mosaic_image, (224, 224))
                # plt.figure(figsize=(10, 10))
                # plt.imshow(cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                # plt.title(f'Mosaïque {mosaic_index} - {sub_folder}')
                # plt.show()
                #print(mosaic_filename)
                cv2.imwrite(mosaic_filename, mosaic_image)
            # else:
                # mosaic_filename = os.path.join(holo_folder, f'{sub_folder}_mosaic_{mosaic_index }_origins_holo.png')
                # print(mosaic_filename)
                # mosaic_image = cv2.resize(mosaic_image, (224, 224))
                # cv2.imwrite(mosaic_filename, mosaic_image)


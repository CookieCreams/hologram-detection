import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import math

base_image_folder_path = r"D:\Hologram\images\images\fraud\copy_without_holo\passport"
# base_image_folder_path = r"D:\Hologram\images\images\fraud\photo_holo_copy\passport"
# base_image_folder_path = r"D:\Hologram\images\images\fraud\photo_replacement\passport"
# base_image_folder_path = r"D:\Hologram\images\images\fraud\pseudo_holo_copy\passport"

base_json_folder_path = r"D:\Hologram\markup\markup\fraud\copy_without_holo\passport"
# base_json_folder_path = r"D:\Hologram\markup\markup\fraud\photo_holo_copy\passport"
# base_json_folder_path = r"D:\Hologram\markup\markup\fraud\photo_replacement\passport"
# base_json_folder_path = r"D:\Hologram\markup\markup\fraud\pseudo_holo_copy\passport"

mask_holo_path = r"D:\Hologram\templates\templates\hologram_masks\passport_hologram_mask.png"
fraud_folder = r"D:\Hologram\dataset5\test\No_Holo"
os.makedirs(fraud_folder, exist_ok=True)  # Crée le dossier s'il n'existe pas déjà

mask_holo = cv2.imread(mask_holo_path)
mask_holo_height, mask_holo_width = mask_holo.shape[:2]

patch_size = 200
left_right_margin = 400
top_bottom_margin = 200

mask_holo = mask_holo[top_bottom_margin:mask_holo_height-top_bottom_margin, 
                               left_right_margin:mask_holo_width-left_right_margin]
mask_holo_height, mask_holo_width = mask_holo.shape[:2]

# Train
sub_folders = [
    "psp01_01_01",
    "psp03_02_01",
    "psp05_03_01",
    "psp07_04_01",
    "psp02_01_01",
    "psp04_02_01",
    "psp06_03_01",
    "psp08_04_01",
    "psp01_05_01",
    "psp03_01_01",
    "psp05_02_01",
    "psp07_03_01",
    "psp02_05_01",
    "psp04_01_01",
    "psp06_02_01",
    "psp08_03_01",
    "psp01_04_01",
    "psp03_05_01",
    "psp05_01_01",
    "psp07_02_01",
    "psp02_04_01",
    "psp04_05_01",
    "psp06_01_01",
    "psp08_02_01",
    "psp01_03_01",
    "psp03_04_01",
    "psp05_05_01",
    "psp07_01_01",
    "psp02_03_01",
    "psp04_04_01",
    "psp06_05_01",
    "psp08_01_01"
]

# Val
# sub_folders = [
#     "psp09_05_01",
#     "psp10_05_01",
#     "psp09_04_01",
#     "psp10_04_01",
#     "psp09_03_01",
#     "psp10_03_01",
#     "psp09_02_01",
#     "psp10_02_01"
# ]

# Test
# sub_folders = [
#     "psp01_02_01",
#     "psp03_03_01",
#     "psp05_04_01",
#     "psp07_05_01",
#     "psp09_01_01",
#     "psp02_02_01",
#     "psp04_03_01",
#     "psp06_04_01",
#     "psp08_05_01",
#     "psp10_01_01"
# ]


for sub_folder in sub_folders:
    print(sub_folder)
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

            warped_mask = warped_mask[top_bottom_margin:mask_holo_height-top_bottom_margin, 
                                   left_right_margin:mask_holo_width-left_right_margin]
            
            warped_mask = cv2.resize(warped_mask, (mask_holo.shape[1], mask_holo.shape[0]))

            patches = []
            for i in range(0, warped_mask.shape[0], patch_size):
                for j in range(0, warped_mask.shape[1], patch_size):
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

            mosaic_filename = os.path.join(fraud_folder, f'{sub_folder}_mosaic_{mosaic_index }_fraud_copy_without_holo.png')
            # mosaic_filename = os.path.join(fraud_folder, f'{sub_folder}_mosaic_{mosaic_index }_fraud_photo_holo_copy.png')
            # mosaic_filename = os.path.join(fraud_folder, f'{sub_folder}_mosaic_{mosaic_index }_fraud_photo_replacement.png')
            # mosaic_filename = os.path.join(fraud_folder, f'{sub_folder}_mosaic_{mosaic_index }_fraud_pseudo_holo_copy.png')
            #print(mosaic_filename)
            mosaic_image = cv2.resize(mosaic_image, (224, 224))
            cv2.imwrite(mosaic_filename, mosaic_image)

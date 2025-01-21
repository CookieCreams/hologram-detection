# Détection d'hologramme (à partir de vidéos) / Approche par patch

## Objectifs
Le but de ce projet est de pouvoir détecter la présence d'un hologramme sur un passeport et de déterminer si le document est authentique ou s'il s'agit d'une fraude. Pour cela, nous nous sommes servis d'un dataset MIDV-Holo qui possède des vidéos de passeports prises à l'aide de smartphones. À partir des images extraites de ces vidéos, nous essaierons de créer des mosaïques qui représentent l'évolution d'une zone locale sur un passeport.

<img src="readme/holo.png" alt="Pred" width="500"/>

Exemple d'un passeport français contenant un hologramme de la France.

## Dataset

Le dataset utilisé pour créer les mosaïques est celui de MIDV-Holo.

[Lien du dataset](https://github.com/SmartEngines/midv-holo)

Le dataset se compose comme suit :

<img src="readme/midv.png" alt="Pred" width="400"/>

Il contient des vidéos de type "origines" et des vidéos de type "fraude". Pour les fraudes, il existe 4 types : copy_without_holo, pseudo_holo_copy, photo_holo_copy et photo_replacement. Il existe également des fichiers JSON pour chaque image clé qui permettent de réaliser une homographie afin de pouvoir travailler uniquement sur le passeport.

## Méthodologie

<img src="readme/pipeline1.png" alt="Pred" width="1000"/>

Nous utilisons un patch de 200x200 pixels sur les images d'une vidéo. Nous récupérons ces vignettes que nous concaténerons sur une image pour obtenir une mosaïque qui montre l'évolution de cette zone au cours de la vidéo. Nous glissons le patch pour essayer de créer un maximum de mosaïques. Ces mosaïques contenant de l'hologramme ou non constitueront un dataset pour entraîner un modèle CNN afin de classifier ces mosaïques.

Le modèle essaiera de prédire localement sur de nouvelles vidéos s'il y a un bout d'hologramme. Nous pouvons ensuite créer une carte de couleurs de ces prédictions :

<img src="readme/pred.png" alt="Pred" width="500"/>

## Inference

## Résultats

<img src="readme/res1.png" alt="Résultat 1" width="500"/>
<img src="readme/res2.png" alt="Résultat 2" width="500"/>

## Bibliographie

LI Koliaskina et al. “MIDV-Holo : A Dataset for ID Document Hologram Detection in a Video Stream”. In : International Conference on Document Analysis and Recognition. Springer. 2023, p. 486-503. doi : https://doi.org/10.1007/978-3-031-41682-8_30.

Harshal Chaudhari, Rishikesh Kulkarni et M.K. Bhuyan. “Weakly Supervised Learning based Reconstruction of Planktons in Digital In-line Holography”. In : Digital Holography and 3-D Imaging 2022. Optica Publishing Group, 2022, W5A.6. doi : 10.1364/DH.2022.W5A.6. url : https://opg.optica.org/abstract.cfm?URI=DH-2022-W5A.6.

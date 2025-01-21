# Détection d'hologramme (à partir de vidéos)
# Approche par patch

## Objectifs
Le but de ce projet est de pouvoir détecter la présence d'un hologramme sur un passeport et de déterminer si le document est authentique ou s'il s'agit d'une fraude. 
Pour cela, on s'est servi d'un dataset MIDV-Holo qui possède des vidéos de passeport prise à l'aide de smartphones. A partir des frames de ces vidéos, on essayera de créer des mosaïques qui représentent l'évolution d'un zone locale sur un passeport.

## Dataset

![](readme/holo.png)

Exemple d'un passeport français contenant un hologramme de la France

## Méthodologie

![](readme/pipeline.png)

On utilise un patch de 200x200 pixels sur les frames d'une vidéo. On récupère ces vignettes que l'on concatenera sur une image pour obtenir une mosaïque qui montre l'évolution de cette zone au cours de la vidéo. On glisse le patch pour essayer de créer un maximum de mosaïques. Ces mosaïques contenant de l'hologramme ou non constitueront un dataset pour entrainer un modèle CNN pour classifier ces mosaïques. Le modèle pourra essayer de prédire localement sur de nouvelles vidéos si il y a un bout d'hologramme.

## Résultats


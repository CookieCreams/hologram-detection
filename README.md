# Détection d'hologramme (à partir de vidéos)
# Approche par patch

## Objectifs
Le but de ce projet est de pouvoir détecter la présence d'un hologramme sur un passeport et de déterminer si le document est authentique ou s'il s'agit d'une fraude. 
Pour cela, on s'est servi d'un dataset MIDV-Holo qui possède des vidéos de passeport prise à l'aide de smartphones. A partir des frames de ces vidéos, on essayera de créer des mosaïques qui représentent l'évolution d'un zone locale sur un passeport.

![](readme/holo.png)
Exemple d'un passeport français contenant un hologramme de la France

## Méthodologie

![](readme/pipeline.png)

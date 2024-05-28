

# 1:Reconnaissance Écriture Manuscrite (OCR) avec EasyOCR

## Description

EasyOCR est une bibliothèque d'extraction de données à partir de documents numérisés. Elle prend en charge plus de 80 langues,Vous pouvez consulter la liste complète des langues supportées dans la [documentation officielle](https://github.com/JaidedAI/EasyOCR). utilise des modèles de détection et d'identification de texte pré-entraînés et met l'accent sur la rapidité et l'efficacité de la reconnaissance des mots à l'intérieur des images.

## Composants Principaux d'EasyOCR

EasyOCR se compose de trois composants principaux :
1. **Extraction de fonctionnalités** : Utilise des modèles d'apprentissage en profondeur tels que ResNet et VGG. Ces modèles sont responsables de l'identification et de l'extraction des caractéristiques pertinentes du texte dans les images.
2. **Étiquetage de séquence** : Utilise les réseaux LSTM (Long Short-Term Memory) pour interpréter le contexte séquentiel des entités extraites. Cela permet de comprendre la relation entre les caractères dans une séquence, ce qui est crucial pour une reconnaissance précise du texte.
3. **Décodage** : Utilise l'algorithme de classification temporelle connexionniste (CTC) pour convertir les séquences de caractéristiques extraites en texte lisible. Le CTC permet de gérer les séquences de longueur variable et de produire le texte final reconnu.
4. 

## Installation

Pour installer EasyOCR, vous pouvez utiliser la commande suivante pour toutes les langues :
```bash
pip install easyocr
```
Pour installer une langue spécifique, utilisez une commande comme celle-ci (par exemple, pour le chinois) :
```bash
pip install easyocr[fr]
```
**Remarque** : Toutes les langues ne sont pas compatibles entre elles, vous ne pouvez donc pas installer plusieurs langues en même temps.

## Utilisation d'EasyOCR

Pour utiliser EasyOCR, vous devez importer les bibliothèques nécessaires pour lire les images et les afficher, comme `cv2` et `matplotlib`. Vous pouvez également utiliser `PIL` pour effectuer des traitements sur les images.

## Fonctionnement d'EasyOCR

1. **Initialisation du lecteur** : La première étape consiste à initialiser un lecteur EasyOCR en spécifiant les langues que vous souhaitez reconnaître.
2. **Chargement de l'image** : Chargez l'image contenant le texte à reconnaître.
3. **Reconnaissance du texte** : Utilisez la méthode `readtext` du lecteur EasyOCR pour reconnaître le texte dans l'image. Vous pouvez spécifier des paramètres tels que `detail` pour obtenir une sortie simple ou détaillée, et `paragraph` pour combiner tous les résultats en un seul paragraphe.
4. **Affichage des résultats** : Les résultats peuvent être affichés ou traités selon vos besoins.
   

## Images du Projet

Voici quelques captures d'écran du projet :

![Exemple d'image 1](![easy_ocr](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/f4df1765-c9de-44ff-b4af-538f984a8771)
*Description de l'image 1*

![Exemple d'image 2]![easy_ocr](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/798ad456-685b-40d8-a8d1-6663320a8e1d)

*Description de l'image 2*

## Conclusion

EasyOCR est une bibliothèque puissante pour la reconnaissance de texte dans les images, avec un support pour une grande variété de langues et des modèles de pointe pour l'extraction et l'interprétation du texte. Grâce à sa simplicité d'utilisation et à sa rapidité, EasyOCR est un excellent choix pour les projets d'OCR.

Pour plus d'informations et d'exemples d'utilisation, vous pouvez consulter la [documentation officielle d'EasyOCR](https://github.com/JaidedAI/EasyOCR).
_____________________________________________________________________________________________________________________________________________________________________________
# 2:Reconnaissance Écriture Manuscrite (OCR) avec Keras-OCR

## Keras-OCR

Keras-OCR est une bibliothèque de bout en bout pour la détection et la reconnaissance de texte (OCR). Elle est implémentée avec CRAFT comme détecteur de texte et CRNN comme reconnaisseur de texte.

### 1:Composants Principaux de Keras-OCR

1. **Détection de Texte (CRAFT)** : Un modèle pour détecter les régions de texte dans les images.
2. **Reconnaissance de Texte (CRNN)** : Un modèle de réseau neuronal récurrent convolutionnel (CRNN) populaire pour la reconnaissance de texte.

### 2:Caractéristiques de Keras-OCR

- **Fine-tuning** : Supporte le fine-tuning sur des jeux de données personnalisés. Vous pouvez affiner séparément le détecteur et le reconnaisseur.

Bien sûr, voici une version améliorée de cette section :

### 3:Installation de Keras-OCR

Pour installer Keras-OCR, utilisez la commande suivante :

```bash
pip install keras-ocr
```

### 4:Utilisation de Keras-OCR

Pour utiliser Keras-OCR, vous devez importer les bibliothèques nécessaires comme `tensorflow` et `keras`.

1. **Importation des bibliothèques** : Assurez-vous d'importer toutes les bibliothèques requises.

2. **Initialisation du pipeline** : Créez un pipeline Keras-OCR pour la détection et la reconnaissance de texte.

3. **Chargement et traitement de l'image** : Chargez l'image que vous souhaitez traiter et convertissez-la en RGB si nécessaire.

4. **Reconnaissance du texte** : Utilisez le pipeline pour reconnaître le texte dans l'image.

```python
import keras_ocr
import tensorflow as tf
import keras

pipeline = keras_ocr.pipeline.Pipeline()

#  RGB
img_path = 'path_to_your_image.jpg'
img_rgb = keras_ocr.tools.read(img_path)


results = pipeline.recognize([img_rgb])


for (box, text) in results[0]:
    print(f'Text: {text}, Box: {box}')
```

### 5:Ressources de Keras-OCR

Le notebook Colab contient des informations supplémentaires sur l'exécution de ce modèle en temps réel. Les poids du modèle original sont fournis dans ce dépôt.
Par défaut, la prédiction sera tracée sous forme de boîtes autour du texte dans l'image, comme indiqué ci-dessous.

## Images du Projet

Voici quelques captures d'écran du projet :

![Exemple d'image 1]![Screenshot 2024-05-28 122128](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/745a208c-2d24-4148-97fc-339680b30b1d)
*Description de l'image 1*

![Exemple d'image 2](path/to/your/image2.png)
*Description de l'image 2*

## Conclusion

EasyOCR et Keras-OCR sont deux bibliothèques puissantes pour la reconnaissance de texte dans les images. EasyOCR est particulièrement utile pour les applications multilingues rapides et simples. Il est compatible avec 80 langues, tandis que Keras-OCR utilise l'anglais par défaut, mais permet le fine-tuning pour d'autres langues. Keras-OCR offre une flexibilité avancée et des options de fine-tuning pour les utilisateurs nécessitant des ajustements spécifiques.

Pour plus d'informations, consultez les documentations officielles d'[EasyOCR](https://github.com/JaidedAI/EasyOCR) et de [Keras-OCR](https://github.com/faustomorales/keras-ocr).








## Reconnaissance Écriture Manuscrite (OCR) avec EasyOCR,Keras-OCR et tesseract


# 1:Reconnaissance Écriture Manuscrite (OCR) avec EasyOCR

## Description

EasyOCR est une bibliothèque d'extraction de données à partir de documents numérisés. Elle prend en charge plus de 80 langues,Vous pouvez consulter la liste complète des langues supportées dans la [documentation officielle](https://github.com/JaidedAI/EasyOCR). utilise des modèles de détection et d'identification de texte pré-entraînés et met l'accent sur la rapidité et l'efficacité de la reconnaissance des mots à l'intérieur des images.

## Composants Principaux d'EasyOCR

EasyOCR se compose de trois composants principaux :
1. **Extraction de fonctionnalités** : Utilise des modèles d'apprentissage en profondeur tels que ResNet et VGG. Ces modèles sont responsables de l'identification et de l'extraction des caractéristiques pertinentes du texte dans les images.
2. **Étiquetage de séquence** : Utilise les réseaux LSTM (Long Short-Term Memory) pour interpréter le contexte séquentiel des entités extraites. Cela permet de comprendre la relation entre les caractères dans une séquence, ce qui est crucial pour une reconnaissance précise du texte.
3. **Décodage** : Utilise l'algorithme de classification temporelle connexionniste (CTC) pour convertir les séquences de caractéristiques extraites en texte lisible. Le CTC permet de gérer les séquences de longueur variable et de produire le texte final reconnu.
   
    ![image](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/918d7f9b-eb02-4d45-b655-4fdeb75440d1)

                                             EasyOCR Framework
## Installation

Pour installer EasyOCR, vous pouvez utiliser la commande suivante pour toutes les langues :
```bash
pip install easyocr
```
Pour installer une langue spécifique, utilisez une commande comme celle-ci (par exemple, pour le français) :
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

### Image avec EasyOCR

![easy_ocr](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/f4df1765-c9de-44ff-b4af-538f984a8771)

### Tableau des résultats

![result_table](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/798ad456-685b-40d8-a8d1-6663320a8e1d)

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
### Image avec  Keras-OCR
![Screenshot 2024-05-28 122128](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/745a208c-2d24-4148-97fc-339680b30b1d)

## Conclusion

EasyOCR et Keras-OCR sont deux bibliothèques puissantes pour la reconnaissance de texte dans les images. EasyOCR est particulièrement utile pour les applications multilingues rapides et simples. Il est compatible avec 80 langues, tandis que Keras-OCR utilise l'anglais par défaut, mais permet le fine-tuning pour d'autres langues. Keras-OCR offre une flexibilité avancée et des options de fine-tuning pour les utilisateurs nécessitant des ajustements spécifiques.

Pour plus d'informations, consultez les documentations officielles d'[EasyOCR](https://github.com/JaidedAI/EasyOCR) et de [Keras-OCR](https://github.com/faustomorales/keras-ocr).


3:Reconnaissance Écriture Manuscrite (OCR) avec pytesseract

Bien sûr, voici le README avec le paragraphe souligné pour mettre en évidence cette information :

---

# 3:Reconnaissance Écriture Manuscrite (OCR) avec PyTesseract

PyTesseract est une solution OCR open-source qui extrait le texte imprimé ou écrit des images. Il a été développé à l’origine par **Hewlett-Packard** et est actuellement maintenu par Google. Tesseract prend en charge la reconnaissance linguistique pour plus de **100 langues**.



## 1/Configuration de Tesseract

### Installation de Tesseract

Pour utiliser PyTesseract, vous devez d'abord installer Tesseract. Vous pouvez le faire en utilisant les commandes suivantes :

```bash
# Installer Tesseract OCR
!apt-get install -y tesseract-ocr
!apt-get install -y libtesseract-dev

# Installer PyTesseract
!pip install pytesseract
```

### 2/Configuration de PyTesseract

Une fois Tesseract installé, vous devez vous assurer que le chemin du fichier exécutable de Tesseract est correctement configuré. Voici un exemple de configuration sous Python :

```python
import pytesseract

# Définir le chemin de l'exécutable Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

## 3/Fonctionnement de PyTesseract

PyTesseract est un wrapper Python pour l'OCR Tesseract de Google. Il peut lire tous les fichiers image supportés par des bibliothèques d’imagerie telles que Leptonica et Pillow, y compris les formats JPEG, PNG, GIF, BMP, TIFF et bien d’autres. Il est donc souvent utilisé dans les cas d'utilisation de l'OCR en Python pour convertir des images en texte."

### Utilisation de PyTesseract

La reconnaissance de texte à l'aide de PyTesseract peut se faire en une seule ligne de code. Voici un exemple simple d'utilisation :

```python
import cv2
import pytesseract


img = cv2.imread('path_to_your_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)

print(text)
```

## 4/Limites de Tesseract

Bien que Tesseract soit puissant, il a certaines limites :
- **Qualité de l'image** : Tesseract est sujet à des erreurs si la séparation entre le premier plan et l'arrière-plan de l'image n'est pas claire.
- **Écriture manuscrite** : Tesseract ne reconnaît pas l'écriture manuscrite.

### Remarque

Dans notre cas, PyTesseract ne fonctionne pas bien avec des images comme Genova.png. Cependant, **il fonctionne beaucoup mieux avec des images comme Extrait_IQOA_data.png, en particulier celles contenant des tableaux.

## Exemples d'Images du Projet

Voici quelques captures d'écran du projet :

### Images avec   PyTesseract
![Exemple d'image 1]![image](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/43563966-e620-4be5-90ec-a576a85b7a5d)
![Screenshot 2024-05-28 124223](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/48f9eb9a-015c-4fe2-b0d5-a3c1c84af333)


## 5/Conclusion

PyTesseract est une bibliothèque puissante pour la reconnaissance de texte dans des formats PDF ou des images sans arrière-plans complexes, comme dans notre exemple. Bien qu'il ait des limitations avec les écritures manuscrites et certaines images de faible qualité, il reste un outil très utile pour extraire du texte imprimé de diverses images.

Pour plus d'informations, consultez la [documentation officielle de Tesseract](https://github.com/tesseract-ocr/tesseract) et de [PyTesseract](https://github.com/madmaze/pytesseract).

 ## Voici quelque exemples:
 ![image](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/3eaafcd0-c932-4a95-ac4c-cc045304fc2c)

 ![Screenshot 2024-05-28 210055](https://github.com/khadidjasift/OCR_easyocr_-keras_ocr/assets/37297751/550e0064-845a-492f-ae53- aebfd7014f69)









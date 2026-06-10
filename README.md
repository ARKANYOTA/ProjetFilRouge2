# ProjetFilRouge2 - TILDA Textile Classifier & AI Safety Evaluation

## 1. Installation
Installez les dépendances du projet :
```bash
pip install -r requirements.txt
```

## 2. Utilisation
Lancer le script principal d'orchestration :
```bash
python -m src.main
```
Les hyperparamètres peuvent être surchargés en créant un fichier `.env` à la racine ou en définissant des variables d'environnement. Exemple :
```bash
MODEL_NAME=resnet18 BATCH_SIZE=64 python -m src.main
```

## 3. Qualité du Code
Pour garantir la propreté du code et le respect des normes strictes du projet :

### Vérifier le typage (Mypy)
```bash
mypy src tests
```

### Vérifier et corriger le lintage/formatage (Ruff)
```bash
# Analyse de la propreté du code
ruff check src tests

# Formater le code automatiquement
ruff format src tests
```

## 4. Tests Unitaires
Pour exécuter la suite de tests (vérification des dimensions des tenseurs, logique de biais, etc.) :
```bash
pytest
```
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

## 3. Prédiction Kaggle et validation manuelle

Générer localement le CSV avec le meilleur checkpoint ResNet-34 :

```bash
python3 -m src.predict
```

Cette commande ne contient aucune fonction d'upload et ne contacte pas l'API de soumission Kaggle. Elle produit :

- `results/kaggle_submission_resnet34.csv` : seul fichier à soumettre ;
- `results/kaggle_submission_resnet34_audit.csv` : confidences pour inspection locale, à ne pas soumettre ;
- `results/kaggle_submission_resnet34_manifest.json` : provenance, comptages et empreintes SHA-256.

Revalider exactement le fichier après une inspection ou une modification manuelle :

```bash
python3 -m src.predict \
  --validate-only results/kaggle_submission_resnet34.csv
```

Le validateur exige l'en-tête `id;label`, 789 prédictions, les identifiants uniques dans l'ordre numérique exact des images de test, deux colonnes seulement et des labels entiers de 0 à 7. Le séparateur `;` reprend celui de `train.csv`, car l'archive fournie ne contient pas de fichier d'exemple de soumission. Avant l'envoi manuel, vérifier que l'aperçu Kaggle reconnaît bien deux colonnes. Si Kaggle demande explicitement une virgule, régénérer un nouveau fichier avec :

```bash
python3 -m src.predict \
  --delimiter ',' \
  --output results/kaggle_submission_resnet34_comma.csv
```

Ne jamais envoyer le fichier `_audit.csv` ni le manifeste JSON.

## 4. Qualité du Code
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

## 5. Tests Unitaires
Pour exécuter la suite de tests (vérification des dimensions des tenseurs, logique de biais, etc.) :
```bash
pytest
```

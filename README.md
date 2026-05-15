# Iris classifier

Walk-through notebook plus a small reproducible training script for the classic Iris dataset using scikit-learn.

## Layout

- `data/` — reserved for local data files (this project loads Iris from scikit-learn).
- `notebooks/iris_model.ipynb` — step-by-step notebook version of the workflow.
- `src/train.py` — trains a decision tree and writes `outputs/model.joblib` and `outputs/confusion_matrix.png`.
- `tests/test_train.py` — basic pytest checks.
- `outputs/` — generated artifacts (created when you run training).

## Setup (Windows)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Setup (macOS / Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train from the command line

From the project root (with the virtual environment activated):

```bash
python src/train.py
```

## Tests

```bash
pytest
```

## Notebook

Open `notebooks/iris_model.ipynb` in Jupyter, VS Code, or Google Colab after installing dependencies.

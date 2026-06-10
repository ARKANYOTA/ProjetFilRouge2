# Instructions for Coding Agents

Welcome. You are working on the TILDA Textile Classifier and AI Safety codebase. To maintain the integrity and high standards of this repository, you **MUST** follow these instructions strictly before finishing your task.

## 1. Static Type Checking (Mypy)
The repository enforces strict static type checking via Mypy.
- **Rule:** There must be **zero** type issues when you finish.
- Run this command and verify that it returns `Success: no issues found`:
  ```bash
  mypy src tests
  ```
- **Never** use type ignores (`# type: ignore`) to hide structural configuration/typing problems. Fix the underlying type signature or wrapper instead.

## 2. Formatting & Linting (Ruff)
Ensure the code conforms to the project's style guidelines.
- **Rule:** No lint errors, and files must be correctly formatted.
- Run the linter to verify:
  ```bash
  ruff check src tests
  ```
- Format the files automatically if you modified them:
  ```bash
  ruff format src tests
  ```

## 3. Unit Tests (Pytest)
Ensure all existing tests pass and write new tests for any added features or components.
- Run tests:
  ```bash
  pytest
  ```

## 4. Coding Standards
- **Imports:** Always include `from __future__ import annotations` at the top of every Python file.
- **Annotations:** Every function and method MUST have complete type annotations for all parameters and return values (use `-> None` for constructors).
- **Configurations:** Do not hardcode paths, hyper-parameters (learning rates, batch sizes), model names, or bias probabilities. Use or add them to `src/config/settings.py` (Pydantic).
- **Device Safety:** Do not assume a CPU or CUDA backend. Check `settings.device` or dynamically select CPU/CUDA/MPS and send all models/tensors via `.to(device)`.
- **Reproducibility:** Seed all random engines (`random`, `numpy`, `torch`) to ensure deterministic outputs for bias simulation.
- **Results Persistence:** Always persist experiment results (e.g., trained model checkpoints, generated plots, log files, evaluation metrics in CSV/JSON format) in the `results` folder for reproducibility and reference.


## 5. Report Compilation & Writing (LaTeX)
When the task involves running experiments, training models, or evaluating bias, you **MUST** document the findings and results in the LaTeX report:
- **File:** [main.tex](file:///Users/marin.decanini/Documents/picsou/Cours/4A/Machine%20Learning/TP_FIL_ROUGE_2/rapport/rapport/main.tex)
- **Language:** The report must be written in **English**.
- **Instructions Reference:** Ensure all questions raised in [projet_kaggle_instructions.pdf](file:///Users/marin.decanini/Documents/picsou/Cours/4A/Machine%20Learning/TP_FIL_ROUGE_2/projet_kaggle_instructions.pdf) are addressed, specifically:
  - Defining the chosen CNN architectures (number of layers, types of layers, rationale).
  - Reporting training times, optimization choices (mini-batch SGD), and parameters tuning.
  - Explaining real-world bias implications (Question 3.1) and documenting the bias experiment results ($model_1$ vs $model_2$, accuracy, and Disparate Impact (DI) metrics).
  - Structuring validation/test comparisons in clean tables/figures.
  - **Visualizations (Figures/Plots):** Generate explanatory charts (training/validation loss curves, accuracy curves, or bias evaluation plots). Save these generated images in the `results/` folder for persistence, copy/save them in the `rapport/rapport/` directory, and explicitly import them into `main.tex` using the `\begin{figure}` block and `\includegraphics` to provide a clear visual presentation of the results.



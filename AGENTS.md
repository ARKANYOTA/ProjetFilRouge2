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

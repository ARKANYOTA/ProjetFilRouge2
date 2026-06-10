from __future__ import annotations

import sys
from src.config.settings import Settings


def main() -> int:
    """Main execution point for model training, evaluation, and bias safety metrics."""
    settings = Settings()
    print(f"Running TILDA Kaggle Classifier using: {settings.model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


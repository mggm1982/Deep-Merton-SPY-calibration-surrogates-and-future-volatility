"""Backwards-compatible entry point; notebook and CLI use the same modules."""
from finance_project.pipeline import main, example_predict, ROOT

if __name__ == "__main__":
    main()

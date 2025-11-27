"""
Chapter 3: Adding Flexibility

Same ML function, now parameterized and configurable without code changes.
"""

from runnable import PythonJob
from functions_parameterized import train_ml_model_flexible

def main():
    """Show how to run the same function with different parameters."""
    print("=" * 50)
    print("Chapter 3: Adding Flexibility")
    print("=" * 50)

    # Same function, now accepts parameters from environment or config files
    job = PythonJob(function=train_ml_model_flexible)
    job.execute()

    print("\n" + "=" * 50)
    print("Parameter flexibility added:")
    print("- 🔧 Function accepts parameters")
    print("- 🌍 Parameters from environment variables")
    print("- 📁 Parameters from YAML config files")
    print("- 🧪 Run different experiments without code changes")
    print("\nTry these commands:")
    print("RUNNABLE_PRM_n_estimators=200 uv run 03_adding_flexibility.py")
    print("uv run 03_adding_flexibility.py --parameters-file experiment_configs/basic.yaml")
    print("=" * 50)

    return job

if __name__ == "__main__":
    main()

"""
Chapter 3: Adding Flexibility

Same ML function, now parameterized and configurable without code changes.
"""

from functions_parameterized import train_ml_model_flexible

from runnable import Catalog, PythonJob


def main():
    """Show how to run the same function with different parameters."""
    print("=" * 50)
    print("Chapter 3: Adding Flexibility")
    print("=" * 50)

    # Define a Catalog to specify what files to save from the run
    catalog = Catalog(put=["model.pkl", "results.json"])

    # Same function, now wrapped as a Job
    job = PythonJob(
        function=train_ml_model_flexible,
        returns=["results"],
        catalog=catalog,
    )
    job.execute()

    print("\n" + "=" * 50)
    print("Parameter flexibility added:")
    print("- 🔧 Function accepts parameters")
    print("- 🌍 Parameters from environment variables")
    print("- 📁 Parameters from YAML config files")
    print("- 🧪 Run different experiments without code changes")
    print("\nTry these commands:")
    print("RUNNABLE_PRM_n_estimators=200 uv run 03_adding_flexibility.py")
    print(
        'RUNNABLE_PARAMETERS_FILE="experiment_configs/basic.yaml" uv run 03_adding_flexibility.py'
    )
    print("=" * 50)

    return job


if __name__ == "__main__":
    main()

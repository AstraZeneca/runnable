"""
Chapter 2: Making It Reproducible

Same ML function, now wrapped as a Runnable Job for automatic tracking.
"""

from runnable import PythonJob
from functions import train_ml_model_basic

def main():
    """Transform the basic function into a tracked, reproducible job."""
    print("=" * 50)
    print("Chapter 2: Making It Reproducible")
    print("=" * 50)

    # Same function, now wrapped as a Job
    job = PythonJob(function=train_ml_model_basic)
    job.execute()

    print("\n" + "=" * 50)
    print("What Runnable added automatically:")
    print("- 📝 Execution logged with timestamp")
    print("- 🔍 Full run details saved to .run_log_store/")
    print("- ♻️  Results preserved (never overwritten)")
    print("- 🎯 Reproducible anywhere with same code")
    print("=" * 50)

    return job

if __name__ == "__main__":
    main()

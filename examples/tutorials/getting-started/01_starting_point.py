"""
Chapter 1: The Starting Point

This is a typical ML function that "works on my laptop" but has common problems
we'll solve throughout the tutorial.
"""

from functions import train_ml_model_basic


def main():
    """Run the basic ML training - notice the problems this creates."""
    print("=" * 50)
    print("Chapter 1: The Starting Point")
    print("=" * 50)

    # This works, but has problems:
    # - No tracking of when it ran or what the results were
    # - Results get overwritten each time
    # - Parameters are hardcoded
    # - No way to reproduce exact results later
    # - Hard to share or deploy

    results = train_ml_model_basic()

    print("\n" + "=" * 50)
    print("Problems with this approach:")
    print("- Results overwrite each time (no history)")
    print("- Parameters hardcoded in function")
    print("- No tracking of execution details")
    print("- Hard to reproduce exact results")
    print("- Difficult to share or deploy")
    print("=" * 50)

    return results


if __name__ == "__main__":
    main()

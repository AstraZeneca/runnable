# This is the user script that runs something
# For example NN training function
import argparse


def func(integer: int, floater: float, stringer: str):
    print("inside func")
    print(f"integer: {integer}")
    print(f"floater: {floater}")
    print(f"stringer: {stringer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example script with named parameters."
    )

    # Add named arguments (options)
    parser.add_argument("--integer", type=str, help="The integer.")
    parser.add_argument("--floater", type=float, help="The floater.")
    parser.add_argument("--stringer", type=str, help="The stringer.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    func(args.integer, args.floater, args.stringer)

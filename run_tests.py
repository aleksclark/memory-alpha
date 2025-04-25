#!/usr/bin/env python
"""
Script to run tests for Memory Alpha.

This script:
1. Checks if required services (Qdrant, Ollama) are running
2. Runs the tests with pytest
"""

import argparse
import os
import subprocess
import sys


def run_command(command, check=True):
    """Run a command and return its output."""
    result = subprocess.run(command, capture_output=True, text=True, check=check)
    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for Memory Alpha")
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip checking services"
    )
    parser.add_argument(
        "--pytest-args", nargs="*", default=[], help="Additional arguments for pytest"
    )
    args = parser.parse_args()

    # Change to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check if required services are running
    if not args.skip_checks:
        print("Checking required services...\n")
        services_check = run_command(["python", "tests/check_services.py"], check=False)
        print(services_check.stdout)

        if services_check.returncode != 0:
            print(
                "Some services are not running. Do you want to continue anyway? (y/n)"
            )
            response = input().strip().lower()
            if response != "y":
                print("Aborting tests.")
                return 1

    # Run tests
    print("\nRunning tests...\n")
    test_command = ["pytest", "-v"] + args.pytest_args
    print(f"Command: {' '.join(test_command)}")

    test_process = subprocess.run(test_command)
    return test_process.returncode


if __name__ == "__main__":
    sys.exit(main())

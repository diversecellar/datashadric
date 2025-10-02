#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup and development helper script for datashadric package
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """run a command and handle errors"""
    print(f"\n📦 {description}")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def install_package():
    """install the package in development mode"""
    commands = [
        ("pip install -e .", "Installing package in development mode"),
        ("pip install -e .[dev]", "Installing development dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def run_tests():
    """run the test suite"""
    commands = [
        ("python -m pytest tests/ -v", "Running test suite"),
        ("python -m pytest tests/ --cov=datashadric --cov-report=term-missing", "Running tests with coverage"),
    ]
    
    for command, description in commands:
        run_command(command, description)


def lint_code():
    """run code formatting and linting"""
    commands = [
        ("python -m black src/", "Formatting code with black"),
        ("python -m flake8 src/", "Running flake8 linting"),
    ]
    
    for command, description in commands:
        run_command(command, description)


def build_package():
    """build the package for distribution"""
    commands = [
        ("python -m build", "Building package"),
        ("twine check dist/*", "Checking package"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def main():
    """main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python setup_dev.py [install|test|lint|build|all]")
        print("\nCommands:")
        print("  install  - Install package in development mode")
        print("  test     - Run test suite")
        print("  lint     - Format and lint code")
        print("  build    - Build package for distribution")
        print("  all      - Run all commands")
        return
    
    command = sys.argv[1].lower()
    
    # ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if command == "install":
        install_package()
    elif command == "test":
        run_tests()
    elif command == "lint":
        lint_code()
    elif command == "build":
        build_package()
    elif command == "all":
        print("🚀 Running full development setup...")
        if install_package():
            lint_code()
            run_tests()
            build_package()
            print("\n🎉 All operations completed!")
        else:
            print("\n❌ Installation failed, skipping other steps")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: install, test, lint, build, all")


if __name__ == "__main__":
    main()
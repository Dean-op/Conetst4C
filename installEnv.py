import subprocess
import sys
import time


def check_and_install_requirements(requirements_file='./requirements.txt'):
    try:
        # Read the requirements file
        with open(requirements_file, 'r') as file:
            requirements = file.readlines()

        # Strip any extra whitespace or newline characters
        requirements = [req.strip() for req in requirements]

        # Function to check if a package is installed
        def is_package_installed(package):
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'show', package], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
                return True
            except subprocess.CalledProcessError:
                return False

        # Install packages if they are not installed
        for req in requirements:
            if req and not is_package_installed(req):
                print(f"Installing {req}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            else:
                print(f"{req} is already installed.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    check_and_install_requirements()
    print("Press Enter to exit...")

import subprocess
import sys

libraries = [
    "ultralytics",
    "neptune",
    "supervision",
    "opencv-python"
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    for library in libraries:
        try:
            print(f"Installing {library}...")
            install(library)
            print(f"Successfully installed {library}.\n")
        except subprocess.CalledProcessError:
            print(f"Failed to install {library}.\n")

if __name__ == "__main__":
    main()

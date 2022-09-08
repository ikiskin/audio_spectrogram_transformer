import os

# Configuration file for parameters
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
data_dir = os.path.join(
    ROOT_DIR, "data", "ESC-50-master", "audio")

label_file = os.path.join(
    ROOT_DIR, "data", "ESC-50-master", "meta", "esc50.csv")

# Signal rate (Hz) for re-sampling:
sr = 8000
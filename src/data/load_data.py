import os
import rarfile
from src.config import Config

rarfile.UNRAR_TOOL = r"C:\Program Files\WinRAR\UnRAR.exe"

if not os.listdir(Config.data_path):
    raise ValueError("Empty data folder")

for file_name in os.listdir(Config.data_path):
    if not file_name.lower().endswith(".rar"):
        continue

    file_path = os.path.join(Config.data_path, file_name)

    with rarfile.RarFile(file_path) as rf:
        rf.extractall(path=Config.extract_data_path)
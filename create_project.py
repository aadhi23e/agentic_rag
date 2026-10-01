# root> python .\create_project.py

from pathlib import Path

root = Path(".")

# All folders list
folders = [
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "data",
    "templates",
    "static",
    "uploads",
    "tests",
]

# All files list
files = [
    "app/main.py",
    "ingest_sample_kb.py",
    "requirements.txt",
    "run.py",
    ".env",
]

# create folder
for folder in folders:
    (root/folder).mkdir(parents=True, exist_ok=True)

# Create files
for file in files:
    (root/file).touch(exist_ok=True)

print("Folder and files has been created")
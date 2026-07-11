"""Download the Lending Club accepted-loans file via kagglehub (anonymous)."""
import os

os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.path.dirname(__file__), "_kagglehub")

import kagglehub

path = kagglehub.dataset_download(
    "wordsforthewise/lending-club",
    path="accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv.gz",
)
print("DOWNLOADED:", path)

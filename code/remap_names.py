import pandas as pd
import numpy as np

df = pd.read_csv("master_csv.csv")

#Remap names of the methods
model_remap = {
    "ThunderSVM": "TSVC",
    "KNeighborsClassifier": "KNC",
    "RandomForestClassifier": "RFC",
    "ExtraTreesClassifier": "ETC",
    "SGDClassifier": "SGDC",
    "HistGradientBoostingClassifier": "HistGBC",
    "HistGradientBoostingClassifier Tuned": "HistGBC_2",
    "KNeighborsClassifier tuned": "KNC_2",
    "HistGradientBoostingClassifier_tuned": "HistGBC_2"
}

#Remap names of the platforms
platform_remap = {
    "Bare metal": "BM",
    "Bare Metal": "BM",
    #"AWS": "AWS",
    "Gcloud": "GCP"
}
    

df["Model Name"] = df["Model Name"].replace(model_remap)
df["Platform"] = df["Platform"].replace(platform_remap)

df.to_csv("master_csv.csv", index=False)
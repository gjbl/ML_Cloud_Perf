### This code generates plots for the training and prediction times of different models, grouped by the number of cores used and the platform.
# It includes box plots and scatter plots to visualize the data and identify trends or saturation points.
# The data is read from a CSV file, and the plots are saved as PNG files.
# The code uses the seaborn and matplotlib libraries for visualization, and pandas for data manipulation.

### --- The data IS NORMALIZED ---

# If you want to generate the files only, comment the lines with plt.show() and uncomment the lines with plt.savefig()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('master_csv.csv')

# Normalize the data based on the platform benchmark points
# Assuming the benchmark points are:
# AWS: 0.519, BM: 1.0, GCP: 0.56
df.loc[df['Platform'] == 'AWS', ['Train Time', 'Prediction Time']] *= 0.519
df.loc[df['Platform'] == 'GCP', ['Train Time', 'Prediction Time']] *= 0.56

import seaborn as sns
import matplotlib.pyplot as plt

models = df["Model Name"].unique()  #Unique models list
platforms = df["Platform"].unique() #Unique platforms list

#define platform color palette
platform_palette = {
    "AWS": "#1f77b4",       # Blue
    "BM": "#ff7f0e", # Orange
    "GCP": "#2ca02c"      # Green
}


### --- BOXPLOT ---

def box_plot(df, model):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'font.family': 'Times New Roman'
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # 1 row, 2 columns

    # --- BOXPLOT TRAINING TIME ---
    sns.boxplot(
        data=df[df["Model Name"] == model],
        x="Cores", y="Train Time", hue="Platform",
        showfliers=False, whis=[5, 95],
        palette=platform_palette,
        ax=axes[0]
    )
    axes[0].set_yscale("log")
    axes[0].set_title(f"{model} - Training Time vs Number of Cores - Normalized")
    axes[0].set_xlabel("Number of cores used")
    axes[0].set_ylabel("Training time [s]")
    axes[0].grid()
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    # --- BOXPLOT PREDICTION TIME ---
    sns.boxplot(
        data=df[df["Model Name"] == model],
        x="Cores", y="Prediction Time", hue="Platform",
        showfliers=False, whis=[5, 95],
        palette=platform_palette,
        ax=axes[1]
    )
    axes[1].set_yscale("log")
    axes[1].set_title(f"{model} - Prediction Time vs Number of Cores - Normalized")
    axes[1].set_xlabel("Number of cores used")
    axes[1].set_ylabel("Prediction time [s]")
    axes[1].grid()
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    #fig.suptitle(f"{model} - Training and Prediction Time vs Number of Cores", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for main title

    filename = f"{model}_train_pred_boxplot_normalized.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    #plt.show()



### --- SCATTERPLOT ---
def scatter_plot(df, model):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'font.family': 'Times New Roman'
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # 1 row, 2 columns

    # --- scatterplot TRAINING TIME ---
    sns.scatterplot(
        data=df[df["Model Name"] == model],
        x="Cores", y="Train Time", hue="Platform",
        palette=platform_palette,
        #dodge=True,  # Separate platforms along x-axis
        #jitter=False,  # Slight horizontal jitter to prevent overlap
        s=80,  # Size of points
        alpha=0.7,  # Transparency
        marker="o",
        edgecolor="black",  # Outline for visibility
        linewidth=0.5,
        ax=axes[0]
    )
    axes[0].set_yscale("log")
    axes[0].set_title(f"{model} - Training Time vs Number of Cores - Normalized")
    axes[0].set_xlabel("Number of cores used")
    axes[0].set_ylabel("Training time [s]")
    axes[0].grid()
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    # --- scatterplot PREDICTION TIME ---
    sns.scatterplot(
        data=df[df["Model Name"] == model],
        x="Cores", y="Prediction Time", hue="Platform",
        palette=platform_palette,
        #dodge=True,  # Separate platforms along x-axis
        #jitter=False,  # Slight horizontal jitter to prevent overlap
        s=80,  # Size of points
        alpha=0.7,  # Transparency
        marker="o",
        edgecolor="black",  # Outline for visibility
        linewidth=0.5,
        ax=axes[1]
    )
    axes[1].set_yscale("log")
    axes[1].set_title(f"{model} - Prediction Time vs Number of Cores - Normalized")
    axes[1].set_xlabel("Number of cores used")
    axes[1].set_ylabel("Prediction time [s]")
    axes[1].grid()
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    #fig.suptitle(f"{model} - Training and Prediction Time vs Number of Cores", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for main title

    filename = f"{model}_train_pred_scatterplot_normalized.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    #plt.show()
    
    
scatter_plot(df, 'TSVC')

for model in models:
    box_plot(df, model) 
    
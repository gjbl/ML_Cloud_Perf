### This code generates plots for the training and prediction times of different models, grouped by the number of cores used and the platform.
# It includes box plots, scatter plots, and saturation plots to visualize the data and identify trends or saturation points.
# The data is read from a CSV file, and the plots are saved as PNG files.
# The code uses the seaborn and matplotlib libraries for visualization, and pandas for data manipulation.

### --- The data IS NOT NORMALIZED ---

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv("master_csv.csv")

models = df["Model Name"].unique()  #Unique models list
platforms = df["Platform"].unique() #Unique platforms list

#define platform color palette
platform_palette = {
    "AWS": "#1f77b4",       # Blue
    "BM": "#ff7f0e", # Orange
    "GCP": "#2ca02c"      # Green
}


### NON-normalized boxplot --- TRAINING TIME and PREDICTION TIME

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
    axes[0].set_title(f"{model} - Training Time vs Number of Cores")
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
    axes[1].set_title(f"{model} - Prediction Time vs Number of Cores")
    axes[1].set_xlabel("Number of cores used")
    axes[1].set_ylabel("Prediction time [s]")
    axes[1].grid()
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    #fig.suptitle(f"{model} - Training and Prediction Time vs Number of Cores", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for main title

    filename = f"{model}_train_pred_boxplot_not_normalized.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

### non-normalized scatterplot
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
    axes[0].set_title(f"{model} - Training Time vs Number of Cores")
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
    axes[1].set_title(f"{model} - Prediction Time vs Number of Cores")
    axes[1].set_xlabel("Number of cores used")
    axes[1].set_ylabel("Prediction time [s]")
    axes[1].grid()
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    #fig.suptitle(f"{model} - Training and Prediction Time vs Number of Cores", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for main title

    filename = f"{model}_train_pred_scatterplot_not_normalized.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    

### saturation plot
def clean_avg_gain_plot(df, model_name):
        
    grouped = (
        df[df['Model Name'] == model_name]
        .groupby(['Platform', 'Cores'], as_index=False)
        .agg({'Train Time': 'mean', 'Prediction Time': 'mean'})
    )

    metrics = ['Train Time', 'Prediction Time']
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharey=True)

    threshold = 0.10

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.set_theme(style="whitegrid")

        for platform, group in grouped.groupby('Platform'):
            group_sorted = group.sort_values('Cores')
            cores = group_sorted['Cores'].values
            times = group_sorted[metric].values

            gains = []
            core_nums = []

            for i in range(1, len(times)):
                prev_time = times[i - 1]
                curr_time = times[i]
                if prev_time > 0:
                    gain = (prev_time - curr_time) / prev_time
                    gains.append(gain)
                    core_nums.append(cores[i])

            # Plot: Lines and points}
            ax.plot(core_nums, gains, label=platform, color=platform_palette[platform], marker='o')
            ax.scatter(core_nums, gains, color=platform_palette[platform], s=60)

            # Saturation point
            for i, g in enumerate(gains):
                if g < threshold:
                    ax.scatter(core_nums[i], g, color='red', s=100, zorder=5, edgecolors='black')
                    ax.text(core_nums[i], g + 0.02, f'{core_nums[i]} cores', color='red', ha='center', fontsize=9)
                    break

        ax.axhline(threshold, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'{model_name} - {metric} Gain')
        ax.set_xlabel('Gain on core number')
        ax.set_ylabel('Average Gain (Δ%)')
        ax.set_ylim(-0.1 , 1)
        red_point = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Saturation Point')
        ax.legend(handles=[red_point] + ax.get_lines()) 

    filename = f"gain_plot_{model_name}.png".replace(" ", "_")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save PNG
    plt.show()



### - DRAW plots 

scatter_plot(df, 'TSVC')

for model in models:
    clean_avg_gain_plot(df, model)
    box_plot(df, model) 
    
#clean_avg_gain_plot(df, 'KNC')
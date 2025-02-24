import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import font_manager

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


method_map = {
    'baseline': 'Baseline',
    'prompting': 'Generation Stage',
    'clustering': 'Clustering Stage',
    'featurization': 'Full Pipeline'
}

colors = sns.color_palette("colorblind")
color_map = {
    'Baseline': colors[0],
    'Generation Stage': colors[1],
    'Clustering Stage': colors[2],
    'Full Pipeline': colors[3]
}

df = pd.read_csv("experiments/data/metrics_calculated.csv")
df['method'] = df['method'].map(method_map)

df = df.groupby(['dataset', 'method', 'top_features'])[['class_coverage', 'reconstruction_accuracy', 'semantic_preservation']].agg({
    'class_coverage': ['mean', 'std'],
    'reconstruction_accuracy': ['mean', 'std'],
    'semantic_preservation': ['mean', 'std']
}).reset_index()

df.columns = ['dataset', 'method', 'top_features', 
              'class_coverage_mean', 'class_coverage_std',
              'reconstruction_accuracy_mean', 'reconstruction_accuracy_std',
              'semantic_preservation_mean', 'semantic_preservation_std']

datasets = ['dbpedia', 'nyt', 'amazon']
metrics = {
    'Class Coverage': 'class_coverage_mean',
    'Reconstruction Accuracy': 'reconstruction_accuracy_mean',
    'Semantic Preservation': 'semantic_preservation_mean'
}

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10.5, 5))

plt.subplots_adjust(top=0.95, wspace=0.1, hspace=0.1)


legend_lines = []
legend_labels = []

for row, dataset in enumerate(datasets):
    for col, (metric_name, column_name) in enumerate(metrics.items()):
        ax = axes[row][col]

        dataset_df = df[df['dataset'] == dataset]

        for method, color in color_map.items():
            method_data = dataset_df[dataset_df['method'] == method]
            
            if not method_data.empty:
                method_data = method_data[(method_data['top_features'] % 5 == 0) & (method_data['top_features'] >= 10)]

                ax.plot(
                    method_data['top_features'], 
                    method_data[column_name],
                    linewidth=2.5,
                    label=method, 
                    color=color,
                    marker='o',
                    markersize=6,
                    markerfacecolor=color
                )

        if col == 0:
            ax.set_ylabel(dataset.upper(), fontsize=13)
        else:
            ax.set_ylabel("")

        ax.set_xlabel("")
        
        if row == 0:
            ax.set_title(metric_name, fontsize=13, pad=5)

        if row == 0 and col == 0:
            legend_lines = [
                plt.Line2D([0], [0], color=color, linewidth=2.5)
                for color in color_map.values()
            ]
            legend_labels = list(color_map.keys())
        
        if ax.get_legend():
            ax.get_legend().remove()

fig.legend(
    legend_lines,
    legend_labels,
    title=None,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=5,
    frameon=False,
    fontsize=13
)

legend_labels.insert(0, "Method:")
legend_lines.insert(0, plt.Line2D([0], [0], color='none'))

fig.text(
    0.5, 0.02, 
    "Number of Features", 
    ha='center', va='center', 
    fontsize=13
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("metrics_comparison.pdf", 
            dpi=600,
            bbox_inches="tight",
            format="pdf")
plt.show()

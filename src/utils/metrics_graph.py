#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

metrics_eco = ["CPU", "GPU", "CO2"]

retrievers = ["retriever1", "retriever2", "retriever3"]
metrics_retriever = ["nDCG", "Precision", "Recall"]
values_retriever = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5]
]
values_eco_retriever = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5]
]

rerankers = ["reranker1", "reranker2", "reranker3"]
metrics_reranker = ["nDCG", "Precision", "Recall"]
values_reranker = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5]
]
values_eco_reranker = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5]
]

rags = ["rag1", "rag2", "rag3"]
metrics_rag = ["Rouge1", "Meteor", "BERTScore"]
values_rag = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5]
]
values_eco_rag = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5]
]

bar_width = 0.2
colors = ["#b4befe","#94e2d5","#eba0ac"]

### RETRIEVER
fig_retriever, ax_retriever = plt.subplots(figsize=(10, 6))
index = np.arange(len(retrievers))

for i, metric in enumerate(metrics_retriever):
    values = [values_retriever[j][i] for j in range(len(retrievers))]
    ax_retriever.bar(index + i * bar_width, values, bar_width, label=metric, color=colors[i])

ax_retriever.set_title('Comparaison des Retriever')
ax_retriever.set_xticks(index + bar_width)
ax_retriever.set_xticklabels(retrievers)
ax_retriever.set_ylim(0, 1)
ax_retriever.yaxis.grid(True)

plt.tight_layout()
plt.savefig('retrievers.png', format='png', dpi=600, transparent=True)
plt.show()

### RERANKER
fig_reranker, ax_reranker = plt.subplots(figsize=(10, 6))
index = np.arange(len(rerankers))

for i, metric in enumerate(metrics_reranker):
    values = [values_reranker[j][i] for j in range(len(rerankers))]
    ax_reranker.bar(index + i * bar_width, values, bar_width, label=metric, color=colors[i])

ax_reranker.set_title('Comparaison des Reranker')
ax_reranker.set_xticks(index + bar_width)
ax_reranker.set_xticklabels(rerankers)
ax_reranker.set_ylim(0, 1)
ax_reranker.yaxis.grid(True)

plt.tight_layout()
plt.savefig('rerankers.png', format='png', dpi=600, transparent=True)
plt.show()

### RAG
fig_rag, ax_rag = plt.subplots(figsize=(10, 6))
index = np.arange(len(rags))

for i, metric in enumerate(metrics_rag):
    values = [values_rag[j][i] for j in range(len(rags))]
    ax_rag.bar(index + i * bar_width, values, bar_width, label=metric, color=colors[i])

ax_rag.set_title('Comparaison des RAG')
ax_rag.set_xticks(index + bar_width)
ax_rag.set_xticklabels(rags)
ax_rag.set_ylim(0, 1)
ax_rag.yaxis.grid(True)

plt.tight_layout()
plt.savefig('rags.png', format='png', dpi=600, transparent=True)
plt.show()

### ECO
colors = ["#74c7ec","#89dceb","#a6e3a1"]

values_eco_retriever = [
    [1.0, 1.5, 0.5],  # CPU, GPU in kWh, CO2 in kg
    [1.2, 1.6, 0.6],
    [1.4, 1.8, 0.7]
]

values_eco_reranker = [ 
    [1.1, 1.4, 0.4],
    [1.3, 1.5, 0.5],
    [1.5, 1.7, 0.6]
]

values_eco_rag = [
    [1.2, 1.6, 0.6],
    [1.3, 1.7, 0.7],
    [1.4, 1.8, 0.8]
]

def plot_eco(values_eco, models, model_type):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    index = np.arange(len(models))

    for i, metric in enumerate(metrics_eco[:2]):
        values = [values_eco[j][i] for j in range(len(models))]
        ax1.bar(index + i * bar_width, values, bar_width, label=f'{metric} (kWh)', color=colors[i])

    ax1.set_ylabel('Énergie (kWh)')
    ax1.set_ylim(0, 2)
    ax1.set_xticks(index + bar_width)
    ax1.set_xticklabels(models)
    ax1.yaxis.grid(True)

    ax2 = ax1.twinx()
    values_co2 = [values_eco[j][2] for j in range(len(models))]
    ax2.bar(index + 2 * bar_width, values_co2, bar_width, label='Émissions de CO2 (kg)', color=colors[2])
    ax2.set_ylabel('CO2 (kg)')
    ax2.set_ylim(0, 1)

    plt.title(f'Comparaison des {model_type}')
    plt.tight_layout()
    plt.savefig(f'eco_{model_type.lower()}.png', format='png', dpi=600, transparent=True)
    plt.show()

plot_eco(values_eco_retriever, retrievers, 'Retriever')
plot_eco(values_eco_reranker, rerankers, 'Reranker')
plot_eco(values_eco_rag, rags, 'RAG')
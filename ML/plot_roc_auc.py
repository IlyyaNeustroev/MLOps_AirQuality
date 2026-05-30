import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Данные

df = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest',
        'Gradient Boosting',
        'SVM',
        'KNN'
    ],
    'CV F1': [0.742, 0.856, 0.861, 0.768, 0.721],
    'Test Acc': [0.738, 0.851, 0.858, 0.765, 0.718],
    'Test Prec': [0.741, 0.854, 0.860, 0.767, 0.720],
    'Test Rec': [0.738, 0.851, 0.858, 0.765, 0.718],
    'Test F1': [0.739, 0.852, 0.859, 0.766, 0.719]
})

plt.style.use('default')

fig, ax = plt.subplots(figsize=(10, 5))

sns.heatmap(
    df.set_index('Model'),
    annot=True,
    cmap='YlGnBu',
    fmt='.3f',
    ax=ax
)

ax.set_title('Heatmap of Model Metrics')

plt.tight_layout()
plt.savefig(
    './ML/reports/01_heatmap_metrics.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


ranking = df.sort_values('Test F1')

fig, ax = plt.subplots(figsize=(8, 5))

ax.barh(
    ranking['Model'],
    ranking['Test F1']
)

ax.set_title('Model Ranking by Test F1')
ax.set_xlabel('Test F1')

plt.tight_layout()
plt.savefig(
    './ML/reports/02_test_f1_ranking.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()

print('Все графики успешно сохранены:')
print('01_heatmap_metrics.png')
print('02_test_f1_ranking.png')


# Данные из classification_report

df = pd.DataFrame({
    'Class': ['0', '1', '2', '3', '4'],
    'Precision': [0.97, 0.92, 0.95, 0.86, 0.72],
    'Recall': [0.98, 0.93, 0.94, 0.89, 0.92],
    'F1': [0.97, 0.92, 0.94, 0.88, 0.81],
    'Support': [7116, 66015, 117369, 17337, 2957]
})

plt.style.use('default')

fig, ax = plt.subplots(figsize=(10, 6))

df_plot = df[['Class', 'Precision', 'Recall', 'F1']].set_index('Class')

df_plot.plot(
    kind='bar',
    ax=ax,
    width=0.8
)

ax.set_title('Classification Metrics by Class')
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(
    './ML/reports/03_class_metrics.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    df['Class'],
    df['Precision'],
    marker='o',
    linewidth=2,
    label='Precision'
)

ax.plot(
    df['Class'],
    df['Recall'],
    marker='s',
    linewidth=2,
    label='Recall'
)

ax.set_title('Precision and Recall by Class')
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_ylim(0.6, 1.02)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(
    './ML/reports/04_precision_vs_recall.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()

print('03_class_metrics.png')
print('04_precision_vs_recall.png')
# ======================================================
# analyze_25_models_FULL_METRICS.py
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import glob
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Load all results + compute FULL metrics
results = []
metrics_summary = []

csv_files = glob.glob("/aakaou/pipeline[1-4]_*_predictions.csv")
print(f"🔍 Found {len(csv_files)} files")

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        filename = csv_file.split('/')[-1].lower()
        
        # Extract model/pipeline
        pipeline = re.search(r'pipeline(\d+)', filename).group(1)
        model_match = re.search(r'(resnet\d+|densenet\d+|efficientnet[b\d]+|mobilenetv?\d+|vgg\d+|inception|xception)', filename)
        model_name = model_match.group(1).title() if model_match else 'unknown'
        
        y_true = df['actual_class_id'].values
        y_pred = df['pred_class_id'].values
        
        # Extract probabilities (columns prob_nv, prob_mel, etc.)
        probs = df[[f'prob_{cls}' for cls in class_names]].values
        
        acc = accuracy_score(y_true, y_pred)
        
        # NEW: Per-class metrics
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                     output_dict=True, zero_division=0)
        f1_macro = report['macro avg']['f1-score']
        
        # NEW: Multi-class AUC
        y_true_bin = label_binarize(y_true, classes=range(7))
        auc_macro = roc_auc_score(y_true_bin, probs, multi_class='ovr', average='macro')
        
        results.append({
            'model': model_name,
            'pipeline': f'P{pipeline}',
            'accuracy': acc,
            'f1_macro': f1_macro,
            'auc_macro': auc_macro,
            'n_samples': len(df)
        })
        
    except Exception as e:
        print(f"⚠️ Skip {csv_file}: {e}")

df_results = pd.DataFrame(results)
print(f"✅ Loaded {len(df_results)} experiments")

# =========================
# 🔥 NEW: BEST MODEL FULL METRICS PLOTS
# =========================

# Find overall best model
best_model = df_results.loc[df_results['accuracy'].idxmax()]
best_model_name = best_model['model']
best_pipeline = best_model['pipeline']

print(f"\n🏆 BEST MODEL: {best_model_name} ({best_pipeline}) - Acc: {best_model['accuracy']:.4f}")

# Load best model's predictions for detailed analysis
best_file = glob.glob(f"/aakaou/*{best_model_name.lower()}*{best_pipeline[1:]}*_predictions.csv")[0]
best_df = pd.read_csv(best_file)

y_true_best = best_df['actual_class_id'].values
y_pred_best = best_df['pred_class_id'].values
probs_best = best_df[[f'prob_{cls}' for cls in class_names]].values

# 1. BEST MODEL DASHBOARD (Figure 1)
fig = plt.figure(figsize=(20, 15))

# Classification Report
plt.subplot(2, 3, 1)
report_dict = classification_report(y_true_best, y_pred_best, target_names=class_names, 
                                  output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).round(3).T.iloc[:-3, :3]  # Remove accuracy/macro/weighted
sns.heatmap(report_df, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Classification Report\n(Precision, Recall, F1)', fontweight='bold')

# Confusion Matrix
plt.subplot(2, 3, 2)
cm = confusion_matrix(y_true_best, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# ROC Curves
plt.subplot(2, 3, 3)
y_true_bin = label_binarize(y_true_best, classes=range(7))
auc_scores = []
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs_best[:, i])
    auc = roc_auc_score(y_true_bin[:, i], probs_best[:, i])
    auc_scores.append(auc)
    plt.plot(fpr, tpr, linewidth=2, label=f'{cls} (AUC={auc:.3f})')

plt.plot([0,1],[0,1], 'k--', alpha=0.5)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-Rest)', fontweight='bold')
plt.legend(loc='lower right', fontsize=8)

# Summary metrics
plt.subplot(2, 3, (4, 6))
metrics_text = f"""
🏆 BEST MODEL: {best_model_name} ({best_pipeline})
─────────────────────────────────
Accuracy:     {best_model['accuracy']:.4f}
F1-Macro:     {best_model['f1_macro']:.4f}
AUC-Macro:    {best_model['auc_macro']:.4f}
N Samples:    {len(best_df)}

PER-CLASS AUC:
"""
for i, cls in enumerate(class_names):
    metrics_text += f"  {cls}: {auc_scores[i]:.3f}\n"

plt.axis('off')
plt.text(0.1, 0.5, metrics_text, fontsize=12, fontfamily='monospace',
         verticalalignment='center', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

plt.suptitle(f'COMPLETE METRICS: {best_model_name} - {best_pipeline}', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('best_model_full_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 📊 ORIGINAL 4 PLOTS + NEW METRICS TABLES
# =========================

# Your original top plots (unchanged but with new F1/AUC data)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 10 Accuracy
top10_acc = df_results.groupby('model')['accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(10)
axes[0,0].barh(range(len(top10_acc)), top10_acc['mean'], xerr=top10_acc['std'], capsize=5)
axes[0,0].set_yticks(range(len(top10_acc))); axes[0,0].set_yticklabels(top10_acc.index)
axes[0,0].set_xlabel('Accuracy ± Std'); axes[0,0].set_title('Top 10 Accuracy')

# Top 10 F1-Macro (NEW!)
top10_f1 = df_results.groupby('model')['f1_macro'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(10)
axes[0,1].barh(range(len(top10_f1)), top10_f1['mean'], xerr=top10_f1['std'], capsize=5)
axes[0,1].set_yticks(range(len(top10_f1))); axes[0,1].set_yticklabels(top10_f1.index)
axes[0,1].set_xlabel('F1-Macro ± Std'); axes[0,1].set_title('Top 10 F1-Score')

# Pipeline winners
for i, pipe in enumerate(['P1', 'P2', 'P3', 'P4']):
    row, col = divmod(i, 2)
    pipe_best = df_results[df_results['pipeline']==pipe].nlargest(5, 'accuracy')
    axes[row,col].barh(pipe_best['model'], pipe_best['accuracy'], color='coral')
    axes[row,col].set_title(f'{pipe}: Top 5'); axes[row,col].set_xlabel('Accuracy')

plt.suptitle('25-MODEL BENCHMARK: Accuracy + F1 + Pipeline Winners', fontsize=16)
plt.tight_layout()
plt.savefig('summary_4plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Save enhanced results
df_results.to_csv('25_models_enhanced_results.csv', index=False)
print("\n✅ SAVED:")
print("   • best_model_full_metrics.png (Fig 1 - STAR PLOT)")
print("   • summary_4plots.png (Fig 2)")
print("   • 25_models_enhanced_results.csv (w/ F1 + AUC)")

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.suptitle(title, fontsize=16, y=0.98)
    y = 0.94
    for para in paragraphs:
        fig.text(0.07, y, para, fontsize=11, va='top', wrap=True)
        y -= 0.06 + (para.count('\n') * 0.03)
        if y < 0.08:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.94
    pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fmt(val):
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'data', 'mining_results')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    out_pdf = os.path.join(results_dir, 'walkthru.pdf')

    # Load results if present
    cls_csv = os.path.join(results_dir, 'classification_results.csv')
    reg_csv = os.path.join(results_dir, 'regression_results.csv')
    summary_png = os.path.join(results_dir, 'model_comparison_summary.png')
    confusion_png = os.path.join(results_dir, 'confusion_matrix_random_forest.png')
    fi_png = os.path.join(results_dir, 'feature_importance.png')
    reg_scatter_png = os.path.join(results_dir, 'regression_scatter.png')
    eda_png = os.path.join(results_dir, 'eda_comprehensive.png')
    pca_png = os.path.join(results_dir, 'dimensionality_reduction.png')
    proc_png = os.path.join(processed_dir, 'processing_overview.png')
    roc_png = os.path.join(results_dir, 'roc_visualization_random_forest.png')

    cls_df = pd.read_csv(cls_csv) if os.path.exists(cls_csv) else pd.DataFrame()
    reg_df = pd.read_csv(reg_csv) if os.path.exists(reg_csv) else pd.DataFrame()

    with PdfPages(out_pdf) as pdf:
        # Cover
        add_text_page(pdf, 'Project Walkthrough — Nuclear Plant Leak Detection', [
            'This walkthrough summarizes the mining tasks performed, the models used, the 80/20 train-test split, and the results with their associated visualizations.\n'
            f'Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}'
        ])

        # Data and preprocessing
        add_text_page(pdf, 'Data & Preprocessing', [
            '- Dataset: 14 days at 1-minute frequency (20,160 rows).',
            '- Preprocessing includes event-aware outlier handling, gap treatment (≤5 min ffill/bfill, ≤30 min interpolation, >30 min flagged), denoising (rolling median + Savitzky–Golay), and PCA (95% variance, 14 components).',
            f'- Visuals: EDA ({os.path.relpath(eda_png, base_dir)}), Processing Overview ({os.path.relpath(proc_png, base_dir)}), PCA panels ({os.path.relpath(pca_png, base_dir)}).'
        ])

        # Multi-class classification
        multi_rows = cls_df[cls_df['Task'] == 'Multi-class'] if not cls_df.empty else pd.DataFrame()
        if not multi_rows.empty:
            lines = [
                'Task: Multi-class risk classification (Normal, Elevated, Medium Risk, High Risk).',
                'Split: 80/20 train-test with stratification. CV: 5-fold on training set.',
            ]
            for _, r in multi_rows.iterrows():
                lines.append(f"- {r['Algorithm']}: Accuracy={fmt(r['Accuracy'])}, CV_Mean={fmt(r['CV_Mean'])}, CV_Std={fmt(r['CV_Std'])}")
            lines += [
                f"Visuals: Confusion Matrix ({os.path.relpath(confusion_png, base_dir)}), Feature Importance ({os.path.relpath(fi_png, base_dir)}), Summary ({os.path.relpath(summary_png, base_dir)}).",
                'Deduction: Perfect separation on synthetic data; radiation sensors and sump flow dominate importance.'
            ]
            add_text_page(pdf, 'Mining Task 1 — Multi-class Classification', lines)

        # Binary classification
        bin_rows = cls_df[cls_df['Task'] == 'Binary'] if not cls_df.empty else pd.DataFrame()
        if not bin_rows.empty:
            lines = [
                'Task: Binary leak detection (Leak vs No-Leak).',
                'Split: 80/20 train-test with stratification.',
            ]
            for _, r in bin_rows.iterrows():
                lines.append(f"- {r['Algorithm']}: Accuracy={fmt(r['Accuracy'])}, AUC={fmt(r['AUC'])}")
            lines += [
                f"Visuals: ROC (Random Forest) ({os.path.relpath(roc_png, base_dir)}), Summary ({os.path.relpath(summary_png, base_dir)}).",
                'Deduction: AUC≈1.00 across models; features provide perfect separability in this synthetic setup.'
            ]
            add_text_page(pdf, 'Mining Task 2 — Binary Classification', lines)

        # Regression
        if not reg_df.empty:
            lines = [
                'Task: Regression of leak rate (LPM) on leak-only subset.',
                'Split: 80/20 train-test.',
            ]
            for _, r in reg_df.iterrows():
                lines.append(f"- {r['Algorithm']}: R²={fmt(r['R²'])}, RMSE={fmt(r['RMSE'])}")
            lines += [
                f"Visuals: Regression Scatter ({os.path.relpath(reg_scatter_png, base_dir)}), Summary ({os.path.relpath(summary_png, base_dir)}).",
                'Deduction: Random Forest achieves R²≈0.997; slight spread only at highest leak rates.'
            ]
            add_text_page(pdf, 'Mining Task 3 — Regression (Leak Rate)', lines)

        # Association rules
        add_text_page(pdf, 'Mining Task 4 — Association Rules', [
            'Task: Mine co-occurrence patterns using Apriori on binned numeric features.',
            'Result: No interesting rules found at support≥0.01 and lift≥1.1.',
            'Deduction: Leak events are sparse with diverse states; lowering thresholds or focusing on precursor subsets could surface rules.'
        ])

    print(f"Walkthrough PDF written to: {out_pdf}")


if __name__ == '__main__':
    main()





# Nuclear Plant Leak Detection via Data Mining Pipeline

An end-to-end data mining pipeline for detecting and characterizing coolant leaks in nuclear power plants using synthetic minute-level sensor telemetry. This project implements comprehensive data preprocessing, feature engineering, and machine learning models to achieve reliable leak detection and risk assessment.

## 🎯 Project Overview

This project demonstrates a complete data mining workflow for nuclear safety applications:

- **Synthetic Data Generation**: Realistic minute-level sensor data with injected leak events, false positives, and realistic data quality issues
- **Intelligent Preprocessing**: Domain-aware data cleaning that preserves genuine leak signatures while removing noise
- **Advanced Feature Engineering**: Composite indices, rolling statistics, and dimensionality reduction (PCA)
- **Multi-Task ML Pipeline**: Multi-class risk prediction, binary leak detection, and leak rate regression
- **Comprehensive Visualization**: ROC curves, PR curves, confusion matrices, feature importance, and exploratory dashboards
- **Production-Ready Artifacts**: Reproducible results with saved models, scalers, and transformers

## 📊 Key Achievements

- **Perfect Classification** on synthetic test data
- **High-Accuracy Regression** for leak rate estimation
- **56 Feature Engineering Transformations** for multi-perspective analysis
- **5 ML Algorithms** benchmarked (Logistic Regression, SVM, Random Forest, KNN, Decision Tree)
- **Comprehensive Reporting** with visualizations and metrics exports

## 📁 Project Structure

```
nuclear_plant_mining/
├── data/
│   ├── raw/                          # Original synthetic data
│   │   ├── nuclear_sensors_raw.csv
│   │   └── nuclear_events_table.csv
│   ├── processed/                    # Cleaned and transformed data
│   │   ├── nuclear_sensors_processed.csv
│   │   ├── feature_metadata.json
│   │   ├── data_quality_report.json
│   │   └── [scalers and transformers]
│   └── mining_results/               # Model outputs and visualizations
│       ├── classification_results.csv
│       ├── regression_results.csv
│       ├── feature_importance.csv
│       ├── [ROC/PR curves]
│       └── Results_Summary.pdf
├── data_mining_pipeline.py           # Main orchestration script
├── generate_dataset.py               # Synthetic data generation
├── preprocess_transform.py           # Data cleaning and feature engineering
├── export_results_summary.py         # Results aggregation
├── export_walkthru.py                # Detailed walkthrough export
├── roc_viz.py                        # Visualization utilities
└── pipeline_summary.txt              # Methodology documentation
```

## 🔧 Installation

### Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, scipy
- matplotlib, seaborn (visualization)
- joblib (model persistence)
- Optional: missingno (missingness visualization)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AbhayankarBellur/Nuclear-leak-predicition.git
   cd Nuclear-leak-predicition/nuclear_plant_mining
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv nuclear_mining_env
   # Windows
   nuclear_mining_env\Scripts\activate
   # Unix/macOS
   source nuclear_mining_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
   ```

## 🚀 Usage

### Run the Complete Pipeline

Execute the full data mining workflow:

```bash
python data_mining_pipeline.py
```

This will:
1. Generate synthetic sensor data with leak events
2. Preprocess and clean the data
3. Engineer features from raw signals
4. Train and evaluate ML models
5. Generate visualizations and reports
6. Export all results to `data/mining_results/`

### Individual Components

**Generate synthetic data only**:
```bash
python generate_dataset.py
```

**Preprocess and transform data**:
```bash
python preprocess_transform.py
```

**Create ROC/PR visualizations**:
```bash
python roc_viz.py
```

**Export detailed summary**:
```bash
python export_results_summary.py
```

## 📈 Methodology

### Data Generation

- **14 days** of synthetic data at **1-minute frequency** (20,160 samples)
- **23 primary sensors** covering:
  - Reactor Coolant System (RCS) parameters
  - Safety system instrumentation
  - Radiation monitoring
  - Thermal hydraulics
- **8 true leak events** (mix of gradual and rapid onset)
- **5 false positive events** for robustness testing
- **Realistic data quality**: MCAR missingness (~2%), outages (1-6 hours), sensor drift

### Preprocessing Strategy

1. **Range Validation**: Fail loudly on physical impossibilities
2. **Gap Handling**:
   - Short gaps (≤5 min): Forward/backward fill
   - Medium gaps (≤30 min): Time-based interpolation
   - Long gaps (>30 min): Flag without imputation
3. **Outlier Detection**: Global (IQR) + Local (rolling MAD)
4. **Denoising**: Savitzky-Golay filtering (preserves edges)
5. **Drift Correction**: Median-shift estimation and correction
6. **Flagging**: Explicit boolean columns for data quality issues

### Feature Engineering

**Composite Indices** (domain-driven):
- Radiation Index: Normalized airborne gamma
- Hydraulic Imbalance: RCS flow balance anomalies
- Thermal Stress: Temperature gradients and rates

**Rolling Statistics** (30-minute windows):
- Mean, std, skew, kurtosis
- Min/max rate of change
- Signal entropy

**Dimensionality Reduction**:
- Yeo-Johnson transformation (handles zero/negative values)
- StandardScaler normalization
- PCA with 95% variance retention

**Total**: 56 engineered features per timestamp

### Modeling Pipeline

**Classification Tasks**:
1. **Multi-class Risk**: {Normal, Low Risk, High Risk, Critical}
2. **Binary Leak Detection**: {No Leak, Leak}

**Regression Task**:
- Leak rate estimation (L/min)

**Algorithms Evaluated**:
- Logistic Regression (baseline)
- Support Vector Machine (SVM with RBF kernel)
- Random Forest (50 trees)
- K-Nearest Neighbors (k=5)
- Decision Tree (max_depth=10)

**Validation**: 5-fold cross-validation with stratification

### Output Metrics

**Classification**:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion matrices

**Regression**:
- MAE, RMSE, R² Score
- Cross-validation scores

## 📊 Results Summary

Results are exported to `data/mining_results/`:

- `classification_results.csv` - Per-algorithm metrics
- `regression_results.csv` - Regression performance
- `binary_cv_scores.csv` - Binary classification fold scores
- `regression_cv_scores.csv` - Regression fold scores
- `feature_importance.csv` - Feature importance rankings
- `summary_statistics.csv` - Comprehensive statistics
- PDF visualizations - ROC curves, confusion matrices, dashboards

### Key Performance Indicators

All ML models achieved **perfect classification** on this synthetic dataset:
- Accuracy: 100%
- ROC-AUC: 1.00
- PR-AUC: 1.00

Regression results show high fidelity leak rate estimation with R² > 0.95 on test folds.

## 📝 Project Documentation

### Main Report
See `Nuclear_Plant_Leak_Detection_Report.txt` for:
- Detailed methodology and theory
- Experimental setup
- Data quality analysis
- Feature engineering rationale
- Results interpretation
- Production recommendations

### Technical Pipeline Guide
`nuclear_plant_mining/pipeline_summary.txt` includes:
- Section-by-section code snippets
- Rationale for key design choices
- Reproducibility notes

## 🎓 Educational Value

This project demonstrates:

1. **End-to-End Data Mining Workflow**
   - From problem formulation to actionable insights
   
2. **Domain-Driven Engineering**
   - How domain knowledge improves feature engineering
   - Physics-inspired preprocessing strategies

3. **Robust Data Handling**
   - Realistic quality issues and solutions
   - Transparent, auditable preprocessing

4. **Model Evaluation Best Practices**
   - Multi-metric assessment
   - Cross-validation and stratification
   - Visualization-first interpretation

5. **Production Considerations**
   - Reproducible pipelines
   - Artifact persistence
   - Clear documentation

## 🔬 Methodology Highlights

### Why This Approach Works

1. **Minute-Level Cadence**: Captures short-term leak dynamics without requiring sequence learners
2. **Event-Aware Cleaning**: Long outages are flagged, not fabricated
3. **Composite Indices**: Improves interpretability and domain alignment
4. **Multi-Task Learning**: Leak detection + severity estimation provides complementary views
5. **Transparent Preprocessing**: All transformations are auditable and reproducible

### Scalability to Real Data

The pipeline is designed for production deployment:
- Modular components can be retrained independently
- Artifacts (scalers, transformers) are serialized for inference
- All hyperparameters are configurable
- Data quality issues are explicit and logged

## 📚 References

- Scikit-Learn Documentation: https://scikit-learn.org/
- Pandas User Guide: https://pandas.pydata.org/docs/
- Nuclear Safety Principles: Industry best practices in leak detection

## 👥 Team

**Course**: Data Mining (ISWE209L), Fall 2025-26  
**Institution**: Vellore Institute of Technology (VIT)  
**Submission Date**: November 12, 2025

## 📄 License

This project is provided for educational purposes.

## 🤝 Contributing

For improvements or bug fixes:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Push to the branch
5. Open a Pull Request

## 📧 Questions or Issues

Please open an issue on GitHub or contact the project maintainers.

---

**Last Updated**: May 2026  
**Status**: Complete and validated on synthetic data

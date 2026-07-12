# Credit_Risk_Project
[First of all, I want to inform you that the methodology of this project is well funded however the data set was created by the author themself for educational purposes. Thus data is far from a monster.]

This project develops a comprehensive end-to-end credit risk assessment framework that combines machine learning techniques with traditional credit risk management principles to support lending decisions. The primary objective is to estimate the probability that a borrower will default on a loan and translate these predictions into actionable business decisions through expected loss estimation and an internal credit decision engine.

The project begins with extensive exploratory data analysis (EDA), data preprocessing, and feature engineering to identify the most informative borrower characteristics. Several classification algorithms, including Logistic Regression, Random Forest, XGBoost, and LightGBM, are developed and optimized using nested cross-validation and Bayesian hyperparameter optimization (Optuna) to ensure unbiased model selection and robust performance evaluation.

Model performance is assessed using a comprehensive set of classification metrics, including Accuracy, Precision, Recall, F1-score, ROC-AUC, Precision-Recall AUC, calibration analysis, confusion matrices, and learning curves. Model explainability is incorporated through coefficient interpretation for Logistic Regression and SHAP analysis for LightGBM, providing insights into the factors that contribute most significantly to default risk. Robustness analyses are also performed to evaluate the stability and generalization capability of the selected models.

Following comparative evaluation, Logistic Regression is selected as the champion model due to its strong predictive performance, calibration, and interpretability, while LightGBM serves as the challenger model for benchmarking against a more complex ensemble approach.

The project then extends beyond prediction by implementing a practical credit risk decision support system. For each new loan applicant, the trained models estimate the Probability of Default (PD), which is combined with Loss Given Default (LGD) and Exposure at Default (EAD) to calculate Expected Loss (EL) using the standard banking framework:

Expected Loss=Probability of Default×Loss Given Default×Exposure at Default

To support operational decision-making, the system incorporates an internal credit decision engine that assigns borrower risk categories and recommends lending actions such as Approve, Manual Review, or Decline based on predicted default risk. This transforms statistical model outputs into practical credit policy recommendations, bridging the gap between predictive analytics and real-world banking applications.

Overall, the project demonstrates how modern machine learning techniques can be integrated with established credit risk management methodologies to build an interpretable, robust, and production-oriented credit risk assessment system suitable for supporting lending decisions in financial institutions.

Cemre Kol

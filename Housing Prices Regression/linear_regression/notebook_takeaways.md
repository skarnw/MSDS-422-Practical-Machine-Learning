Module 2 Assignment 1

House Prices: Advanced Regression Techniques

Group 3: Ayoola Ogunbona, Cody Padlo, Soraya Karimi

MSDS 422: Practical Machine Learning  
July 7, 2025

**1.0 Introduction**

Building on Group 3’s earlier exploration of the Kaggle Housing Prices: Advanced Regression Techniques dataset, this follow-up extends the original analysis—focused on data quality, outliers, and correlation—into a deeper modeling phase. The objective was to validate initial findings, apply formal feature selection, and compare regression models through rigorous statistical testing and visualization.

**2.0 Data Reassessment and Preprocessing**

The original analysis highlighted strong predictors of SalePrice, particularly GrLivArea, GarageArea, 1stFlrSF, and YearRemodAdd. In this phase, we confirmed that all data points were retained and that square footage extremes did not distort downstream modeling. Scaling transformations helped normalize ranges, support residual diagnostics, and enable fair model comparisons without removing extreme values. To build a robust predictor set, we employed both F-regression for linear associations. These methods reinforced the relevance of earlier-identified variables and guided refinement of the final predictor set. Six features were finalized for modeling: TotalBsmtSF, GrLivArea, FullBath, GarageArea, OverallQual, and YearRemodAdd.

**3.0 Model Selection and Evaluation**

Seven models were compared using cross-validation metrics including R², Adjusted R², AIC (Aikake Information Criterion), and MSE (Mean Squared Error). Models 2 through 5 (various configurations of linear regression with and without scaling and forward selection) showed the most consistent R² values with negligible extreme outliers, indicating reliable explanatory power. Models 6 (Polynomial Regression) and 7 (GAM–Generalized Additive Model) had high median R² but suffered from extreme variability, likely due to parameter sensitivity or overfitting in specific folds. In terms of AIC, Models 3 and 4 (Standard and MinMax Scaled Linear Regression) had the narrowest spread, suggesting that normalization reduced instability due to scale differences. MSE results echoed these findings, with Models 2 through 5 tying for the lowest median MSE. These models demonstrated better generalization, while Models 6 and 7 had greater volatility.

Residual plots revealed non-constant variance in most models, typically in a funnel shape, undermining confidence in standard regression assumptions. The residual plots for Model 6 and 7 (after outlier removal) suggest homoscedasticity, resulting in better model reliability.

**4.0 Log Transformation and Final Model Choice**

To address heteroscedasticity, the Y variable (SalePrice) was log-transformed across Models 2 through 5. This transformation improved AIC and residual variance while minimizing the differences across scaling techniques. Post-transformation, residual plots lost their funnel shape, and the metrics for Models 2 through 4 converged. However, multicollinearity concerns led to the elimination of Models 2 and 5.

Ultimately, Model 4 (Linear Regression with MinMax Scaling and log-transformed Y) was selected as the final model due to its performance stability, lack of multicollinearity, and interpretability. Although Model 3 showed nearly identical metrics, the decision favored MinMax scaling for its visual clarity in earlier phases.

**5.0 Test Results, Kaggle Submission, and Recommendations for Future Work**

Predictions on the test dataset followed expected trends, though a single outlier raised concerns of overfitting. Overall, SalePrice variance in the test set was narrower than in training data. The model was submitted to Kaggle for evaluation, yielding an RMSE of 0.46312 and placing 4565 out of 4795 entries, at the time of submission. While modest, the score provides a baseline for future iterations.

To improve performance, several steps are recommended: tuning the parameters of Polynomial and GAM models; incorporating categorical variables via encoding; enhancing outlier and null handling through imputation or synthetic data techniques (e.g., SMOTE); and deeper statistical validation, including leverage and influence analysis. While linear regression remains interpretable and competitive, future approaches may benefit from tree-based models or ensemble methods for better generalization.

**6.0 References**

Montoya, Anna, and DataCanary. 2016. “House Prices - Advanced Regression Techniques.” Kaggle. 2016. <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>.

**7.0 Appendix A: Kaggle Results**

Username: Cody Padlo

Score: 0.46312

**8.0 Appendix B: Code Analysis**
Module 3 Assignment 1

House Prices: Advanced Regression Techniques

Group 3: Ayoola Ogunbona, Cody Padlo, Soraya Karimi

MSDS 422: Practical Machine Learning

July 13, 2025

**1.0 Introduction**

Continuing Group 3’s prior work on the _Kaggle Housing Prices_ dataset, this phase applies regression techniques to validate findings, engineer features, and comparing model performance using cross-validation.

**2.0 Cross-Validation and Evaluation Strategy**

All regression models were built using a consistent cross-validation framework. This approach ensured that model comparisons were fair and avoided overfitting to a single train-test split. Each model's RMSE was computed using validation predictions and aggregated across folds.

**3.0 Lasso Regression**

Lasso regression, implemented using LassoCV, applies L1 regularization to induce sparsity by shrinking some coefficients to zero. This simplifies interpretation and can act as an embedded feature selection method. The optimal alpha value we found was 1177.06, and the model achieved an R² of 0.7709 and a mean squared error (MSE) of about 144.45 million. Our analysis showed that OverallQual, 1stFlrSF, and 2ndFlrSF were the most important predictors. The coefficient path plot indicated that feature weights decreased with increased regularization, while the scatterplot of actual versus predicted prices confirmed a strong linear relationship, with slight underprediction for higher prices. Residuals were mostly centered around zero, with a few high-value outliers. The Lasso model balanced accuracy and interpretability and allowed us to create and export predictions for Kaggle evaluation, for a RMSE of 0.48638.

**4.0 Ridge Regression**

Ridge regression with L2 regularization was applied using a standardized pipeline, with RidgeCV identifying an optimal alpha of 1.42. The model retained all features, with 1stFlrSF and OverallQual showing the strongest influence on sale price. Other predictors like GarageArea, 2ndFlrSF, and TotalBsmtSF had moderate impact, while FullBath contributed minimally. Ridge produced a Kaggle RMSE of 0.17141 and an R^2 of 0.8510, indicating strong overall performance but slightly lower accuracy than ElasticNet. Residuals showed evidence of heteroskedasticity, though no log transformation was applied. The model remains robust and interpretable, with balanced coefficient shrinkage across predictors.

**5.0 ElasticNet Regression**

ElasticNet regression combines both L1 and L2 penalties, balancing sparsity and robustness. Hyperparameters were optimized using ElasticNetCV, tuning both alpha and L1_ratio. A ElasticNet pipeline was built to standardize chosen features X, and cross validation was employed to determine the best L1 ratio and alpha. From the cross validation, an L1 ratio of 0.5 was determined, indicating that penalty terms for both Lasso and Ridge were employed in equal proportions. Application of ElasticNet adjusted coefficients in a similar manner to Lasso, resulting in a negative coefficient for FullBath and the largest coefficient for OverallQual. Cross validation was also performed to determine metrics for comparison to other models, producing a lower RMSE than Ridge regression. Residual plots indicated heteroskedasticity, so the pipeline was rerun with a log transformation on y. R2 in log space, Adjusted R2, and AIC for the log-transformed pipeline were comparable to the metrics for the non-log-transformed ElasticNet pipeline. Finally, the log-transformed EN pipeline was applied to the test dataframe, and visual analysis of the predictions shows a similar distribution of predictions as the log-transformed scaled Model 4 previously attempted. The model produced a final Kaggle RMSE of 0.16795.

**6.0 Model Comparison and Kaggle Evaluation**

Lasso, Ridge, and ElasticNet regressions each offered unique strengths in modeling housing prices. Lasso emphasized interpretability by shrinking some coefficients to zero, achieving an R² of 0.7709 and Kaggle RMSE of 0.48638. Ridge retained all features with more uniform shrinkage, performing well (R² = 0.8510) and a lower Kaggle RMSE of 0.17141. ElasticNet balanced both penalties, outperforming Ridge slightly with a Kaggle RMSE of 0.16795, after applying log transformation to address heteroskedasticity.

**7.0 Conclusions**

ElasticNet emerged as the top performer, (rank 3685/4825 as of submission), slightly surpassing Ridge in predictive accuracy while maintaining model interpretability. Consistent cross-validation ensured fair evaluation, and log transformation improved robustness. Overall, regularized regression techniques proved effective for modeling housing prices, with trade-offs between sparsity, complexity, and predictive performance.

**8.0 References**

Montoya, Anna, and DataCanary. 2016. “House Prices - Advanced Regression Techniques.” Kaggle. 2016. <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>.

**9.0 Appendix A: Kaggle Results**

Username: Cody Padlo

Model Type: Lasso Regression

Score: 0.48638

Model Type: Ridge Regression

Score: 0.17141

Model Type: ElasticNet Regression

Score: 0.16795

**10.0 Appendix B: Code Analysis**
Module 1 Assignment 1

House Prices: Advanced Regression Techniques

Group 3: Ayoola Ogunbona, Cody Padlo, Soraya Karimi

MSDS 422: Practical Machine Learning  
June 29, 2025

**1.0 Introduction**

Group 3 analyzed the Kaggle Housing Prices: Advanced Regression dataset to prepare it for predictive modeling. The project focused on addressing data quality issues, identifying outliers, selecting and engineering features, and comparing numerical scaling methods. The collaborative work was conducted in Google Colab, with the code available in Appendix A.

**2.0 Data Exploration, Analysis, and Insights**

Initial exploration examined data structure, missing values, outliers, and distribution patterns. Correlation analysis identified four strong predictors of SalePrice: GrLivArea (0.71), GarageArea (0.62), 1stFlrSF (0.61), and YearRemodAdd (0.51). GrLivArea had the strongest relationship with SalePrice but also showed extreme values, suggesting outliers. GarageArea and 1stFlrSF, while informative, may introduce multicollinearity due to their overlap with overall living space. GarageArea also included zero values, indicating homes without garages. YearRemodAdd suggested a positive impact of remodeling on price, though its relationship may not be strictly linear.

Multiple regression validated these four features, returning very low p-values and confirming their statistical significance. Scatterplots reinforced their predictive value but also indicated that YearRemodAdd may benefit from further analysis. Most features in the dataset showed moderate positive correlations with SalePrice; negative correlations were rare, the lowest being –0.14 for OverallCond.

**3.0 Feature Engineering**

To enhance the dataset’s representational power, several new features were created. AgeOfHouse, the difference between year sold and year built, captured home age. Most homes were under 30 years old, though older homes may behave differently in the market. A binary variable, Remodeled, indicated whether a house had been updated since construction and showed a nearly even distribution across the dataset. HasPool identified homes with pools—just seven in the dataset—making it a rare but potentially high-impact feature. Finally, TotalSqFeet was developed by aggregating key square footage features to better represent livable space. While highly correlated with SalePrice, variability among high-end homes suggested that square footage alone does not fully explain price.

**4.0 Feature Scaling and Outlier Treatment**

To standardize numerical variables, both Min-Max Scaling and Standard Scaling were applied. Min-Max Scaling rescaled features to a 0–1 range, preserving distribution shape but compressing outliers. In contrast, Standard Scaling centered features around the mean and scaled by unit variance, making extreme values more visible. Some values exceeded +20 or fell below –4 standard deviations, confirming the presence of outliers. While Min-Max concealed these outliers, Standard Scaling exposed them, possibly affecting model performance. These differences underscore the importance of selecting appropriate scaling techniques and suggest that future iterations may benefit from more robust methods or direct outlier treatment.

**5.0 Conclusion**

This analysis of Ames housing prices identified several strong predictors of sale price, including GrLivArea, GarageArea, 1stFlrSF, and YearRemodAdd. Engineered features such as AgeOfHouse, Remodeled, HasPool, and TotalSqFeet added valuable context to the dataset. Scaling comparisons showed how outliers influence the modeling process and revealed trade-offs between normalization techniques. Together, these steps create a strong foundation for building effective regression models and improving our understanding of the factors that influence housing prices.

**6.0 References**

Montoya, Anna, and DataCanary. 2016. “House Prices - Advanced Regression Techniques.” Kaggle. 2016. <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>.

**7.0 Appendix A: Exploratory Data Analysis Code**
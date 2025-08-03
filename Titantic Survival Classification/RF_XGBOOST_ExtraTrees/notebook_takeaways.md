Module 5 Assignment 1

Titanic: Machine Learning Through Disaster

Group 3: Ayoola Ogunbona, Cody Padlo, Soraya Karimi

MSDS 422: Practical Machine Learning

July 27, 2025

**1.0 Introduction**

The Titanic dataset from Kaggle challenges participants to predict passenger survival using demographic and travel data. Building on our earlier models from Module 4, this phase focuses on tree-based ensemble methods, which are known for their strong classification performance on structured data. Three advanced models were implemented using cross-validation and hyperparameter tuning. Each model’s predictions were submitted to Kaggle for evaluation, with the goal of improving upon previous scores and identifying the most effective ensemble strategy.

**2.0 Random Forest**

Created a Random Forest Classifier as our first tree-based model to set a strong foundation for ensemble learning. Tuned four main settings—number of trees, maximum depth, criteria for splitting, and feature selection, using a grid search strategy with 12-fold cross-validation to improve its performance. The final model achieved a mean cross-validation accuracy of approximately 83%. This shows it can generalize well with little overfitting. The performance checks, including a confusion matrix and classification report, confirmed that the model accurately identified survivors and non-survivors, maintaining a good balance between precision and recall. Generated predictions from the tuned model on the test set and submitted them to Kaggle, setting a benchmark for our future ensemble models.

**3.0 XGBOOST, Gradient Boosted Tree**

A gradient boosting tree algorithm, XGBOOST, was trained and tested on the Titanic dataset. XGBOOST is a gradient boosting algorithm based on C++ for faster computation, and has built-in functions for feature importance and parallelism. Cross-validated tuning was employed only over a subset of parameters, specifically n_estimators, max_depth, and learning rate, across 320 total fits. The best model resulted in hyperparameter values of 0.1, 2, and 125, respectively. Feature importance plots indicated that Fare and Age were most useful for key decision-making by the model. Cross-validation was optimized for accuracy, yielding the best model with an accuracy of 0.83. Another cross-validation effort was conducted using the best model for metric evaluation, yielding a mean F1 score of 0.738 and a mean ROC of 0.790, which provides some confidence in the best model’s predictive ability. The best models were then used to produce predictions against the test set and sent to Kaggle for evaluation.

**4.0 Extra Trees**

An Extra Trees classifier was built to explore tree-based ensemble methods using the Titanic dataset. A full preprocessing pipeline was developed that included mean imputation for numeric features. Hyperparameter tuning was performed using a 12-fold cross-validation grid search over 5 n_estimators, 3 max_features, 5 max_depth, and 2 criterion. The optimal model used 160 estimators, sqrt as the max feature strategy, and entropy as the splitting criterion, with a maximum tree depth of 8. Cross-validation produced an accuracy of approximately 0.8301. Evaluation on the train dataset revealed balanced precision and recall, with a relatively low false positive rate and strong specificity. ROC and precision-recall curves confirmed good separation between survival outcomes. The model was then used to generate survival predictions on the test set and submitted to Kaggle for evaluation.

**5.0 Model Comparison**

All three ensemble models—Random Forest, XGBoost, and Extra Trees—achieved similar cross-validation accuracy around 83%, indicating strong predictive performance. The Random Forest model demonstrated solid generalization with minimal overfitting and a balanced classification report, making it a reliable baseline. XGBoost, while more computationally intensive, offered additional insight via feature importance and scored highest in ROC and F1 metrics, reflecting robust discrimination and balance. Extra Trees, with its randomized splits, also performed comparably well, offering slightly better specificity. Overall, XGBoost showed a slight edge in predictive depth, while Random Forest and Extra Trees provided strong, stable alternatives.

**6.0 Conclusions**

All three ensemble models performed similarly, with XGBoost showing a slight edge in predictive metrics. Random Forest and Extra Trees offered strong, stable alternatives with balanced performance. These findings highlight the effectiveness of tree-based methods for structured data and reinforce the value of model comparison in predictive tasks.

**7.0 References**

Cukierski, Will. 2012. “Titanic - Machine Learning from Disaster.” Kaggle. 2012. <https://www.kaggle.com/c/titanic/overview>.

**8.0 Appendix A: Kaggle Results**

Username: Cody Padlo

Model: Random Trees

Score: 0.77751

Model: Gradient Boosted Trees

Score: 0.75598

Model: Extra Trees

Score: 0.77990


**9.0 Appendix B: Code Analysis**
Module 4 Assignment 1

Titanic: Machine Learning Through Disaster

Group 3: Ayoola Ogunbona, Cody Padlo, Soraya Karimi

MSDS 422: Practical Machine Learning

July 20, 2025

**1.0 Introduction**

The “Titanic: Learning From Disaster” dataset from Kaggle consists of ten variables and an indicator representing if an individual survived the 1912 Titanic sinking. Group 3 performed exploratory data analysis, feature selection, and generated a classification model to predict passenger survival.

**2.0 LDA/QDA Regression**

A linear discrimination analysis pipeline was built and trained on numeric, standard-scaled features. Imputation employing a “median” fill was included in the preprocessing steps. 12-fold cross-validation was used for metrics evaluation, resulting in an accuracy score of 0.7924. When the model was tested against a train/test split of size 0.2, F1 scores across classes were above 0.71, giving some confidence in the precision and recall for the model. The confusion matrix provided context on the LDA model, where a relatively high “true negative” count and low “false positive” count calculated to a specificity measure of 89%. In contrast, the Quadratic Discriminant Analysis (QDA) achieved lower performance with a score of 0.74162, despite a well-structured pipeline and clean validation strategy. The confusion matrix and classification report revealed moderate precision and recall but showed that the model struggled to generalize well on unseen data. Predictions from the LDA and QDA model, when tested against the test set, were sent to Kaggle for comparison against other submissions

**3.0 K-Nearest Neighbor (KNN)**

A KNN classifier was trained using only the numeric features from the dataset. A preprocessing pipeline was constructed to handle missing values with median imputation and standardize the features before applying the KNN algorithm. Hyperparameter tuning was performed using a 12-fold cross-validation grid search to identify the optimal number of neighbors, with accuracy as the scoring metric. The best model achieved a cross-validation accuracy of approximately 0.8239 and selected 7 neighbors as the optimal setting. Evaluation on a hold-out validation set yielded a validation accuracy of 0.8380, with strong separation observed in the ROC and precision-recall curves. The confusion matrix revealed balanced performance across both survival outcomes. Finally, the model was applied to the test set, generating survival predictions.

**4.0 Logistic Regression**

A logistic regression model was trained on eight selected features. Hyperparameter tuning using cross-validation was employed to adjust regularization, and ultimately, an L1-regularized logistic regression model with the best F1 score was chosen. A Box-Tidwell test for confirming the assumption of linearity of the variables and the log-odds of the outcome concluded that two variables, Age and Sibsq, had a potentially non-linear relationship with the log-odds outcome. To adjust for this, a spline was included in the regression pipeline to transform the two identified variables. Numeric variables were standard-scaled, categorical variables were one-hot encoded, and all variables underwent median imputation. Cross-validation produced an F1 score of 0.761 and an AUC-ROC of 0.858 for the best model. Final predictions against the test set produced a similar distribution of 'Died' and 'Survived' compared to the training set, and were submitted to Kaggle.

**5.0 Model Comparison**

All four models were analyzed for comparison: LDA, QDA, KNN and logistic regression. Each model has different strengths and assumptions that affect their performances. Logistic regression performed the best with a Kaggle score of 0.76315. Logistic regression’s relative simplicity lends to better explainability compared to models like KNN, and the implemented pipeline adjusted for multicollinearity and linear assumptions. K-Nearest Neighbors achieved a score of 0.7076, and although the score indicates adequate performance, it further requires careful preparation and fine-tuning. Relevant KNN hyperparameter tuning includes choosing the optimal neighbor quantity, distances, and weights. LDA assumes that data features are normally distributed and have equal class variances. Although LDA performed reasonably well, its linear approach didn't fully capture the data’s complexity. QDA, which allows for more complex shapes in data, had the lowest score of 0.74162. Its performance suffered potentially due to overfit of the data, in combination with an imbalanced dataset and small sample size.

**6.0 Conclusions**

While the scores for all four models were relatively similar, the regularized logistic regression model won out with the highest score at 0.76315. All four models employed cross-validation for metrics evaluation and standard scaling, though differences in pipeline build and model type were evident in final scoring. Future attempts to improve the results of model generation may include testing SVM, tree-based modelling, neural networks, or other advanced classification techniques.

**7.0 References**

Cukierski, Will. 2012. “Titanic - Machine Learning from Disaster.” Kaggle. 2012. <https://www.kaggle.com/c/titanic/overview>.

**8.0 Appendix A: Kaggle Results**

Username: Cody Padlo

Model: LDA

Score: 0.75837

Model: QDA

Score: 0.74162

Model: K-Nearest Neighbor

Score: 0.76076

Model: Logistic Regression

Score: 0.76315


**9.0 Appendix B: Code Analysis**
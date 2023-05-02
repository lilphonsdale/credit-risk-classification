# credit-risk-classification

## Overview of the Analysis

In this analysis we set out to accurately classify loans as either 'Healthy' or 'High-Risk'.

We start with a dataset with the following columns:

'loan_size': $ amount of the loan
'interest_rate': interest charged on the loan
'borrower_income': the borrower's income
'debt_to_income': the ratio of total_debt /  borrower_income
'num_of_accounts': the borrower's number of accounts
'derogatory_marks': the number of derogatory marks against that borrower's credit
'total_debt': the total amount of debt carried by the borrower
'loan_status': a binary where 0 = Healthy and 1 = High-Risk. This will be our target. 

We split the data set so that loan_status is our labels and the rest are our features. Then we split our labels and features into testing and training data.

We instantiate a Logistic Regression model, from scikit-learn's linear_model module. This model is fitted using our training data and then used to make predictions. 

After evaluating the performance of the model with the original training data, we resample using imbalanced-learn's RandomOverSampler. Once again we fit the Logistic Regression model, and make predictions with this resampled data.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Logistic Regression using the original data:
  * Accuracy: .99
  * Precision: 
    * Healthy Loan - 1
    * High-Risk Loan - 0.87
  * Recall
   * Healthy Loan - 1
   * High-Risk Loan - 0.89

* Logistic Regression using the resampled data:
  * Accuracy: 1
  * Precision: 
    * Healthy Loan - 1
    * High-Risk Loan - 0.87
  * Recall
   * Healthy Loan - 1
   * High-Risk Loan - 1

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

The most important difference between the two models is the Recall score for High-Risk loans. In this analysis, we set out to positively identify High-Risk loans, which means that our primary success metric is going to be Recall; we need to make sure we do not miss any High-Risk loans, because defaults are losses. 

In the first model, the unbalanced training data prevented the model from accurately identifying High-Risk loans. This is to be expected because we knew that there were a lot more Healthy loans in the training data than High-Risk loans. We can see that the model had an easy time identifying Healthy Loans, and the high overall Accuracy score of 99% actually masks the relatively poor performance predicting High-Risk loans.

Imbalanced-learn's RandomOverSampler is described as an "Object to over-sample the minority class". This is what we need to do to improve the model's performance predicting High-Risk loans. Once we used RandomOverSampler and refit the Logistic Regression model with the resampled training data, we see a dramatic improvement in the model's ability to predict High-Risk loans. In fact, the Recall score is 1, which means that all of the High-Risk loans in the dataset were classified as such. There are no false negatives. 

We could go further by looking at the cost incurred by false positives, but it's safe to assume that it is less than the losses saved by eliminating false negatives. 

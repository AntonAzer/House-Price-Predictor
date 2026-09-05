# House Prices Prediction: XGBoost vs. Random Forest

## Project Overview
This project applies machine learning techniques to predict residential home prices using the Kaggle "House Prices - Advanced Regression Techniques" dataset. The target variable (`SalePrice`) is transformed using $\log(1 + x)$ to normalize target distribution and minimize root-mean-squared error (RMSE) on log-transformed values.

## Performance Evaluation
Models are evaluated on the Kaggle platform using Root Mean Squared Logarithmic Error (RMSLE). Lower values represent better model performance.

| Model Algorithm | Submission File | Kaggle RMSLE Score |
| :--- | :--- | :--- |
| **XGBoost Regressor** | `xgbmodel_houseproblem.csv` | **0.14266** |
| **Random Forest Regressor** | `submission (1).csv` | **0.15123** |

The XGBoost model achieved a significant improvement over Random Forest, reducing evaluation error from **0.15123** down to **0.14266**.

## Architectural Differences: XGBoost vs. Random Forest

### 1. Ensemble Paradigm
* **Random Forest (Bagging):** Constructs a set of decision trees in parallel independently. Each tree is trained on a random bootstrap sample of the dataset. The final output is an average of predictions across all individual trees.
* **XGBoost (Gradient Boosting):** Builds decision trees sequentially in an additive manner. Each new tree is explicitly trained to predict the residual errors (gradients) of the ensemble of trees constructed before it.

### 2. Error Reduction Focus
* **Random Forest:** Primary objective is **variance reduction**. Averaging independent overfitted trees smooths out overall predictions, but cannot lower structural bias inherent to weak tree representations.
* **XGBoost:** Reduces both **bias and variance**. By using gradient descent along the loss function, boosting iteratively focuses capacity on hard-to-predict observations.

### 3. Regularization Mechanisms
* **Random Forest:** Relies on hyperparameter constraints such as `max_depth`, `min_samples_split`, and sub-sampling features to control model variance. It does not include direct penalty terms in its loss function.
* **XGBoost:** Features explicit **L1 (Lasso)** and **L2 (Ridge)** regularization parameters ($\alpha$ and $\lambda$) within its loss function objective. This controls model complexity by penalizing high leaf-weights directly.

## Reasons for XGBoost Superior Performance on This Dataset

1. **Complex Feature Interactions:** Real estate prices depend on non-linear combinations of features (e.g., total square footage interacted with neighborhood tier). Sequential gradient boosting learns subtle localized sub-patterns that independent parallel trees average out.
2. **Handling Sparse High-Dimensional Data:** One-hot encoding produces a sparse feature space(I will explain what is the sparse data at the end). XGBoost uses a sparsity-aware split finding algorithm that determines optimal default split directions for zero/missing values, whereas Random Forest splits randomly selected feature subsets regardless of sparsity density.
3. **Continuous Metric Optimization:** House price regression involves continuous predictions with high variance. The second-order Taylor expansion used in XGBoost's loss function optimizes continuous targets with higher gradient accuracy than the basic variance-reduction split strategy used in Random Forest.


 ## Both Models Results
   
   <img width="1386" height="293" alt="image" src="https://github.com/user-attachments/assets/cc914795-6704-4b40-9a23-e83709beaa47" />

   what in the NB file is the XG. model , the only difference to try the Regression model to edit this line to

   ```model = RandomForestRegressor(n_estimators=100, random_state=42) ```
   
   And I'm already import that.
   
   <img width="1290" height="553" alt="image" src="https://github.com/user-attachments/assets/ef0702a8-15ea-4995-9a18-463f59ddd01e" />

   

   
**Competition [Link on Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)**

Sparse-Data (mentioned above): while converting the text-form data like name of the street into numbering data to the model it becomes a lot of columns that contains zeros (sparse data), and ordinary Regression Tree deal with it randomly
,but XG. uses better algo in this case.

## More
For more of Kaggle competition take a look at real-world actual challenge and how I model it with rank 131 in the leaderboard [LLM Classification Finetuning](https://www.kaggle.com/code/antonazer/llm-prob?scriptVersionId=347357348).


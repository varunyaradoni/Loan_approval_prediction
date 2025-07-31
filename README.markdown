# Loan Approval Prediction Project

## Overview
This project aims to predict loan approval outcomes using various machine learning models. The dataset used is `LoanApprovalPrediction.csv`, which contains features related to loan applicants such as gender, marital status, income, and credit history. The notebook `Loan_approval_prediction.ipynb` implements data preprocessing, exploratory data analysis, and model training to predict whether a loan application will be approved (`Y`) or not (`N`).

## Dataset
The dataset includes the following columns:
- **Loan_ID**: Unique identifier for each loan application
- **Gender**: Applicant's gender (Male/Female)
- **Married**: Marital status (Yes/No)
- **Dependents**: Number of dependents (0, 1, 2, 3)
- **Education**: Education level (Graduate/Not Graduate)
- **Self_Employed**: Self-employment status (Yes/No)
- **ApplicantIncome**: Applicant's income
- **CoapplicantIncome**: Co-applicant's income
- **LoanAmount**: Loan amount requested
- **Loan_Amount_Term**: Loan term in months
- **Credit_History**: Credit history (0 or 1)
- **Property_Area**: Property location (Urban/Semiurban/Rural)
- **Loan_Status**: Loan approval status (Y/N)

## Requirements
To run the notebook, you need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Notebook Structure
The `Loan_approval_prediction.ipynb` notebook is structured as follows:
1. **Import Libraries**: Import necessary libraries for data manipulation, visualization, and modeling.
2. **Load Data**: Load the dataset from `LoanApprovalPrediction.csv`. Note: The original code uses Google Colab's drive mounting; modify the file path if running locally.
3. **Data Preprocessing**:
   - Handle missing values by filling them with the mean of the respective column.
   - Drop the `Loan_Status` column to create feature set `X` and target `Y`.
4. **Exploratory Data Analysis**:
   - Visualize the relationship between `Gender`, `Married`, and `Loan_Status` using a bar plot.
5. **Model Training**:
   - Split data into training (60%) and testing (40%) sets.
   - Train four models: Random Forest Classifier, K-Nearest Neighbors, Support Vector Classifier, and Logistic Regression.
   - Evaluate model performance on both training and testing sets using accuracy scores.
6. **Model Evaluation**:
   - Print accuracy scores for each model on the training and testing datasets.

## Key Findings
- **Training Accuracy**:
  - RandomForestClassifier: 98.04%
  - KNeighborsClassifier: 78.49%
  - SVC: 68.72%
  - LogisticRegression: 79.61%
- **Testing Accuracy**:
  - RandomForestClassifier: 82.50%
  - KNeighborsClassifier: 63.75%
  - SVC: 69.17%
  - LogisticRegression: 80.83%
- Random Forest Classifier performs best on both training and testing sets, though it may indicate overfitting due to the high training accuracy.
- Logistic Regression also shows strong performance with a good balance between training and testing accuracy.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Prepare the Dataset**:
   - Ensure `LoanApprovalPrediction.csv` is in the project directory or update the file path in the notebook.
3. **Run the Notebook**:
   - Open `Loan_approval_prediction.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to load data, preprocess, visualize, and train models.
   - If not using Google Colab, replace the drive mounting code with:
     ```python
     data = pd.read_csv("LoanApprovalPrediction.csv")
     ```
4. **Modify and Experiment**:
   - Adjust model hyperparameters (e.g., `n_estimators` for Random Forest, `n_neighbors` for KNN) to improve performance.
   - Add feature engineering steps, such as encoding categorical variables or scaling numerical features, to address the Logistic Regression convergence warning.

## Notes
- The Logistic Regression model raises a convergence warning, suggesting the need for data scaling. Consider adding a preprocessing step like:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  ```
- The dataset assumes no missing values after mean imputation, but further validation of this approach is recommended.
- For production use, consider encoding categorical variables (e.g., `Gender`, `Married`, `Education`, `Property_Area`) using techniques like one-hot encoding or label encoding.

## Future Improvements
- Implement feature encoding for categorical variables to improve model compatibility.
- Apply feature scaling to address the Logistic Regression convergence issue.
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Add cross-validation to ensure robust model evaluation.
- Explore additional models, such as Gradient Boosting or XGBoost, for better performance.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if applicable).
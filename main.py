import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

# Adjusting display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Loading dataset
vehicle_sales_count = pd.read_csv('vehicle_sales_count.csv')

# Renaming attributes
column_names = {'Year ': 'Year', 'Month ': 'Month', 'Total Sales New': 'Total Sales New', 'Total Sales Used':
    'Total Sales Used'}
vehicle_sales_count = vehicle_sales_count.rename(columns=column_names)


# Visualizing numerical distribution
fig, axes = plt.subplots(2, 2)
plt.suptitle('Distribution of Numerical Attributes')
sns.histplot(vehicle_sales_count['New'], ax=axes[0, 0], kde=True, bins=10)
axes[0, 0] = sns.histplot(vehicle_sales_count['Used'], ax=axes[1, 0], kde=True, bins=10)
axes[0, 1] = sns.histplot(vehicle_sales_count['Total Sales New'], ax=axes[0, 1], kde=True, bins=10)
axes[1, 1] = sns.histplot(vehicle_sales_count['Total Sales Used'], ax=axes[1, 1], kde=True, bins=10)
plt.tight_layout()


# Boxplot visualization for outlier detection and range
numerical_attributes = vehicle_sales_count.columns[2:].to_list()
range_numerical = {}
for index in numerical_attributes:
    range_numerical[index] = (int(vehicle_sales_count[index].max() - vehicle_sales_count[index].min()))
range_numerical = {k: v for k, v in sorted(range_numerical.items(), key=lambda item: item[1])}

fig2, axes2 = plt.subplots(2, 2)
plt.suptitle('Outlier Detection')
sns.boxplot(vehicle_sales_count['New'], ax=axes2[0, 0], orient='h')
sns.boxplot(vehicle_sales_count['Used'], ax=axes2[1, 0], orient='h')
sns.boxplot(vehicle_sales_count['Total Sales New'], ax=axes2[0, 1], orient='h')
sns.boxplot(vehicle_sales_count['Total Sales Used'], ax=axes2[1, 1], orient='h')
plt.tight_layout()

# Correlation analysis
correlation_analysis = vehicle_sales_count[numerical_attributes].corr()

# VIF to understand multicollinearity
X = sm.add_constant(vehicle_sales_count[numerical_attributes])
VIF = pd.DataFrame()
VIF['Attributes'] = X.columns
VIF['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


# Ordinal encoder
Month_Encoder = OrdinalEncoder(categories=[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])
vehicle_sales_count['Months'] = Month_Encoder.fit_transform(vehicle_sales_count[['Month']])

# min-max encoder
MM_Scaler = MinMaxScaler()
vehicle_sales_count['New_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['New']])
vehicle_sales_count['Used_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['Used']])
vehicle_sales_count['Total_New_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['Total Sales New']])
vehicle_sales_count['Total_Used_MinMax'] = MM_Scaler.fit_transform(vehicle_sales_count[['Total Sales Used']])

joblib.dump(Month_Encoder, 'month_encoder.pkl')
joblib.dump(MM_Scaler, 'min_max_scaler.pkl')

# Predicting 2023
vehicle_sales_in_2023 = vehicle_sales_count.loc[vehicle_sales_count['Year'] == 2023].reset_index()
vehicle_sales_in_2023.to_csv('vehicle_sales_in_2023.csv', index=False)
# creating new dataframe for modeling
cleaned_df = vehicle_sales_count[
    ['Year', 'Months', 'New_MinMax', 'Used_MinMax', 'Total_New_MinMax', 'Total_Used_MinMax']]
cleaned_df = cleaned_df.loc[cleaned_df['Year'] != 2023].reset_index(drop=True)
X = cleaned_df[['Year', 'Months', 'New_MinMax', 'Used_MinMax', 'Total_Used_MinMax']]
Y = cleaned_df[['Total_New_MinMax']]

# Training_Testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Linear Regression
regression_1 = LinearRegression().fit(x_train, y_train)
y_predict_regression_1 = regression_1.predict(x_test)
regression_1_mse = mean_squared_error(y_true=y_test, y_pred=y_predict_regression_1)
regression_1_r2 = r2_score(y_true=y_test, y_pred=y_predict_regression_1)

# Decision Tree Regressor #1
decision_1 = DecisionTreeRegressor()
decision_1 = decision_1.fit(x_train, y_train)
y_predict_decision_1 = decision_1.predict(x_test)
decision_1_mse = mean_squared_error(y_true=y_test, y_pred=y_predict_decision_1)
decision_1_r2 = r2_score(y_true=y_test, y_pred=y_predict_decision_1)

# Decision Tree Regressor #2
decision_2 = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth = 10,
    min_samples_split= 50,
    min_samples_leaf = 40,
    random_state=42,
)
decision_2 = decision_2.fit(x_train, y_train)
y_predict_decision_2 = decision_2.predict(x_test)
decision_2_mse = mean_squared_error(y_true=y_test, y_pred=y_predict_decision_2)
decision_2_r2 = r2_score(y_true=y_test, y_pred=y_predict_decision_2)

# Decision Tree Regressor #3
decision_3 = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth = 10,
    min_samples_split= 50,
    min_samples_leaf = 65,
    random_state=42,
)
decision_3 = decision_3.fit(x_train, y_train)
y_predict_decision_3 = decision_3.predict(x_test)
decision_3_mse = mean_squared_error(y_true=y_test, y_pred=y_predict_decision_3)
decision_3_r2 = r2_score(y_true=y_test, y_pred=y_predict_decision_3)

# Decision Tree Regressor #4
decision_4 = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth = 15,
    min_samples_split= 10,
    min_samples_leaf = 10,
    random_state=42,
)
decision_4 = decision_4.fit(x_train, y_train)
y_predict_decision_4 = decision_4.predict(x_test)
decision_4_mse = mean_squared_error(y_true=y_test, y_pred=y_predict_decision_4)
decision_4_r2 = r2_score(y_true=y_test, y_pred=y_predict_decision_4)

# Binning target variable for binary classification models
threshold = 0.5
y_train_binned = (y_train >= threshold).astype(int)
y_test_binned = (y_test >= threshold).astype(int)

# Logistic regression 1
logistic_regression_1 = LogisticRegression()
logistic_regression_1 = logistic_regression_1.fit(x_train, y_train_binned)
y_predict_logistic_regression_1 = logistic_regression_1.predict(x_test)
logistic_regression_acc = accuracy_score(y_true=y_test_binned,y_pred=y_predict_logistic_regression_1)

# Logistic regression 2
logistic_regression_2 = LogisticRegression(
    penalty = 'l1', max_iter=5000, class_weight='balanced', C=0.8, solver='liblinear')
logistic_regression_2 = logistic_regression_2.fit(x_train, y_train_binned)
y_predict_logistic_regression_2 = logistic_regression_2.predict(x_test)
logistic_regression_acc_2 = accuracy_score(y_true=y_test_binned,y_pred=y_predict_logistic_regression_2)

# Decision tree classifier 1
decision_classifier_1 = DecisionTreeClassifier(
    max_depth = 10,
    random_state=42,
    min_samples_leaf=10,
    min_samples_split=15
)
decision_classifier_1 = decision_classifier_1.fit(x_train, y_train_binned)
y_predict_decision_classifier_1 = decision_classifier_1.predict(x_test)
decision_classifier_acc_1 = accuracy_score(y_true=y_test_binned,y_pred=y_predict_decision_classifier_1)



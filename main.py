import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")
t_df = train_df.copy()
t_test = test_df.copy() 

# delete unused column
t_df = t_df.drop(columns=["Loan_ID"])
t_test = t_test.drop(columns=["Loan_ID"])

# Independent Variables
X = t_df.drop("Loan_Status", axis=1)
# Dependent Variable
y = t_df["Loan_Status"].map({"Y": 1, "N": 0})

# y and x in test
y_test = t_test["Loan_Status"].map({"Y": 1, "N": 0})
x_test = t_test.drop("Loan_Status", axis=1)

# divided data
x_train, x_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
x_train = x_train.copy()
x_val = x_val.copy()

# TRAIN PREPROCESSING
print(x_train.duplicated().sum())
print(x_train.isnull().sum())

# Replace null values with the mean
mean_term = x_train["Loan_Amount_Term"].mean()
x_train["Loan_Amount_Term"] = x_train["Loan_Amount_Term"].fillna(mean_term)
x_val["Loan_Amount_Term"] = x_val["Loan_Amount_Term"].fillna(mean_term)

mean_loan = x_train["LoanAmount"].mean()
x_train["LoanAmount"] = x_train["LoanAmount"].fillna(mean_loan)
x_val["LoanAmount"] = x_val["LoanAmount"].fillna(mean_loan)
x_test["LoanAmount"] = x_test["LoanAmount"].fillna(mean_loan)

# Replace null values with the mode
mode_vals = {}
for col in ["Self_Employed", "Gender", "Dependents", "Credit_History"]:
    mode_vals[col] = x_train[col].mode().iloc[0]
    x_train[col] = x_train[col].fillna(mode_vals[col])
    x_val[col] = x_val[col].fillna(mode_vals[col])

# to cap and delete outliers
cols_dif_big = ["CoapplicantIncome", "ApplicantIncome"]
cols_dif_small = ["Loan_Amount_Term", "LoanAmount"]

# cap
bounds_small = {}
for col in cols_dif_small:
    Q1 = x_train[col].quantile(0.25)
    Q3 = x_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    bounds_small[col] = (lower, upper)
    x_train[col] = x_train[col].clip(lower, upper)
    x_val[col] = x_val[col].clip(lower, upper)

# delete
mask = pd.Series(True, index=x_train.index)
for col in cols_dif_big:
    Q1 = x_train[col].quantile(0.25)
    Q3 = x_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask &= (x_train[col] >= lower) & (x_train[col] <= upper)

x_train = x_train[mask]
y_train = y_train[mask]

# Replace null values in test with the mean of training
x_test["Loan_Amount_Term"] = x_test["Loan_Amount_Term"].fillna(mean_term)

# Replace null values in test with the mode of training
for col in ["Self_Employed", "Gender", "Dependents", "Credit_History"]:
    x_test[col] = x_test[col].fillna(mode_vals[col])

# to make text to numbers
for df in [x_train, x_val, x_test]:
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
    df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
    df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})

# vectors
x_train = pd.get_dummies(x_train, columns=["Property_Area", "Dependents"], drop_first=True)
x_val = pd.get_dummies(x_val, columns=["Property_Area", "Dependents"], drop_first=True)
x_test = pd.get_dummies(x_test, columns=["Property_Area", "Dependents"], drop_first=True)

# make number of columns in training = test
x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

# check
print(x_train.isnull().sum())
print(x_test.isnull().sum())

# scale features to mean=0, std=1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

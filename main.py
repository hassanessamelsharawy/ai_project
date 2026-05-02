import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score

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

#Train model
log_results = []
c=np.logspace(-4, 2, 30)
log_solver=['liblinear','lbfgs']

best_c_log=None
best_log_solver=None
best_score_log=-1

'''for j in log_solver :
   for i in c:
       log_model=LogisticRegression(solver=j,C=i,random_state=0)
       log_model.fit(x_train, y_train)
        # Evaluate
       log_score = log_model.score(x_val, y_val)
       log_results.append({
           "C": i,
           "solver": j,
           "accuracy": log_score
       })
        # Update best Logistic
       if log_score > best_score_log:
           best_score_log = log_score
           best_c_log = i
           best_log_solver=j
'''

for j in log_solver:
    for i in c:
        log_model = LogisticRegression(solver=j, C=i, random_state=0)
        log_model.fit(x_train, y_train)
        y_pred = log_model.predict(x_val)
        log_score = f1_score(y_val, y_pred)
        log_results.append({
            "C": i,
            "solver": j,
            "f1": log_score
        })
        if log_score > best_score_log:
            best_score_log = log_score
            best_c_log = i
            best_log_solver = j


log_model=LogisticRegression(solver=best_log_solver,C=best_c_log,random_state=0)
log_model.fit(x_train, y_train)
log_df = pd.DataFrame(log_results)

SVM_results = []
svm_kernel=['linear','poly','rbf']
best_c_svm=None
best_svm_kernel=None
best_score_svm=-1

for j in svm_kernel :
   for i in c:
       svm_model = SVC(C=i, kernel=j, gamma='scale')
       svm_model.fit(x_train, y_train)
       y_pred = svm_model.predict(x_val)
       # Evaluate
       svm_score = f1_score(y_val, y_pred)
       SVM_results.append({
           "C": i,
           "Kernel": j,
           "f1": svm_score
       })
        # Update best SVM
       if svm_score > best_score_svm:
           best_score_svm = svm_score
           best_c_svm = i
           best_svm_kernel=j

svm_model=SVC(C=best_c_svm, kernel=best_svm_kernel, gamma='scale')
svm_model.fit(x_train, y_train)
SVM_df = pd.DataFrame(SVM_results)


DT_results = []
max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, None]
min_samples_split= [2, 5, 10]
best_min_samples_split=None
best_max_depth = None
best_score_DT = -1
for j in min_samples_split:
    for i in max_depth_values:
        DT_model = DecisionTreeClassifier(criterion='gini', max_depth=i, min_samples_split=j,random_state=0)
        DT_model.fit(x_train,y_train)
        y_pred = DT_model.predict(x_val)
        DT_score = f1_score(y_val, y_pred)
        DT_results.append({
            "max_depth": i,
            "min_samples_split": j,
            "f1": DT_score
        })
        if(DT_score > best_score_DT):
            best_score_DT = DT_score
            best_max_depth = i
            best_min_samples_split=j

DT_model = DecisionTreeClassifier(criterion='gini', max_depth=best_max_depth, min_samples_split=best_min_samples_split, random_state=0)
DT_model.fit(x_train,y_train)
DT_df = pd.DataFrame(DT_results)


# log_params = {
#     "C": np.logspace(-4, 2, 30),
#     "solver": ["liblinear", "lbfgs"]
# }
#
# log_grid = GridSearchCV(
#     LogisticRegression(random_state=0),
#     param_grid=log_params,
#     cv=5,
#     scoring="accuracy"
# )
# log_grid.fit(x_train, y_train)
# log_model=log_grid.best_estimator_
#
#
# svm_params = {
#     "C": np.logspace(-4, 2, 20),
#     "kernel": ["linear", 'poly',"rbf"]
# }
#
# svm_grid = GridSearchCV(
#     SVC(gamma="scale"),
#     param_grid=svm_params,
#     cv=5,
#     scoring="accuracy"
# )
# svm_grid.fit(x_train, y_train)
# svm_model=svm_grid.best_estimator_
#
# dt_params = {
#     "max_depth": [1, 2, 3, 5, 10, 15, None],
#     "min_samples_split": [2, 5, 10]
# }
#
# DT_grid = GridSearchCV(
#     DecisionTreeClassifier(random_state=0),
#     param_grid=dt_params,
#     cv=5,
#     scoring="accuracy"
# )
#
# DT_grid.fit(x_train, y_train)
# DT_model=DT_grid.best_estimator_

# Report & Accuracy

#logistic Regression
print('Logistic Regression')
print('Accuracy: '+ str(log_model.score(x_test, y_test)*100)+'%')
yp_log=log_model.predict(x_test)
print("F1_Score:"+str(f1_score(y_test, yp_log)*100)+"%")
print("Precision:"+str(precision_score(y_test, yp_log)*100)+"%")
print("Recall:"+str(recall_score(y_test, yp_log)*100)+"%")
print('Confusion Matrix :')
print(confusion_matrix(y_test, yp_log))
print('-------------------------------------------------------------')


#SVM
print('SVM')
print('Accuracy: '+ str(svm_model.score(x_test, y_test)*100)+'%')
yp_SVM=svm_model.predict(x_test)
print("F1_Score:"+str(f1_score(y_test, yp_SVM)*100)+"%")
print("Precision:"+str(precision_score(y_test, yp_SVM)*100)+"%")
print("Recall:"+str(recall_score(y_test, yp_SVM)*100)+"%")
print('Confusion Matrix :')
print(confusion_matrix(y_test, yp_SVM))
print('-------------------------------------------------------------')

#Decision Tree
print('Decision Tree ')
print('Accuracy: '+ str(DT_model.score(x_test, y_test)*100)+'%')
yp_DT=DT_model.predict(x_test)
print("F1_Score:"+str(f1_score(y_test, yp_DT)*100)+"%")
print("Precision:"+str(precision_score(y_test, yp_DT)*100)+"%")
print("Recall:"+str(recall_score(y_test, yp_DT)*100)+"%")
print('Confusion Matrix :')
print(confusion_matrix(y_test, yp_DT))
print('-------------------------------------------------------------')
# print(log_df)
# print('-------------------------------------------------------------')
# print(SVM_df)
# print('-------------------------------------------------------------')
# print(DT_df)

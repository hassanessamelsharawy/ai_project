import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# ============================================================
# VISUALIZATIONS (on raw data before preprocessing)
# ============================================================

# 1. Distribution of Loan Status
plt.figure(figsize=(6, 4))
train_df["Loan_Status"].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("loan_status_distribution.png")
plt.show()

# 2. Distribution plots for numerical features
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(train_df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
plt.suptitle("Numerical Features Distribution", fontsize=14)
plt.tight_layout()
plt.savefig("numerical_distributions.png")
plt.show()

# 3. Categorical features vs Loan Status
cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents", "Credit_History"]
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(train_df[col], train_df["Loan_Status"])
    ct.plot(kind="bar", ax=axes[i], color=["salmon", "steelblue"])
    axes[i].set_title(f"{col} vs Loan Status")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Categorical Features vs Loan Status", fontsize=14)
plt.tight_layout()
plt.savefig("categorical_vs_loan_status.png")
plt.show()

# 4. Correlation Heatmap
corr_df = train_df.copy()
corr_df["Loan_Status"] = corr_df["Loan_Status"].map({"Y": 1, "N": 0})
corr_df["Gender"] = corr_df["Gender"].map({"Male": 1, "Female": 0})
corr_df["Married"] = corr_df["Married"].map({"Yes": 1, "No": 0})
corr_df["Self_Employed"] = corr_df["Self_Employed"].map({"Yes": 1, "No": 0})
corr_df["Education"] = corr_df["Education"].map({"Graduate": 1, "Not Graduate": 0})
corr_df = corr_df.drop(columns=["Loan_ID", "Property_Area", "Dependents"])

plt.figure(figsize=(10, 7))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 5. Boxplots: Income & LoanAmount vs Loan Status
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=train_df, ax=axes[0], palette=["salmon", "steelblue"])
axes[0].set_title("Applicant Income vs Loan Status")
sns.boxplot(x="Loan_Status", y="LoanAmount", data=train_df, ax=axes[1], palette=["salmon", "steelblue"])
axes[1].set_title("Loan Amount vs Loan Status")
plt.tight_layout()
plt.savefig("income_loanamount_boxplots.png")
plt.show()

# 6. Scatter Plots
colors = {"Y": "steelblue", "N": "salmon"}
labels = {"Y": "Approved", "N": "Rejected"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter 1: ApplicantIncome vs LoanAmount
for status, grp in train_df.groupby("Loan_Status"):
    axes[0].scatter(grp["ApplicantIncome"], grp["LoanAmount"],
                    c=colors[status], label=labels[status],
                    alpha=0.5, s=30, edgecolors="none")
axes[0].set_title("Applicant Income vs Loan Amount")
axes[0].set_xlabel("Applicant Income")
axes[0].set_ylabel("Loan Amount")
axes[0].legend()

# Scatter 2: ApplicantIncome vs CoapplicantIncome
for status, grp in train_df.groupby("Loan_Status"):
    axes[1].scatter(grp["ApplicantIncome"], grp["CoapplicantIncome"],
                    c=colors[status], label=labels[status],
                    alpha=0.5, s=30, edgecolors="none")
axes[1].set_title("Applicant Income vs Coapplicant Income")
axes[1].set_xlabel("Applicant Income")
axes[1].set_ylabel("Coapplicant Income")
axes[1].legend()

# Scatter 3: LoanAmount vs Loan_Amount_Term
for status, grp in train_df.groupby("Loan_Status"):
    axes[2].scatter(grp["LoanAmount"], grp["Loan_Amount_Term"],
                    c=colors[status], label=labels[status],
                    alpha=0.5, s=30, edgecolors="none")
axes[2].set_title("Loan Amount vs Loan Term")
axes[2].set_xlabel("Loan Amount")
axes[2].set_ylabel("Loan Amount Term (months)")
axes[2].legend()

plt.suptitle("Scatter Plots by Loan Status", fontsize=14)
plt.tight_layout()
plt.savefig("scatter_plots.png")
plt.show()

# ============================================================
# PREPROCESSING
# ============================================================

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

# # Replace null values with the mean
# mean_term = x_train["Loan_Amount_Term"].mean()
# x_train["Loan_Amount_Term"] = x_train["Loan_Amount_Term"].fillna(mean_term)
# x_val["Loan_Amount_Term"] = x_val["Loan_Amount_Term"].fillna(mean_term)
#
# mean_loan = x_train["LoanAmount"].mean()
# x_train["LoanAmount"] = x_train["LoanAmount"].fillna(mean_loan)
# x_val["LoanAmount"] = x_val["LoanAmount"].fillna(mean_loan)
# x_test["LoanAmount"] = x_test["LoanAmount"].fillna(mean_loan)


# Replace null values with the mean
median_term = x_train["Loan_Amount_Term"].median()
x_train["Loan_Amount_Term"] = x_train["Loan_Amount_Term"].fillna(median_term)
x_val["Loan_Amount_Term"] = x_val["Loan_Amount_Term"].fillna(median_term)

median_loan = x_train["LoanAmount"].median()
x_train["LoanAmount"] = x_train["LoanAmount"].fillna(median_loan)
x_val["LoanAmount"] = x_val["LoanAmount"].fillna(median_loan)
x_test["LoanAmount"] = x_test["LoanAmount"].fillna(median_loan)


# Replace null values with the mode
mode_vals = {}
for col in ["Self_Employed", "Gender", "Dependents", "Credit_History"]:
    mode_vals[col] = x_train[col].mode().iloc[0]
    x_train[col] = x_train[col].fillna(mode_vals[col])
    x_val[col] = x_val[col].fillna(mode_vals[col])

# to cap and delete outliers
all_cols = ["CoapplicantIncome", "ApplicantIncome",
            "Loan_Amount_Term", "LoanAmount"]

for col in all_cols:
    Q1 = x_train[col].quantile(0.25)
    Q3 = x_train[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    x_train[col] = x_train[col].clip(lower, upper)
    x_val[col] = x_val[col].clip(lower, upper)

y_train = y_train.loc[x_train.index]

# Replace null values in test with the mean of training
x_test["Loan_Amount_Term"] = x_test["Loan_Amount_Term"].fillna(median_term)

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

# ============================================================
# TRAIN MODELS
# ============================================================

# --- Logistic Regression ---
log_results = []
c = np.logspace(-4, 2, 30)
log_solver = ['liblinear', 'lbfgs']

best_c_log = None
best_log_solver = None
best_score_log = -1

for j in log_solver:
    for i in c:
        log_model = LogisticRegression(solver=j, C=i, random_state=0)
        log_model.fit(x_train, y_train)
        y_pred = log_model.predict(x_val)
        log_score = f1_score(y_val, y_pred)
        log_results.append({"C": i, "solver": j, "f1": log_score})
        if log_score > best_score_log:
            best_score_log = log_score
            best_c_log = i
            best_log_solver = j

log_model = LogisticRegression(solver=best_log_solver, C=best_c_log, random_state=0)
log_model.fit(x_train, y_train)
log_df = pd.DataFrame(log_results)

# --- SVM ---
SVM_results = []
svm_kernel = ['linear', 'poly', 'rbf']
best_c_svm = None
best_svm_kernel = None
best_score_svm = -1

for j in svm_kernel:
    for i in c:
        svm_model = SVC(C=i, kernel=j, gamma='scale')
        svm_model.fit(x_train, y_train)
        y_pred = svm_model.predict(x_val)
        svm_score = f1_score(y_val, y_pred)
        SVM_results.append({"C": i, "Kernel": j, "f1": svm_score})
        if svm_score > best_score_svm:
            best_score_svm = svm_score
            best_c_svm = i
            best_svm_kernel = j

svm_model = SVC(C=best_c_svm, kernel=best_svm_kernel, gamma='scale')
svm_model.fit(x_train, y_train)
SVM_df = pd.DataFrame(SVM_results)

# --- Decision Tree ---
DT_results = []
max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, None]
min_samples_split = [2, 5, 10]
best_min_samples_split = None
best_max_depth = None
best_score_DT = -1

for j in min_samples_split:
    for i in max_depth_values:
        DT_model = DecisionTreeClassifier(criterion='gini', max_depth=i, min_samples_split=j, random_state=0)
        DT_model.fit(x_train, y_train)
        y_pred = DT_model.predict(x_val)
        DT_score = f1_score(y_val, y_pred)
        DT_results.append({"max_depth": i, "min_samples_split": j, "f1": DT_score})
        if DT_score > best_score_DT:
            best_score_DT = DT_score
            best_max_depth = i
            best_min_samples_split = j

DT_model = DecisionTreeClassifier(criterion='gini', max_depth=best_max_depth, min_samples_split=best_min_samples_split, random_state=0)
DT_model.fit(x_train, y_train)
DT_df = pd.DataFrame(DT_results)


# --- random forest ---

RDF_results = []
n_trees=[50,100,200,300,400,500,700,1000]
best_depth=None
best_n_tree=None
best_RDF_score=-1
for i in n_trees:
    for j in max_depth_values:
        RDF_model=RandomForestClassifier(n_estimators=i, max_depth=j, random_state=0)
        RDF_model.fit(x_train, y_train)
        y_pred = RDF_model.predict(x_val)
        RDF_score = f1_score(y_val, y_pred)
        RDF_results.append({"n_estimators": i, "max_depth": j, "f1": RDF_score})
        if RDF_score > best_RDF_score:
            best_RDF_score=RDF_score
            best_depth=j
            best_n_tree=i

RDF_model = RandomForestClassifier(n_estimators=best_n_tree, max_depth=best_depth, random_state=0)
RDF_model.fit(x_train, y_train)
RDF_df = pd.DataFrame(RDF_results)


# ============================================================
# EVALUATION
# ============================================================

# Logistic Regression
print('Logistic Regression')
print('Accuracy: ' + str(log_model.score(x_test, y_test) * 100) + '%')
yp_log = log_model.predict(x_test)
print("F1_Score:" + str(f1_score(y_test, yp_log) * 100) + "%")
print("Precision:" + str(precision_score(y_test, yp_log) * 100) + "%")
print("Recall:" + str(recall_score(y_test, yp_log) * 100) + "%")
print('Confusion Matrix :')
print(confusion_matrix(y_test, yp_log))
print('-------------------------------------------------------------')

# SVM
print('SVM')
print('Accuracy: ' + str(svm_model.score(x_test, y_test) * 100) + '%')
yp_SVM = svm_model.predict(x_test)
print("F1_Score:" + str(f1_score(y_test, yp_SVM) * 100) + "%")
print("Precision:" + str(precision_score(y_test, yp_SVM) * 100) + "%")
print("Recall:" + str(recall_score(y_test, yp_SVM) * 100) + "%")
print('Confusion Matrix :')
print(confusion_matrix(y_test, yp_SVM))
print('-------------------------------------------------------------')

# Decision Tree
print('Decision Tree ')
print('Accuracy: ' + str(DT_model.score(x_test, y_test) * 100) + '%')
yp_DT = DT_model.predict(x_test)
print("F1_Score:" + str(f1_score(y_test, yp_DT) * 100) + "%")
print("Precision:" + str(precision_score(y_test, yp_DT) * 100) + "%")
print("Recall:" + str(recall_score(y_test, yp_DT) * 100) + "%")
print('Confusion Matrix :')
print(confusion_matrix(y_test, yp_DT))
print('-------------------------------------------------------------')

# Random Forest
print('Random Forest')
print('Accuracy: ' + str(RDF_model.score(x_test, y_test) * 100) + '%')
yp_RDF = RDF_model.predict(x_test)
print("F1 Score: " + str(f1_score(y_test, yp_RDF) * 100) + "%")
print("Precision: " + str(precision_score(y_test, yp_RDF) * 100) + "%")
print("Recall: " + str(recall_score(y_test, yp_RDF) * 100) + "%")
print('Confusion Matrix:')
print(confusion_matrix(y_test, yp_RDF))

# ============================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================

models = ['Logistic Regression', 'SVM', 'Decision Tree','Random Forest']
accuracies = [
    log_model.score(x_test, y_test) * 100,
    svm_model.score(x_test, y_test) * 100,
    DT_model.score(x_test, y_test) * 100,
RDF_model.score(x_test, y_test) * 100
]
f1_scores = [
    f1_score(y_test, yp_log) * 100,
    f1_score(y_test, yp_SVM) * 100,
    f1_score(y_test, yp_DT) * 100,
f1_score(y_test, yp_RDF) * 100
]
precisions = [
    precision_score(y_test, yp_log) * 100,
    precision_score(y_test, yp_SVM) * 100,
    precision_score(y_test, yp_DT) * 100,
precision_score(y_test, yp_RDF) * 100
]
recalls = [
    recall_score(y_test, yp_log) * 100,
    recall_score(y_test, yp_SVM) * 100,
    recall_score(y_test, yp_DT) * 100,
recall_score(y_test, yp_RDF) * 100
]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='steelblue')
ax.bar(x - 0.5*width, f1_scores,  width, label='F1 Score',  color='salmon')
ax.bar(x + 0.5*width, precisions, width, label='Precision', color='mediumseagreen')
ax.bar(x + 1.5*width, recalls,    width, label='Recall',    color='mediumpurple')

ax.set_xlabel('Model')
ax.set_ylabel('Score (%)')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
# print(log_df)
# print('-------------------------------------------------------------')
# print(SVM_df)
# print('-------------------------------------------------------------')
# print(DT_df)

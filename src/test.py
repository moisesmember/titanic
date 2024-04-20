import matplotlib.pyplot as plt
import pandas as pd
#import janitor as jn
from sklearn import (
    ensemble,
    preprocessing,
    tree
)
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)
from yellowbrick.classifier import (
    ConfusionMatrix,
    ROCAUC
)
from yellowbrick.model_selection import (
    LearningCurve
)

# Read data
df = pd.read_excel("./database/titanic3.xls")
orig_df = df

# #pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(orig_df)
# #pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

print("---------------------------------------------")
print(df.dtypes)
print(df.shape)
print("---------------------------------------------")
print(df.describe().iloc[:, :2])
print("---------------------------------------------")
print(df.isnull().sum())
print("---------------------------------------------")
print(df.sex.value_counts(dropna=False))
print("---------------------------------------------")
print(df.embarked.value_counts(dropna=False))
print("---------------------------------------------")

df = df.drop(
    columns=[
        "name", "ticket", "home.dest", "boat", "body", "cabin"
    ]
)
print(df)
df = pd.get_dummies(df)
df = df.drop(columns="sex_male")
df = pd.get_dummies(df, drop_first=True)
print(df.isnull().sum())

#pd.set_option('display.max_columns', None)

y = df.survived
X = df.drop(columns="survived")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train)
print(X_test)
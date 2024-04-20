from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import (
    LogisticRegression
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import (
    KNeighborsClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier
)
import xgboost


class BaseModelAnalysis:
    @staticmethod
    def analysis(X, y):
        for model in [
            DummyClassifier,
            LogisticRegression,
            DecisionTreeClassifier,
            KNeighborsClassifier,
            GaussianNB,
            SVC,
            RandomForestClassifier,
            xgboost.XGBClassifier,
        ]:
            cls = model()
            kfold = model_selection.KFold(
                n_splits=10, shuffle=True, random_state=42
            )
            s = model_selection.cross_val_score(
                cls, X, y, scoring="roc_auc", cv=kfold
            )
            print(
                f" {model.__name__:22} AUC"
                f" {s.mean():.3f} STD: {s.std():.2f}"
            )

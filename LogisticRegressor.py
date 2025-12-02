# Test Logisitic Regression
from sklearn.linear_model import LogisticRegression

def testLogisticRegression(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf.score(X_val, y_val)
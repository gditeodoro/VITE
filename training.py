from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def readData(filename, test_size):
    data = pd.read_csv(filename)
    x = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]

    x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=test_size, stratify=y, random_state=10)

    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_ts = scaler.transform(x_ts)

    return x_tr, y_tr, x_ts, y_ts, data, scaler

def trainModel(x, y,n_estimators,max_depth,max_features):
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,class_weight='balanced',
                                   max_features=max_features,random_state=100)
    model.fit(x,y)
    return model

def evaluateModel(model, x, y):
    y_pred = model.predict(x)
    print("Accuracy {}".format(accuracy_score(y, y_pred)))
    print(confusion_matrix(y, y_pred))
    return accuracy_score(y, y_pred),confusion_matrix(y, y_pred)
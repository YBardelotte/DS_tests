import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from typing import List

class WeakLogisticRegression:
    
    def __init__(self):
        self.model = LogisticRegression()
        
    def fit(self, X, y, sample_weights):
        self.model.fit(X, y, sample_weight=sample_weights)
        
    def predict(self, X):
        return self.model.predict(X)

class AdaBoostClassifier:
    
    def __init__(self, n_estimators = 50):
        self.n_estimators: int = n_estimators
        self.alpha: List = []
        self.models: List = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        sample_weight = np.ones(n_samples) / n_samples        
        
        for _ in range(self.n_estimators):
            print(f"Estimation number {_}")
            logistic = WeakLogisticRegression()
            logistic.fit(X, y, sample_weight)
            y_pred = logistic.predict(X)
            
            error = float(np.dot(sample_weight,( y_pred != y).astype(float)) / np.sum(sample_weight))
            # print(error)
            # Compute the alpha value
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            # print(alpha)
            # Update the weights
            sample_weight *= np.exp(-alpha * y_pred * y)
            sample_weight /= np.sum(sample_weight)
            
            # Store the model and alpha
            self.models.append(logistic)
            self.alpha.append(alpha)
            
    def predict(self, X):
        # Weighted sum of predictions
        pred_sum = sum(alpha * model.predict(X) for model, alpha in zip(self.models, self.alpha))
        return np.sign(pred_sum)
    
df = pd.read_csv(r"D:\Documentos\DS_tests\data\Iris.csv", sep=",")

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df.Species

lab_encoder = LabelEncoder()
y = lab_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54)

ada = AdaBoostClassifier(n_estimators=10)
ada.fit(X_train,y_train)
y_pred = ada.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

print(acc_score)
# print(f1_score)
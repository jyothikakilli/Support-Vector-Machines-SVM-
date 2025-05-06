import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Adjust the parameters
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred = svm_linear.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Linear SVM Accuracy: {accuracy:.2f}')
svm_nonlinear = SVC(kernel='rbf')
svm_nonlinear.fit(X_train, y_train)

y_pred_nl = svm_nonlinear.predict(X_test)
accuracy_nl = accuracy_score(y_test, y_pred_nl)
print(f'Non-Linear SVM Accuracy: {accuracy_nl:.2f}')
def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100),
                         np.linspace(X[:,1].min(), X[:,1].max(), 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
    plt.show()

plot_decision_boundary(svm_linear, X_test, y_test)
plot_decision_boundary(svm_nonlinear, X_test, y_test)
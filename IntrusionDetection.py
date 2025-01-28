import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from scipy.sparse import csr_matrix, hstack

#Load dataset
data = pd.read_csv("hikari.csv")

#Inspect class distribution
data['Label'].value_counts(normalize=True).plot(kind='bar', title='Class Distribution')

#Create sample dataset
data = data.sample(frac=0.1, random_state=42)

#Split features and labels
x = data.drop(columns=["Label"])
y = data['Label']

#Encode non-numeric columns with LabelEncoder
non_numeric_cols = x.select_dtypes(include=['object']).columns
le = LabelEncoder()

#Use LabelEncoder for categorical data
for col in non_numeric_cols:
    x[col] = le.fit_transform(x[col].astype(str))

#Convert to sparse matrix format column-wise
numeric_cols = x.select_dtypes(include=['int64', 'float64']).columns
sparse_numeric = csr_matrix(x[numeric_cols].values)
sparse_categorical = csr_matrix(x[non_numeric_cols].values)

#Combine numeric and categorical sparse matrices
x_sparse = hstack([sparse_numeric, sparse_categorical])

#Feature reduction with PCA
pca = PCA(n_components=50, random_state=42)
x_reduced = pca.fit_transform(x_sparse.toarray())

#Scale the reduced features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_reduced)

#Encode target variable
y_encoded = LabelEncoder().fit_transform(y)

#Split data into training and testing sets with a 80/20 split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

#Display set shapes
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Define grid search param combos to find optimal hyperparameters
# param_grid_rf = {
#     'n_estimators': [50, 100],  
#     'max_depth': [10, 20],      
#     'min_samples_split': [2, 5], 
#     'min_samples_leaf': [1, 2]
# }

# param_grid_knn = {
#     'n_neighbors': [3, 5],      
#     'metric': ['euclidean', 'manhattan'],
#     'weights': ['uniform', 'distance']
# }

#Create grid search to calculate optimal hyperparameters
# grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
#                               param_grid=param_grid_rf, 
#                               scoring='accuracy', 
#                               cv=3, 
#                               verbose=2)

# grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), 
#                                param_grid=param_grid_knn, 
#                                scoring='accuracy', 
#                                cv=3, 
#                                verbose=2)

#Evaluate model with cross-validation
def evaluate_model(model, x_train, y_train):
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    return cv_scores.mean(), cv_scores.std()

#Threshold Dummy Classifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(x_test))
print("Baseline Dummy Classifier Accuracy:", dummy_acc)

#Random Forest Classifier
rf = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=50, random_state=42)
rf_mean_acc, rf_std_acc = evaluate_model(rf, x_train, y_train)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print("\nRandom Forest:")
print(f"Cross-Validation Accuracy: {rf_mean_acc:.4f} (+/- {rf_std_acc:.4f})")
print(f"Test Accuracy: {acc_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")

#K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(metric='manhattan', n_neighbors=5, weights='uniform')
knn_mean_acc, knn_std_acc = evaluate_model(knn, x_train, y_train)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
print("\nK-Nearest Neighbors:")
print(f"Cross-Validation Accuracy: {knn_mean_acc:.4f} (+/- {knn_std_acc:.4f})")
print(f"Test Accuracy: {acc_knn:.4f}")
print(f"F1 Score: {f1_knn:.4f}")

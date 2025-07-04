from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

n_range = range(1, 21)


cv_scores = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation
    cv_scores.append(np.mean(scores))

# Find the k with the highest score
best_k = k_range[np.argmax(cv_scores)]
print(f"Best value for n_neighbors: {best_k}")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'l2_regularization': [0.0, 1.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Display results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score (accuracy): {grid_search.best_score_}")

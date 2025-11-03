import pickle
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
print("Loading data...")
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convert letter labels to numbers (A=0, B=1, etc.)
label_map = {chr(65+i): i for i in range(26)}  # A=0, B=1, ..., Z=25
numeric_labels = np.array([label_map[label] for label in labels])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, numeric_labels, test_size=0.2, shuffle=True, stratify=numeric_labels)

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create base XGBoost classifier
base_model = XGBClassifier(
    objective='multi:softmax',
    num_class=26,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Set up GridSearchCV with cross-validation
print("Starting GridSearchCV. This may take some time...")
start_time = time.time()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=2
)

# Fit the grid search
grid_search.fit(x_train, y_train)

# Print the best parameters and score
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"\nBest cross-validation accuracy: {grid_search.best_score_*100:.2f}%")

# Train final model with best parameters
final_model = XGBClassifier(
    objective='multi:softmax',
    num_class=26,
    use_label_encoder=False,
    eval_metric='mlogloss',
    **grid_search.best_params_
)

final_model.fit(x_train, y_train)

# Evaluate on test set
y_pred = final_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Set Accuracy: {test_accuracy*100:.2f}%")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and best parameters
print("\nSaving model and parameters...")
with open('model.p', 'wb') as f:
    pickle.dump({
        'model': final_model,
        'best_params': grid_search.best_params_,
        'training_time': time.time() - start_time,
        'test_accuracy': test_accuracy
    }, f)

print(f"\nTotal training time: {(time.time() - start_time)/60:.2f} minutes")

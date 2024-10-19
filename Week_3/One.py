import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('Week_3\insurance_claims.csv')

data.replace('?', pd.NA, inplace=True)
data.dropna(axis=1, thresh=int(0.8 * len(data)), inplace=True) 
data.dropna(inplace=True) 

# Convert categorical variables into dummy/one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop('fraud_reported_Y', axis=1)  # Assuming 'fraud_reported_Y' is the target
y = data['fraud_reported_Y']

# Step 3: Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
f1_dt = f1_score(y_test, y_pred_dt)
print("Decision Tree F1 Score:", f1_dt)
print("\nClassification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['Not Fraud', 'Fraud'], filled=True)
plt.show()

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf)
print("Random Forest F1 Score:", f1_rf) 
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(5)

print("Top 5 Important Features:\n", feature_importance)

# Optional: Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 5 Important Features')
plt.gca().invert_yaxis()
plt.show()
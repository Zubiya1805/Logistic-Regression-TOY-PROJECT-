import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

# 1. Load Data 
df = pd.read_csv('placement.csv')
df = df.iloc[:, 1:]  # Drop unnamed index column
print("Dataset shape:", df.shape)
print(df.head())

# 2. EDA 
plt.figure(figsize=(8, 6))
plt.scatter(df['cgpa'], df['iq'], c=df['placement'], cmap='viridis')
plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('Placement by CGPA and IQ')
plt.colorbar(label='Placement (0=No, 1=Yes)')
plt.savefig('eda_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. Feature / Target Split 
X = df.iloc[:, 0:2]   # cgpa, iq
y = df.iloc[:, -1]    # placement

# 4. Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 5. Feature Scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 6. Model Training 
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 7. Evaluation 
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nPredictions : {y_pred}")
print(f"Actual      : {y_test.values}")
print(f"Accuracy    : {acc:.2f}")

# 8. Decision Boundary Plot 
plot_decision_regions(X_train, y_train.values, clf=clf, legend=2)
plt.title('Logistic Regression Decision Boundary (Scaled Features)')
plt.xlabel('CGPA (scaled)')
plt.ylabel('IQ (scaled)')
plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()

# 9. Save Model 
pickle.dump(clf, open('model.pkl', 'wb'))
print("\nModel saved to model.pkl")

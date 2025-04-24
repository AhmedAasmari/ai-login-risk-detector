# ai-login-risk-detector
A simple AI model that classifies login attempts as safe or potential attacks using fake data for training. Focused on understanding false positives in cybersecurity.


## 🔍 Project Overview
This project simulates login attempt data and uses a Decision Tree Classifier to detect potential attacks.  
It focuses on analyzing **False Positives**, which are safe login attempts misclassified as threats.

The project is built using:
- Python
- Pandas
- NumPy
- scikit-learn
- Seaborn / Matplotlib


## 📦 Import Required Libraries

We begin by importing the necessary Python Libraries :

- **pandas** and **numpy** is for data handling and numerical operations.
- **scikit-learn** modules for splitting the dataset, training the model, and evaluating performance.
- **matplotlib** and **seaborn** for visualizing the results, especially the confusion matrix.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```


## 🧪 Generate Simulated Login Data

To train our AI model, we generate a fake dataset that simulates login attempts.  
Each record contains:

- `ip_risk_score`: A random value (1–99) representing how risky the IP is.
- `failed_attempts`: Number of failed login attempts (0–5).
- `login_time_hour`: The hour of the login attempt (0–23).
- `is_attack`: Whether it was an attack (1) or a normal login (0).  
- 15% of the data is labeled as attacks to simulate real-world imbalance.

```python
np.random.seed(42)

data = {
    'ip_risk_score': np.random.randint(1, 100, 200),
    'failed_attempts': np.random.randint(0, 6, 200),
    'login_time_hour': np.random.randint(0, 24, 200),
    'is_attack': np.random.choice([0, 1], size=200, p=[0.85, 0.15])
}

df = pd.DataFrame(data)
```


## 📊 Prepare the Dataset and Train the Model

After generating the fake login attempt data, we explore the first few rows using `df.head()` to verify the structure.

```python
df.head()
```
<img width="440" alt="image" src="https://github.com/user-attachments/assets/455efcb5-0e03-4fcf-850e-67644e4e7f2d" />


This confirms the dataset includes:

- ip_risk_score

- failed_attempts

- login_time_hour

- is_attack (our target label)


## 🎯 Select Features and Target
We define:

- X: the input features used to make predictions

- y: the target output (whether it's an attack)

```python
X = df[['ip_risk_score', 'failed_attempts', 'login_time_hour']]
y = df['is_attack']
```


## ✂️ Split the Data
We split the data into training and testing sets using train_test_split, with 30% reserved for testing.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


## 🌳 Train a Decision Tree Classifier
We train a simple Decision Tree Classifier on the training data.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```
The model is now ready to make predictions based on new login attempts.


## 📈 Evaluate the Model

After training, we test the model on unseen data using `X_test`.

```python
y_pred = model.predict(X_test)
```


## 🧪 Generate Evaluation Metrics
We calculate:

- **Confusion Matrix**: to analyze prediction outcomes

- **Classification Report**: to check precision, recall, and F1-score

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
```


## 📊 Visualize Confusion Matrix with Labels
We use matplotlib and seaborn to plot the confusion matrix with clear **TP / FP / FN / TN** labels.

```python
labels = [
    f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}",
    f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"
]
labels = np.array(labels).reshape(2,2)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=['Safe', 'Attack'], yticklabels=['Safe', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with TP/FP/FN/TN')
plt.tight_layout()
plt.show()
```

<img width="554" alt="image" src="https://github.com/user-attachments/assets/09df1e4b-3134-4796-96c1-84a9f505e8a9" />


## 📝 Interpretation
From the confusion matrix:

- **TN (True Negatives)**: Correctly predicted safe logins

- **TP (True Positives)**: Correctly predicted attacks

- **FP (False Positives)**: Safe logins predicted as attacks → **⚠️ False alarms**

- **FN (False Negatives)**: Missed attacks → **⚠️ Dangerous**

This helps evaluate how good the model is at minimizing false positives, which is the core goal of this project.

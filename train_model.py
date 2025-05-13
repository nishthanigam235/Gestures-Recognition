# train_model.py

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

data_dir = "asl_data"
all_data = []

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file), header=None)
        df.dropna(inplace=True)

        # Skip files not having 127 columns (126 features + label)
        if df.shape[1] != 127:
            print(f"Skipping {file} due to incorrect shape: {df.shape}")
            continue

        all_data.append(df)

# Combine all valid data
data = pd.concat(all_data, ignore_index=True)
X = data.iloc[:, :-1].astype(float)
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)

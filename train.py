import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Price': [50, 70, 90, 110, 130]
}

df = pd.DataFrame(data)

X = df[['Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("HousePriceProject/house_data.csv")

# Convert text → number
df['location'] = df['location'].map({
    'rural': 0,
    'suburban': 1,
    'urban': 2
})

# Features & target
X = df[['area', 'bedrooms', 'location']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
print("Predicted Price:", model.predict([[1600, 3, 2]]))

# Graph
plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
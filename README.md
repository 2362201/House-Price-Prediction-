# House-Price-Prediction-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\Users\adity\Downloads\housing.csv", sep='\s+', header=None)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
              'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'Price']
print("Columns:", df.columns)
print(df.head(2))
print(df.head())
print(df.isnull().sum())
sns.pairplot(df)
plt.show()
X = df[['RM', 'LSTAT', 'PTRATIO']] 
Y = df[['Price']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred  = model.predict(X_test)
print("MSE:",mean_squared_error(Y_test, y_pred))
print("R2 Score:", r2_score(Y_test, y_pred))
df_results = pd.DataFrame({'Actual':Y_test, 'Predicted':y_pred})
print(df_results.head())
df_bar = pd.DataFrame({'Actual': Y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df_bar = df_bar.head(25)

# Create bar plot
df_bar.plot(kind='bar', figsize=(12, 6))
plt.title('Actual vs Predicted House Prices (First 25 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.tight_layout()
plt.show()
plt.xlabel("Actual Prices")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted House Price")
plt.show()

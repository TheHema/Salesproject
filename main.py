import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='SalesDB'`
)
print('Connected:', db)
mycursor = db.cursor()
mycursor.execute('SELECT Product_ID, Sales_Amount, Quantity_Sold, Sale_Date FROM Products')
data = mycursor.fetchall()
df = pd.DataFrame(data, columns=['Product_ID', 'Sales_Amount', 'Quantity_Sold', 'Sale_Date'])
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
df['Day'] = df['Sale_Date'].dt.day
df['Month'] = df['Sale_Date'].dt.month
df['Year'] = df['Sale_Date'].dt.year
X = df[['Day', 'Month', 'Quantity_Sold']]
y = df['Sales_Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Linear Regression Predicted Sales Amounts:", predictions)
print("Actual Sales Amounts:", y_test.values)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R-squared:", r2_score(y_test, predictions))

plt.figure(figsize=(6, 4))
plt.scatter(X_test['Day'], y_test, color='blue', label='Actual Sales Amount', zorder=2)
plt.scatter(X_test['Day'], predictions, color='red', linestyle='--', marker='o', label='Predicted Sales Amount', zorder=1)
plt.xlabel('Day')
plt.ylabel('Sales Amount')
plt.title('Actual vs. Predicted Sales Amount (Linear Regression)')
plt.legend()
plt.grid(True)
plt.savefig('salesdata_comparison_lr.png')
plt.close()

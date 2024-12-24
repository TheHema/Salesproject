# Import necessary libraries
import mysql.connector  # For connecting to the MySQL database
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.linear_model import LinearRegression  # For building a linear regression model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For model evaluation
import matplotlib  # For data visualization
matplotlib.use('Agg')  # Use 'Agg' backend to save plots instead of displaying them
import matplotlib.pyplot as plt  # For creating and saving plots

# Connect to the MySQL database
db = mysql.connector.connect(
    host='localhost',  # Database host
    user='root',  # Database username
    password='root',  # Database password
    database='SalesDB'  # Database name
)
print('Connected:', db)  # Print confirmation of database connection

# Execute an SQL query to retrieve sales data
mycursor = db.cursor()
mycursor.execute('SELECT Product_ID, Sales_Amount, Quantity_Sold, Sale_Date FROM Products')
data = mycursor.fetchall()  # Fetch all rows from the query result

# Load the data into a Pandas DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['Product_ID', 'Sales_Amount', 'Quantity_Sold', 'Sale_Date'])

# Convert the Sale_Date column to a datetime object
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])

# Extract day, month, and year from the Sale_Date for feature engineering
df['Day'] = df['Sale_Date'].dt.day
df['Month'] = df['Sale_Date'].dt.month
df['Year'] = df['Sale_Date'].dt.year

# Define features (X) and target variable (y) for the model
X = df[['Day', 'Month', 'Quantity_Sold']]  # Feature variables
y = df['Sales_Amount']  # Target variable: sales amount

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Print the predicted sales amounts and actual sales for comparison
print("Linear Regression Predicted Sales Amounts:", predictions)
print("Actual Sales Amounts:", y_test.values)

# Evaluate the model using common regression metrics
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))  # Measures average error
print("Mean Squared Error:", mean_squared_error(y_test, predictions))  # Penalizes larger errors
print("R-squared:", r2_score(y_test, predictions))  # Measures how well the model explains the variance

# Visualize the results by comparing actual and predicted sales amounts
plt.figure(figsize=(6, 4))
plt.scatter(X_test['Day'], y_test, color='blue', label='Actual Sales Amount', zorder=2)  # Actual sales
plt.scatter(X_test['Day'], predictions, color='red', linestyle='--', marker='o', label='Predicted Sales Amount', zorder=1)  # Predicted sales
plt.xlabel('Day')  # Label for x-axis
plt.ylabel('Sales Amount')  # Label for y-axis
plt.title('Actual vs. Predicted Sales Amount (Linear Regression)')  # Plot title
plt.legend()  # Add a legend
plt.grid(True)  # Add a grid for better readability
plt.savefig('salesdata_comparison_lr.png')  # Save the plot as a PNG file for the portfolio
plt.close()  # Close the plot to free resources

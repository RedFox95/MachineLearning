import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the training data
train_df = pd.read_csv("train_data.csv")

# Extract features and target variable
X_train = train_df[['revenue', 'expenses']]
y_train = train_df['risk']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
n_samples = 1000
revenue = np.random.rand(n_samples) * 1000
expenses = np.random.rand(n_samples) * 1000
profit_margin = (revenue - expenses) / revenue

# Assume companies with a negative profit margin are high risk
risk = (profit_margin < 0).astype(int)

# Create a DataFrame
df = pd.DataFrame({
    'revenue': revenue,
    'expenses': expenses,
    'risk': risk
})

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Optionally, save the datasets to CSV files
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

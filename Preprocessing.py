import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("games-regression-dataset.csv")

# Print number of missing values in each column
print(df.isnull().sum())

# Number of duplicated data
print(f"Number of duplicated data {df.duplicated().sum()}")

# Dataframe size
originalSize = df.shape[0]
print(f"The total size is {originalSize}")

# Drop the duplicates
df = df.drop_duplicates()
print("Duplicated data dropped !")
print(f"Number of dropped columns {originalSize - df.shape[0]}")


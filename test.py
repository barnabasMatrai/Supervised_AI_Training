import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv(os.getcwd() + "\\dataset.csv")

print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.head())

# class_counts = df['popularity'].value_counts()

# class_counts.plot(kind='bar')

# plt.title("Class Distribution")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()

df.boxplot(column="danceability")
plt.show()
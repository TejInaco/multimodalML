import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

IMAGES_PATH = 'data/images/'
TABULAR_PATH = 'data/styles.csv'
SAVE_PATH = 'data/prepared_data.csv'

df = pd.read_csv(TABULAR_PATH, nrows=None, error_bad_lines=False)   # error_bad_lines=False drops instances with too many columns
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)

df = df.loc[df['image'].isin(os.listdir(IMAGES_PATH))]  # keep rows that have an image in the IMAGES_PATH
df = df.drop('year', axis=1)

plt.style.use('bmh')
## Exploratory Analysis
'''
# Season
df.season.value_counts().sort_values().plot(kind='barh')
plt.show()
# Master Category
df.masterCategory.value_counts().sort_values().plot(kind='barh')
plt.show()
# Sub Category
df.subCategory.value_counts().sort_values().plot(kind='barh')
plt.tight_layout()
plt.show()
# Articly Type
plt.figure(figsize=(10,50))
df.articleType.value_counts().sort_values().plot(kind='barh')
plt.tight_layout()
plt.show()
# Base colour
df.baseColour.value_counts().sort_values().plot(kind='barh')
plt.tight_layout()
plt.show()
# Usage
df.usage.value_counts().sort_values().plot(kind='barh')
plt.show()

## Data Preparation
# Balacing label samples
num_of_samples = len(df[df.season == 'Spring'])
spring = df[df.season == 'Spring']
summer = df[df.season == 'Summer']
winter = df[df.season == 'Winter']
fall = df[df.season == 'Fall']
summer_sample = summer.sample(n=num_of_samples)
winter_sample = winter.sample(n=num_of_samples)
fall_sample = fall.sample(n=num_of_samples)
frames = [spring, summer_sample, winter_sample, fall_sample]
final_df = pd.concat(frames)

final_df.season.value_counts().sort_values().plot(kind='barh')
plt.show()
'''
df.to_csv(SAVE_PATH, index=False)
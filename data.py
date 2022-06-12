import csv
import random
import pandas as pd

NUM_ROWS = 100
NUM_COLS = 100

rows_written = 0
data = []

for i in range(NUM_ROWS):
    row = []
    for j in range(NUM_COLS - 1):
        row.append(random.uniform(0, 1))

    # append the target variable as a random integer of two categories [0, 1]
    row.append(random.randint(0, 1) * 1.0)

    data.append(row)
    rows_written += 1

df = pd.DataFrame.from_records(data)

# write the dataframe with rows to csv
file_name = "data.csv"
df.to_csv(file_name, index=False)

print(f'wrote dataframe of {rows_written} rows to: {file_name}')

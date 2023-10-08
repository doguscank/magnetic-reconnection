import re
import pandas as pd

with open("data/description.txt", "r") as f:
    raw_data = f.read()

column_data = raw_data.split("\n")
columns = []

# Ids etc. must be int value
int_data_columns = [0, 1, 2, 3, 4, 5, 6, 7]

for column_datum in column_data:
    cleaned_line = re.sub(r'\s+', ' ', column_datum).rstrip()
    column = cleaned_line.split(" ")[1:]
    columns.append(" ".join(column).replace("/", ""))

with open("data/raw_data.lst", "r") as f:
    raw_data = f.read()

line_data = raw_data.split("\n")[:-1]
data = []

for line_datum in line_data:
    cleaned_line = re.sub(r'\s+', ' ', line_datum)
    datum = cleaned_line.split(" ")
    datum = [int(d) if idx in int_data_columns else float(d) for idx, d in enumerate(datum)]
    data.append(datum)

df = pd.DataFrame(data=data, columns=columns)

# Get unique years
years = df['YEAR'].unique()

# Export data for each year to a separate JSON file
for year in years:
    filtered_df = df[df['YEAR'] == year]
    file_path = f'data/{year}.json'
    filtered_df.to_json(file_path, orient='records')

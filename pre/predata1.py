import pandas as pd
from datetime import datetime

df = pd.read_csv("international-airline-passengers.csv", engine = "python", skipfooter =3)

#rename the columns' name#
df.columns = ["month", "passengers"]
# print(df.columns)
# print(df['passengers'])

#add a new column#
df['ones'] = 1
df['dt'] = df.apply(lambda row: datetime.strptime(row['month'], "%Y-%m"), axis=1)
print(df.head(5))




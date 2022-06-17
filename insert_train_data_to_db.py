import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text as sa_text
from database import engine
from models import churn_train
from database import SessionLocal

# read data
df = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv")
print(df.head())

# Starts database insertion
session = SessionLocal()

# First truncate table with sqlalchemy
session.execute(sa_text(''' TRUNCATE TABLE churn_train  '''))
session.commit()

# Second insert training data
records_to_insert = []

for df_idx, line in df.iterrows():
    records_to_insert.append(
                    churn_train(RowNumber = line[0],
                    CustomerId =line[1],
                    Surname=line[2],
                    CreditScore=line[3],
                    Geography =line[4],
                    Gender=line[5],
                    Age=line[6],
                    Tenure =line[7],
                    Balance=line[8],
                    NumOfProducts=line[9],
                    HasCrCard =line[10],
                    IsActiveMember=line[11],
                    EstimatedSalary=line[12]
                    )
    )

session.bulk_save_objects(records_to_insert)
session.commit()
# Ends database insertion

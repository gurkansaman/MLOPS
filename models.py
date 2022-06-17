from database import Base
from sqlalchemy import Column, String, Integer, Float, DateTime, Text
from sqlalchemy.sql import func


class churn(Base):
    __tablename__ = "churn"
    __table_args__ = {'extend_existing': True}

    RowNumber = Column(Integer, autoincrement=True, primary_key=True)
    CustomerId = Column(Float)
    Surname = Column(String(64))
    CreditScore = Column(Integer)
    Geography = Column(String(64))
    Gender = Column(String(64))
    Age = Column(Integer)
    Tenure = Column(Integer)
    Balance = Column(Float)
    NumOfProducts = Column(Integer)
    HasCrCard = Column(Integer)
    IsActiveMember = Column(Integer)
    EstimatedSalary = Column(Float)
    prediction = Column(String)
    prediction_time = Column(DateTime(timezone=True), server_default=func.now())
    client_ip = Column(String(20))



class churn_train(Base):
    __tablename__ = "churn_train"
    __table_args__ = {'extend_existing': True}

    RowNumber = Column(Integer, autoincrement=True, primary_key=True)
    CustomerId = Column(Float)
    Surname = Column(Text)
    CreditScore = Column(Float)
    Geography = Column(Text)
    Gender = Column(Text)
    Age = Column(Float)
    Tenure = Column(Float)
    Balance = Column(Float)
    NumOfProducts = Column(Float)
    HasCrCard = Column(Float)
    IsActiveMember = Column(Float)
    EstimatedSalary = Column(Float)



import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', None)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    # location cols
    location_cols = [col for col in dataframe.columns if ("Zip" in col or "Longitude" in col or  "Lati" in col )]

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and col not in location_cols]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O" and col not in location_cols]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O" and col not in location_cols]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car and col not in location_cols]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and "ID" not in col and col not in location_cols]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'location_cols: {len(location_cols)}')
    return location_cols, cat_cols, num_cols, cat_but_car
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# read data
df = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv")
df.head()
df.columns
df = df.drop(columns=['RowNumber','CustomerId','Surname'])
df.head()

# Checking:
check_df(df, head=5)

# Column Categorize:
location_cols, cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols
cat_but_car
cat_cols

#Missing Values:

df.isnull().sum()
# Sorun yoktur.

#Outlier
#check_outlier(df,num_cols)
#Outlier yoktur.

#Label
cat_cols.remove('Exited')
cat_cols

# Kategorik değişkenlerin analizi:
df.Geography.value_counts()
df.NumOfProducts.value_counts()
df.HasCrCard.value_counts()
df.IsActiveMember.value_counts()
df.Exited.value_counts()


# Ağaç tabanlı bir algoritma kullanacağım için Normalizasyona gerek duymadım.
# Ağaç tabanlı bir algoritma kullanacağım için One-hot benzeri dönüşüme gerek yoktur.

# Feature matrix
X=df[df.columns[~df.columns.isin(["Exited"])]]
X.head()

# Output variable
y =df.loc[:,"Exited"]
y.head()

# split test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

###### Entegrasyon #####################
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

experiment_name = "Churn_Modelling"
mlflow.set_experiment(experiment_name)

registered_model_name="ChurnModel"


###############################################
# CART PIPELINE
################################################

numeric_features = ["CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_features = ["Geography", "Gender"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

cart_model = DecisionTreeClassifier(random_state=17)
pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", GridSearchCV(cart_model,
                                                                       param_grid={'max_depth': [2, 5, 6, 8, 10, None], "min_samples_split": [2, 4, 5, 6, 8, 10]}, cv=2)
)])

pipe.fit(X_train, y_train)


#Sonuçlar
cv_results = cross_validate(pipe, X_test, y_test, cv=5,  scoring=["accuracy", "f1", "roc_auc"])

test_accuracy = cv_results['test_accuracy'].mean()
test_f1 = cv_results['test_f1'].mean()
test_roc_auc = cv_results['test_roc_auc'].mean()

##################################################
# MLFLOW

with mlflow.start_run(run_name="with-CART2") as run:
    estimator = pipe
    mlflow.log_param("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(estimator, "model", registered_model_name=registered_model_name)
    else:
        mlflow.sklearn.log_model(estimator, "model")


#import pandas as pd
#churn
#churn = pd.DataFrame([[1, 'Germany','Male', 22, 12, 3, 2, 1, 2, 3]],
#                     columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
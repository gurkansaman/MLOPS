In this project we will deploy advertising model using mlflow model registry.

## 1. Start mlflow
```commandline
cd mlflow

docker-compose up --build -d

docker-compose stop prod test jenkins
```

## 2. Copy/push your project to VM.

## 3. Activate/create conda/virtual env

## 4. Install requirements

## 5. Train and register your model to mlflow
` python model_development/train_with_mlflow.py`

## 6. Learn your model version
- From MLflow UI learn model version. Enter it main.py and copy/push main.py to VM.

## 7. Start uvicorn
```commandline
(fastapi) [train@localhost 10]$ uvicorn \
main:app --host 0.0.0.0 \
--port 8002 \
--reload
```
## 8. Open docs
` http://localhost:8002/docs# `


## 9. Docker
```commandline
docker image build -t mlflow-fastapi-advertising:1.0 .

docker run --rm \
--name ml-prediction \
-p 8002:8000 -d mlflow-fastapi-advertising:1.0
```
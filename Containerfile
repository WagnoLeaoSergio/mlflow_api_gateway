FROM python:3.7-alpine
COPY . /app
WORKDIR /app
RUN pip install .
RUN mlflow_api_gateway create-db
RUN mlflow_api_gateway populate-db
RUN mlflow_api_gateway add-user -u admin -p admin
EXPOSE 5000
CMD ["mlflow_api_gateway", "run"]

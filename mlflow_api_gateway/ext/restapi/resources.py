import os
import time
import mlflow

from joblib import dump
from werkzeug.utils import secure_filename

from flask import abort, jsonify, request, send_file
from flask_restful import Resource

from ...processor import train_model


class MLFlowGateway(Resource):

    def get(self, user_id):

        filter_string = f"attributes.status = 'FINISHED' and tags.`mlflow.user` = '{user_id}'"

        experiment_name = "prototype_v0"

        mlflow.set_tracking_uri("http://127.0.0.1:5001")

        user_runs = mlflow.search_runs(
            filter_string=filter_string,
            experiment_names=[experiment_name],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if len(user_runs) == 0:
            return {
                "OK": False,
                "error": "No run registered for the specified user"
            }

        artifact_uri = user_runs["artifact_uri"][0]

        logged_model = f"{artifact_uri}/model"

        loaded_model = mlflow.pyfunc.load_model(logged_model)

        raise NotImplementedError(
            "Model serialization for sending not implemented yet.")

    def post(self, user_id="USER1"):

        health_file = request.files.get('health_file')

        if health_file is None:
            return {
                "OK": False,
                "Error": "No health_file provided"
            }, 400

        health_file_name = secure_filename(health_file.filename)

        health_file.save(os.path.join("datasets", health_file_name))

        now = int(time.time() * 1000)
        user = user_id
        run_name = f"{user}_{now}"
        experiment_name = "prototype_v0"
        experiment_id = "0"

        mlflow.set_tracking_uri("sqlite:///mlruns.db")

        current_experiment = mlflow.search_experiments(
            filter_string=f"name = '{experiment_name}'"
        )

        if len(current_experiment) == 0:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = current_experiment[0].experiment_id

        run_id = None
        predicted_data = None

        with mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
        ) as run:
            run_id = run.info.run_id
            mlflow.set_tag("mlflow.user", user)
            mlflow.autolog()

            predicted_data = train_model()

        mlflow.register_model(
            f"runs:/{run_id}/sklearn-model",
            "sklearn-k_nearest_neighboor-model",
            tags={"user_id": user_id}
        )

        return {
            "predictions": predicted_data
        }, 200

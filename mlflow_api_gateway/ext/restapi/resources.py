import os
import time
import mlflow
import numpy as np
import pandas as pd

from werkzeug.utils import secure_filename

from flask import request
from flask_restful import Resource

from ...processor import preprocess_data, build_dataset, train_model

from hyperopt import tpe, hp, fmin, Trials


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
        experiment_name = "prototype_v1"
        experiment_id = "0"

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        # mlflow.set_tracking_uri("sqlite:///mlruns.db")

        current_experiment = mlflow.search_experiments(
            filter_string=f"name = '{experiment_name}'"
        )

        if len(current_experiment) == 0:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = current_experiment[0].experiment_id

        predicted_data = None

        health_data = pd.read_csv(os.path.join("datasets", "health_data.csv"))
        dados = preprocess_data(health_data)
        x, y = build_dataset(dados)

        search_space = {
            "user": user,
            "run_name": run_name,
            "experiment_id": experiment_id,
            "x": x,
            "y": y,
            "algorithm": hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
            "n_neighbors": hp.uniformint('n_neighbors', 1, 10),
            "p": hp.choice('p', [1, 2]),
            "leaf_size": hp.choice('leaf_size', [20, 30, 40]),
            "weights": hp.choice('weights', ['uniform', 'distance'])
        }

        trials = Trials()

        fmin(
            fn=train_model,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )

        best_model = trials.results[
            np.argmin(
                [r['loss'] for r in trials.results]
            )
        ]['model']

        best_run = trials.results[
            np.argmin(
                [r['loss'] for r in trials.results]
            )
        ]['run_id']

        predicted_data = []
        latest_measures = len(x) - int(len(x) * 0.15)

        y_pred = best_model.predict(x.iloc[latest_measures:])

        for i, y in enumerate(y_pred):
            predicted_data.append(
                [
                    str(dados.index[latest_measures + i]),
                    y
                ]
            )

        mlflow.register_model(
            f"runs:/{best_run}/sklearn-model",
            "sklearn-k_nearest_neighbor-model",
            tags={"user_id": user_id}
        )

        return {
            "predictions": predicted_data
        }, 200

import mlflow
import pandas as pd
from sklearn import metrics
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from hyperopt import STATUS_OK

from mlflow.models.signature import infer_signature

INPUT_FOLDER = "data"
OUTPUT_FOLDER = "results"
SENSOR_DATA_FILE = 'sensor_data.csv'
HEALTH_RATE_FILE = "heart_rate.csv"


def preprocess_data(dados):

    dados_bpm = dados[['date', 'heart']]
    dados_bpm = dados_bpm.rename(columns={'heart': 'heart_rate'})
    dados_bpm = dados_bpm.rename(
        columns={'date': 'heart_rate_start_time'})
    dados_bpm = dados_bpm[dados_bpm['heart_rate'] != -1]

    dados_bpm['heart_rate_update_time'] = dados_bpm['heart_rate_start_time']
    dados_bpm['heart_rate_create_time'] = dados_bpm['heart_rate_start_time']
    dados_bpm['heart_rate_end_time'] = dados_bpm['heart_rate_start_time']

    dados_bpm['heart_rate_max'] = dados_bpm['heart_rate']
    dados_bpm['heart_rate_min'] = dados_bpm['heart_rate']

    dados_freq = dados_bpm.reset_index().drop(columns=['index'])
    dias_experimentos = dados_freq['heart_rate_start_time'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date()).unique()

    for dia in dias_experimentos:

        horarios_frequencias = []

        for row in dados_freq.iterrows():

            date_row = datetime.strptime(
                row[1]['heart_rate_start_time'],
                '%Y-%m-%d %H:%M:%S'
            ).date()

            if dia == date_row:
                horarios_frequencias.append(
                    (
                        row[0],
                        datetime.strptime(
                            row[1]['heart_rate_start_time'],
                            '%Y-%m-%d %H:%M:%S'
                        ).strftime('%H:%M:%S'),
                        row[1]['heart_rate']
                    )
                )

        horarios_frequencias.sort(key=lambda x: x[0])

        if len(horarios_frequencias) > 1:
            for previous, current in zip(
                horarios_frequencias,
                horarios_frequencias[1:]
            ):
                max_freq = max(previous[2], current[2])
                min_freq = min(previous[2], current[2])

                dados_freq['heart_rate'][previous[0]] = (
                    max_freq + min_freq) / 2

                dados_freq['heart_rate_max'][previous[0]] = max_freq
                dados_freq['heart_rate_min'][previous[0]] = min_freq

                curr_date = datetime.strptime(
                    dados_freq['heart_rate_start_time'][current[0]],
                    '%Y-%m-%d %H:%M:%S'
                )

                end_date = curr_date - timedelta(minutes=1)
                dados_freq['heart_rate_end_time'][previous[0]] = end_date

    dados_freq.sort_values(by=["heart_rate_start_time"])
    return dados_freq


def build_dataset(dados):

    dados.rename(
        columns={
            "heart_rate_start_time": "inicio",
            "heart_rate": "frequencia",
            "heart_rate_max": "maximo",
            "heart_rate_min": "minimo"
        },
        inplace=True
    )

    dados.drop(["heart_rate_update_time", "heart_rate_create_time",
               "heart_rate_end_time"], axis=1, inplace=True)

    dados["intervalo_min_max"] = dados.maximo - dados.minimo

    dados['aumento_frequencia'] = \
        dados['frequencia'] - dados['frequencia'].shift(-1)

    dados['aceleracao_frequencia'] = \
        dados['aumento_frequencia'] - dados['aumento_frequencia'].shift(-1)

    dados.inicio = pd.to_datetime(dados.inicio)

    dados.set_index("inicio", inplace=True)

    labels = ["minimo", "maximo", "aumento_frequencia", "intervalo_min_max"]

    x = dados[labels]
    y = dados["frequencia"]

    x.fillna(method='ffill', inplace=True)
    y.fillna(method='ffill', inplace=True)

    return x, y


def train_model(params):

    with mlflow.start_run(
        experiment_id=params["experiment_id"],
        nested=True
    ) as run:

        mlflow.set_tag("mlflow.user", params["user"])

        x, y = params['x'], params['y']

        # TODO: colocar test_size no hyperopt?
        test_size = 0.20

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            shuffle=False,
            random_state=7
        )

        mlflow.log_param("algorithm", params["algorithm"])
        mlflow.log_param("n_neighbors", params["n_neighbors"])
        mlflow.log_param("p", params["p"])
        mlflow.log_param("leaf_size", params["leaf_size"])
        mlflow.log_param("weights", params["weights"])

        knn = KNeighborsRegressor(
            algorithm=params['algorithm'],
            n_neighbors=params['n_neighbors'],
            p=params['p'],
            leaf_size=params['leaf_size'],
            weights=params['weights']
        )

        knn.fit(x_train, y_train)

        y_pred = knn.predict(x_test)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)

        signature = infer_signature(x_test, y_test)

        mlflow.sklearn.log_model(knn, "model", signature=signature)

    return {
        'status': STATUS_OK,
        'loss': mse,
        'model': knn,
        'run_id': run.info.run_id
    }

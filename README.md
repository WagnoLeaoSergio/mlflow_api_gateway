# mlflow_api_gateway Flask Application

Awesome mlflow_api_gateway created by WagnoLeaoSergio

## Installation

From source:

```bash
git clone https://github.com/WagnoLeaoSergio/mlflow_api_gateway mlflow_api_gateway
cd mlflow_api_gateway
make install
```

From pypi:

```bash
pip install mlflow_api_gateway
```

## Executing

This application has a CLI interface that extends the Flask CLI.

Just run:

```bash
$ mlflow_api_gateway
```

or

```bash
$ python -m mlflow_api_gateway
```

To see the help message and usage instructions.

## First run

```bash
mlflow_api_gateway create-db   # run once
mlflow_api_gateway populate-db  # run once (optional)
mlflow_api_gateway add-user -u admin -p 1234  # ads a user
mlflow_api_gateway run
```

Go to:

- Website: http://localhost:5000
- Admin: http://localhost:5000/admin/
  - user: admin, senha: 1234
- API GET:
  - http://localhost:5000/api/v1/**


> **Note**: You can also use `flask run` to run the application.

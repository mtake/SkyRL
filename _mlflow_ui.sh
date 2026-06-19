#!/usr/bin/env bash
set -x
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///$HOME/mlflow.db}
mlflow ui --host 0.0.0.0 --backend-store-uri $MLFLOW_TRACKING_URI

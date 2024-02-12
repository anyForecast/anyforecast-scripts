import click
import mlflow
import pandas as pd
from anyforecast_models.models import Seq2Seq
from anyforecast_models.pipelines import PreprocessorEstimatorPipeline
from anyforecast_models.preprocessing import make_preprocessor

from anyforecast_scripts.options import anyforecast_options


@click.command()
@anyforecast_options
def train(
    group_cols,
    datetime,
    target,
    time_varying_known,
    time_varying_unknown,
    static_categoricals,
    static_reals,
    max_prediction_length,
    max_encoder_length,
    freq,
    device,
    max_epochs,
    verbose,
    train,
):
    """Simple encoder-decoder arquitecture for time series forecasting.

    An additional embedding layer allows to condition the encoder module on
    static categorical data.
    """
    X = pd.read_csv(train)
    X[group_cols] = X[group_cols].astype("category")
    X[datetime] = pd.to_datetime(X[datetime])

    with mlflow.start_run():

        preprocessor = make_preprocessor(group_cols, datetime, target, freq)

        estimator = Seq2Seq(
            group_ids=group_cols,
            time_idx=datetime,
            target=target,
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            device=device,
            max_epochs=max_epochs,
            verbose=verbose,
        )

        pipe = PreprocessorEstimatorPipeline(
            preprocessor, estimator, inverse_steps=["datetime", "target"]
        )
        pipe.fit(X)

        mlflow.sklearn.log_model(pipe, "model")


if __name__ == "__main__":
    train()

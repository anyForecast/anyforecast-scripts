import click
import pandas as pd
from anyforecast_models.models import Seq2Seq
from anyforecast_models.pipelines import PreprocessorEstimatorPipeline
from anyforecast_models.preprocessing import make_preprocessor

from anyforecast_scripts.networks.options import network_options


@click.command()
@network_options
def train(
    filepath,
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
):
    X = pd.read_csv(filepath)
    X[group_cols] = X[group_cols].astype("category")
    X[datetime] = pd.to_datetime(X[datetime])

    preprocessor = make_preprocessor(group_cols, datetime, target, freq)
    Xt = preprocessor.fit_transform(X)

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


if __name__ == "__main__":
    train()

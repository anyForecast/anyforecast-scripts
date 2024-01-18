import click
import pandas as pd
from sklearn.pipeline import Pipeline
from skorch_forecasting.models import Seq2Seq

from anyforecast_scripts.cli_options import skorchforecasting_options
from anyforecast_scripts.preprocessing._preprocessing import make_preprocessor


@click.command()
@skorchforecasting_options
def train(
    train,
    group_ids,
    timestamp,
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
    data = pd.read_csv(train)
    data[group_ids] = data[group_ids].astype("category")
    data[timestamp] = pd.to_datetime(data[timestamp])

    preprocessor = make_preprocessor(group_ids, timestamp, target, freq)
    estimator = Seq2Seq(
        group_ids=group_ids,
        time_idx=timestamp,
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

    pipe = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    pipe.fit(data)


if __name__ == "__main__":
    train()

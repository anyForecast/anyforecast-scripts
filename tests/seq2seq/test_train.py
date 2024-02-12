from click.testing import CliRunner

from anyforecast_scripts.data import TimeseriesDataset, load_stallion
from anyforecast_scripts.seq2seq import train


def test_seq2seq_train():
    runner = CliRunner()

    stallion_ds: TimeseriesDataset = load_stallion()
    group_cols = ",".join(stallion_ds.group_cols)

    args = [
        "--group_cols",
        group_cols,
        "--datetime",
        stallion_ds.datetime,
        "--target",
        stallion_ds.target,
        "--time_varying_known",
        "price_regular,price_actual,discount",
        "--time_varying_unknown",
        stallion_ds.target,
        "--static_categoricals",
        group_cols,
        "--freq",
        stallion_ds.freq,
        "--train",
        stallion_ds.filepath,
    ]

    result = runner.invoke(train.train, args)
    assert result.exit_code == 0

from click.testing import CliRunner

from anyforecast_scripts.data import TimeseriesDataset, load_stallion
from anyforecast_scripts.networks.seq2seq import main


def test_seq2seq_train():
    runner = CliRunner()

    stallion_ds: TimeseriesDataset = load_stallion()
    group_cols = ",".join(stallion_ds.group_cols)

    args = [
        "--filepath",
        stallion_ds.filepath,
        "--group-cols",
        group_cols,
        "--datetime",
        stallion_ds.datetime,
        "--target",
        stallion_ds.target,
        "--time-varying-known",
        "price_regular,price_actual,discount",
        "--time-varying-unknown",
        stallion_ds.target,
        "--static-categoricals",
        group_cols,
        "--freq",
        stallion_ds.freq,
    ]

    result = runner.invoke(main.train, args)
    assert result.exit_code == 0

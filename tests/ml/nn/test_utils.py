from secretflow.ml.nn.fl.utils import History
from secretflow.ml.nn.metrics import Mean, Precision


def test_record_should_ok():
    # GIVEN
    mean = Mean(name='mean', total=10.0, count=2.0)
    acc = Precision('acc', [0.5], [10.0], [20.0])
    his = History()

    # WHEN
    his.record_global_history(metrics=[mean, acc])
    his.record_local_history('alice', metrics=[mean])

    # THEN
    assert his.global_history['mean'] == [mean.result()]
    assert his.global_history['acc'] == [acc.result().numpy()]
    assert (
        his.global_detailed_history['mean'][0].result().numpy() == mean.result().numpy()
    )

    assert (
        his.global_detailed_history['acc'][0].result().numpy() == acc.result().numpy()
    )

    assert his.local_history['alice']['mean'] == [mean.result().numpy()]
    assert (
        his.local_detailed_history['alice']['mean'][0].result().numpy()
        == mean.result().numpy()
    )

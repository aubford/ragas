import typing as t
from ragas.callbacks import new_group


def fbeta_score(tp, fp, fn, beta=1.0):
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta**2
    fbeta = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta


def new_conditional_group(
    *args,
    skip_tracing: bool = False,
    **kwargs,
) -> t.Tuple[t.Any, t.Any]:
    class MockRunManager:
        def on_chain_error(self, error, **kwargs):
            print(f"Error: {error}")

        def on_chain_end(self, outputs, **kwargs):
            pass

    if skip_tracing is True:
        return MockRunManager(), []
    else:
        return new_group(*args, **kwargs)

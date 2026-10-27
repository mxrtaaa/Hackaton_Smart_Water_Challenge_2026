import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

"""Metryki ewaluacji prognoz przepływu i zużycia energii.

Zera są poprawnymi wartościami i nie są usuwane.
Braki danych trzeba obsłużyć przed ewaluacją.
"""

mae = mean_absolute_error

__all__ = ["mae", "mase", "pinball_loss", "coverage"]


def _validate_1d(values, name):
    """Sprawdza poprawność jednowymiarowych danych liczbowych."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} musi zawierać wartości liczbowe."
        ) from exc

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} musi być niepustym wektorem jednowymiarowym."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} zawiera NaN lub nieskończoność."
        )

    return array


def _validate_pair(y_true, y_pred):
    """Sprawdza długość i zgodność indeksów danych."""
    if (
        isinstance(y_true, pd.Series)
        and isinstance(y_pred, pd.Series)
        and not y_true.index.equals(y_pred.index)
    ):
        raise ValueError(
            "Indeksy y_true i y_pred nie są identyczne."
        )

    actual = _validate_1d(y_true, "y_true")
    predicted = _validate_1d(y_pred, "y_pred")

    if len(actual) != len(predicted):
        raise ValueError(
            "y_true i y_pred muszą mieć tę samą długość."
        )

    return actual, predicted


def mase(y_true, y_pred, y_train, m=24):
    """Mean Absolute Scaled Error.

    y_true: rzeczywiste wartości testowe/walidacyjne
    y_pred: prognozy modelu
    y_train: historyczne wartości treningowe
    m: sezonowość w liczbie obserwacji
    """
    actual, predicted = _validate_pair(y_true, y_pred)
    train = _validate_1d(y_train, "y_train")

    if (
        isinstance(m, (bool, np.bool_))
        or not isinstance(m, (int, np.integer))
        or m < 1
    ):
        raise ValueError(
            "m musi być dodatnią liczbą całkowitą."
        )

    if len(train) <= m:
        raise ValueError(
            "Długość y_train musi być większa niż m."
        )

    # Sprawdzanie regularności szeregu z indeksem czasowym.
    if (
        isinstance(y_train, pd.Series)
        and isinstance(y_train.index, pd.DatetimeIndex)
    ):
        index = y_train.index

        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError(
                "Indeks czasowy musi być rosnący i bez duplikatów."
            )

        # Uwzględnia także regularne okresy kalendarzowe.
        if len(index) >= 3 and pd.infer_freq(index) is None:
            raise ValueError(
                "y_train ma nieregularny indeks czasowy. "
                "Sprawdź brakujące okresy; nie zamieniaj ich na zera."
            )

    # Błąd sezonowo naiwnej prognozy na treningu.
    naive_mae = mean_absolute_error(
        train[m:],
        train[:-m]
    )

    if naive_mae == 0:
        raise ValueError(
            "MASE jest niezdefiniowana: "
            "MAE prognozy naiwnej wynosi zero."
        )

    # Błąd modelu względem skali błędu baseline.
    model_mae = mean_absolute_error(actual, predicted)

    return float(model_mae / naive_mae)


def pinball_loss(y_true, y_pred, alpha=0.97):
    """Pinball Loss dla prognozy kwantylowej.

    Domyślnie alpha=0.97, czyli Q97.
    Niższy wynik oznacza lepszą prognozę kwantyla.
    """
    actual, predicted = _validate_pair(y_true, y_pred)

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha musi należeć do przedziału (0, 1)."
        )

    return float(
        mean_pinball_loss(
            actual,
            predicted,
            alpha=alpha
        )
    )


def coverage(y_true, y_pred):
    """Empiryczne pokrycie górnego kwantyla.

    Zwraca liczbę z zakresu [0, 1].
    Dla dobrze skalibrowanego Q97 oczekujemy około 0.97.
    """
    actual, predicted = _validate_pair(y_true, y_pred)

    return float(np.mean(actual <= predicted))

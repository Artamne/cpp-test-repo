"""Синтетические данные с ИЗВЕСТНОЙ ошибкой фазы — §7 задания.

Главный принцип: на реальной записи истинной ошибки фазы не знает никто,
поэтому проверять правильность там нельзя. Синтетика, куда ошибку вносим мы,
— единственное место, где есть с чем сравнивать.

Порядок функций — порядок выполнения:

    scene                   g_true: какая сцена   (§7.4)
    phase_error             phi_err: какая ошибка (§7.3)
    range_doppler_from_data h(k,n) с внесённой ошибкой

Связь истины и оценки. Данные строятся так, чтобы (5-3) при phi = 0 давало
расфокусированное изображение, а при phi = phi_err — ровно g_true:

    h(k,n) = ОПФ_азимут{ g_true } * e^{-j phi_err(k)}
    (5-3):  g = ПФ{ h e^{j phi} } = g_true   тогда и только тогда, когда
                                             phi = phi_err (с точностью до
                                             постоянной и линейной по k)

Значит ИСТИНА, с которой сравнивается результат этапа C, — это phi_err.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend import Backend

#: Виды сцены, §7.4 задания.
SCENES = ("points", "points_and_clutter", "clutter_only")

#: Виды вносимой ошибки фазы, §7.3 задания.
ERROR_KINDS = ("quadratic", "cubic", "jitter", "mixture")

#: Дрожание (§7.3) — это неучтённое ОТКЛОНЕНИЕ НОСИТЕЛЯ, а оно плавное:
#: несколько периодов на апертуру, а не белый шум от бина к бину. Диапазон
#: гармоник выбран так, чтобы отклонение было заведомо не описываемо
#: квадратичной моделью этапа B (её период на апертуре — один), но и не
#: вырождалось в шум, к которому никакой автофокус не применим.
JITTER_HARMONICS = (2, 3, 4, 5)

#: Яркость точечной цели относительно среднеквадратичного уровня фона
#: в сцене 'points_and_clutter'. Не выдумано: выбрано так, чтобы отклик
#: точки был на 40 дБ над фоном — обычный для РСА рабочий контраст, и
#: при этом порог решения №1 (-120 дБ) заведомо не срабатывает.
POINT_TO_CLUTTER_DB = 40.0


@dataclass
class Scene:
    """Истинное изображение и координаты внесённых точечных целей.

    Координаты нужны, чтобы замерить выигрыш по пику и ширину отклика по
    азимуту (§8 задания) именно там, куда точка была положена, а не там,
    где её потом кто-то нашёл."""

    image: np.ndarray
    kind: str
    points: list[tuple[int, int]] = field(default_factory=list)


def scene(kind: str, M: int, N: int, rng: np.random.Generator, n_points: int = 6) -> Scene:
    """Истинное сфокусированное изображение g_true — §7.4 задания.

    Принимает вид сцены, размеры (M, N) и генератор; возвращает Scene —
    комплексный массив (M, N) и список координат точечных целей.

      'points'              одиночные точечные цели: есть чему фокусироваться,
                            результат очевиден. Содержит ТОЧНЫЕ нули — именно
                            на этой сцене работает решение реализации №1.
      'points_and_clutter'  точки плюс распределённый фон: ближе к жизни.
      'clutter_only'        только фон, ни одной точки. ОБЯЗАТЕЛЬНАЯ сцена:
                            энтропийный критерий на однородном спекле может
                            не иметь выраженного минимума.

    Фон — полностью развитый спекл: комплексное гауссово поле, у которого
    |g|^2 распределено экспоненциально.
    """
    if kind not in SCENES:
        raise ValueError(f"неизвестная сцена {kind!r}; есть {SCENES}")

    g = np.zeros((M, N), dtype=np.complex128)
    points: list[tuple[int, int]] = []

    if kind in ("points", "points_and_clutter"):
        amplitude = 1.0 if kind == "points" else 10.0 ** (POINT_TO_CLUTTER_DB / 20.0)
        rows = rng.choice(M, size=n_points, replace=n_points > M)
        cols = rng.choice(N, size=n_points, replace=n_points > N)
        for m, n in zip(rows, cols):
            g[m, n] = amplitude * np.exp(2j * np.pi * rng.random())
            points.append((int(m), int(n)))

    if kind in ("points_and_clutter", "clutter_only"):
        g += (rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))) / np.sqrt(2.0)

    return Scene(image=g, kind=kind, points=points)


def phase_error(
    kind: str, M: int, edge_rad: float, rng: np.random.Generator, T_a: float = 1.0
) -> np.ndarray:
    """Вносимая ошибка фазы phi_err — §7.3 задания. Это ИСТИНА опыта.

    Принимает вид ошибки, длину M, размах на краю апертуры в радианах,
    генератор и время синтеза; возвращает вектор длины M.

      'quadratic'  квадратичная по f_a — расфокусировка, то, что этап B
                   и описывает
      'cubic'      кубическая — остаточная асимметрия
      'jitter'     дрожание — неучтённое отклонение носителя: плавная
                   случайная добавка из гармоник JITTER_HARMONICS
      'mixture'    смесь — живой случай

    Размах задаётся значением на краю апертуры (|f_a| максимально), потому
    что именно край определяет расфокусировку. Для дрожания размах — это
    среднеквадратичное значение по апертуре.
    """
    if kind not in ERROR_KINDS:
        raise ValueError(f"неизвестный вид ошибки {kind!r}; есть {ERROR_KINDS}")

    f_a = np.fft.fftfreq(M, d=T_a / M)
    u = f_a / np.abs(f_a).max()  # нормированная частота, край = +-1

    if kind == "quadratic":
        return edge_rad * u**2
    if kind == "cubic":
        return edge_rad * u**3
    if kind == "jitter":
        return edge_rad * _smooth_deviation(u, rng)
    # 'mixture': квадратичная основа, кубическая асимметрия и дрожание
    return edge_rad * (0.6 * u**2 + 0.3 * u**3 + 0.3 * _smooth_deviation(u, rng))


def _smooth_deviation(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Плавное отклонение носителя единичного СКО: сумма гармоник
    JITTER_HARMONICS по апертуре со случайными амплитудами и фазами.

    Принимает нормированную частоту u (край апертуры = +-1) и генератор;
    возвращает вектор той же длины со среднеквадратичным значением 1.
    """
    deviation = np.zeros_like(u)
    for harmonic in JITTER_HARMONICS:
        deviation += rng.normal() * np.sin(np.pi * harmonic * (u + 1.0) / 2.0 + 2 * np.pi * rng.random())
    return deviation / deviation.std()


def range_doppler_from_scene(
    backend: Backend, g_true: np.ndarray, phi_err: np.ndarray
) -> np.ndarray:
    """Данные h(k,n) в области дальность-доплер с внесённой ошибкой фазы.

    Принимает бэкенд, истинное изображение g_true (M, N) и истинную ошибку
    phi_err длины M; возвращает комплексный массив h(k,n) размера (M, N).

    Обращение (5-3): ядро (5-3) — это прямое ПФ с множителем alpha, поэтому
    обратное к нему несёт множитель 1/(alpha M). Ошибка вносится множителем
    e^{-j phi_err}, чтобы (5-3) при phi = phi_err вернуло ровно g_true.

    Массив h ФИКСИРОВАН на всё время работы алгоритма и не пересчитывается
    (§2 документа).
    """
    g = backend.asarray(g_true)
    h_ideal = backend.xp.fft.ifft(g, axis=0) / backend.alpha
    return h_ideal * backend.xp.exp(-1j * backend.asarray(phi_err))[:, None]

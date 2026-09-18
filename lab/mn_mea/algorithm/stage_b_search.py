# -*- coding: utf-8 -*-
"""Этап B без ИНС: начальная фаза phi^(0) поиском полинома по энтропии сцены.

Книжный этап B (stage_b_initial.py) строит phi^(0) из параметров движения
носителя (4-83), (4-84), (5-27)…(5-30). Когда ИНС нет, параметров нет, и
этап B в run_real.py не делался: этап C шёл из нуля. На настоящей записи
это не работает, и вот почему — замер на срезе владельца с ЗАВЕДОМО
внесённой квадратичной фазой 3 рад (8 рад на краю полосы):

    шаг Ньютона (5-8) из нуля      0,2…1,2 % от истины по модулю
    косинус шага с истиной         от -0,6 до +0,6, в среднем около нуля
    после 200 итераций             остаток 0,96…1,25 от внесённого

Пофазный градиент (5-13) в блоке из 273 x 1335 отсчётов клаттера тонет в
шуме: у каждого доплеровского бина свой случайный вклад от спекла, и
осмысленную часть градиента — общую для всех бинов — за ним не видно. На
подделке с яркими точками на тёмном фоне того же не было, потому что там
шума в градиенте нет.

Полином же ищется по ВСЕЙ сцене и всего с несколькими коэффициентами:
шум по бинам усредняется, и энтропия как функция коэффициента гладкая с
одним минимумом (замер: двумерный скан по квадратичной и кубической на
срезе владельца — одна впадина). На той же записи с внесённой ошибкой
поиск нашёл -6,5 при внесённых -3 и собственном остатке сцены -3.

Это НЕ замена этапа C и НЕ улучшение формул книги: найденный полином идёт
в этап C как phi^(0), ровно на место, которое книга отводит этапу B. Этап C
из этой точки снимает то, что полиномом не описывается.

Порядок функций — порядок выполнения:

    band_axis          нормированная частота u: 0 вне полосы, +-1 на её краю
    legendre           многочлен Лежандра P_d(u)
    polynomial_phase   phi = sum c_d P_d(u) (R/R_B0)^(d-1) по степеням DEGREES
    range_scale_axis   R_n / R_B0 по стробам
    scene_entropy      (5-6) по всей сцене при данной phi — одно БПФ
    refine_coefficient один коэффициент: перебор с шагом, потом парабола
    search             широкий скан квадратичной, сетка по двум степеням, уточнение
    segment_bounds     границы сегментов по азимуту
    segment_data       сегмент сцены с полями в области дальность-доплер
    search_segments    своя (P2, P3) на каждый сегмент
    coefficients_at    поправка в строке m — интерполяция по центрам сегментов
    apply_segments     исправленная сцена: перекрытие с сохранением, одна поправка на плитку
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import stage_c_iterate as C
from backend import Backend

#: Степени многочленов Лежандра P_d(u). Первая (линейная) не ищется: она
#: лишь сдвигает картинку по азимуту, энтропии не меняет и снимается в
#: этапе D (remove_image_shift). Базис именно Лежандра, а не степени u:
#: u^2 и u^4 на отрезке [-1, 1] почти коллинеарны, и покоординатный спуск
#: по ним ползёт — замер: u^4 уходил в +12,8, пока u^2 не дошёл до
#: своего минимума. Многочлены Лежандра на этом отрезке ортогональны.
#:
#: Только вторая и третья степень — ошибка скорости (квадратичная) и
#: ускорения (кубическая), то, что даёт полёт. Четвёртая и пятая
#: проверены и ОТБРОШЕНЫ: на срезе владельца они опускают энтропию ещё на
#: 0,005 из 0,040, но с заведомо внесённой чисто квадратичной ошибкой
#: коэффициент при P_4 менялся на 2,9 между чистыми и испорченными
#: данными — впадина по нему плоская, и он ловит шум. Что выше третьей
#: степени, снимает этап C по бинам.
DEGREES = (2, 3)

#: Начальный шаг перебора коэффициента, радианы (коэффициент при P_d;
#: |P_d| <= 1, так что это и размах на краю полосы). Выбран по замеру:
#: остатки на срезе владельца 8…14 рад, впадина энтропии шириной около
#: 10 рад; шаг 4 с пятью точками накрывает +-8 и не перескакивает впадину,
#: а расширение перебора (см. refine_coefficient) достаёт и дальше.
STEP_INITIAL_RAD = 4.0

#: Кругов покоординатного спуска; шаг каждый круг делится пополам. Четыре
#: круга дают точность около шага/8 = 0,5 рад — дальше доводит этап C.
ROUNDS = 4

#: Точек перебора вокруг текущего значения: -2, -1, 0, +1, +2 шага.
PROBE_OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)

#: Полуширина грубой сетки по каждому коэффициенту, в шагах STEP_INITIAL_RAD:
#: сетка накрывает +-GRID_HALF_STEPS*шаг = +-16 рад. Сетка идёт ПЕРЕД
#: покоординатным спуском: впадина энтропии на клаттере мелкая (0,04 при
#: 13,4) и с несколькими ложбинами, и спуск по одной координате застревает
#: — замер на срезе владельца: спуск нашёл dS -0,017 там, где сетка
#: показывала -0,036. Сетка по двум коэффициентам — 81 оценка.
GRID_HALF_STEPS = 4

#: Сколько раз перебор может сдвинуться за край, если минимум упёрся в
#: крайнюю точку. Без этого поиск не достаёт ошибку больше 2*шаг*ROUNDS:
#: замер с внесёнными 8 рад поверх собственных 10 — нашёл 14 из 18.
EXTEND_LIMIT = 6

#: Широкий предварительный скан ОДНОЙ квадратичной: полуширина в радианах
#: и шаг. Ошибка скорости продукта в 10…15 % даёт квадратичную фазу в
#: сотни радиан (при K_a = 38 Гц/с и T_a = 3,8 с полная фаза апертуры
#: 436 рад, её десятая часть — 44), и сетка 9 x 9 с шагом 4 её не достаёт.
#: Шаг 8: впадина энтропии при таких ошибках широкая, замер на срезе
#: владельца — от -40 до +40 рад энтропия гладкая, одна впадина.
PRESCAN_HALF_RAD = 320.0
PRESCAN_STEP_RAD = 8.0

#: Прореживание по дальности для предварительного скана: каждый такой
#: строб. Энтропии как критерию хватает и четверти стробов — она среднее
#: по миллионам пикселей, — а скан от этого вчетверо быстрее. Сетка и
#: уточнение идут по всем стробам.
PRESCAN_RANGE_STRIDE = 4

#: Показатель степени дальности при коэффициенте d-й степени: фаза
#: масштабируется как (R / R_B0) ** (d - 1). Вывод для ошибки скорости:
#: остаток phi(f_d) = pi f_d^2 (1/K_ист - 1/K_взят), K = 2 v^2 / (lambda R),
#: то есть при данной доплеровской частоте phi ~ R — квадратичная растёт
#: с дальностью линейно (d - 1 = 1). Для ошибки продольного ускорения
#: dR = v da t^3 / (2 R), t = -lambda R f_d / (2 v^2), откуда phi ~ R^2
#: (d - 1 = 2). Без масштабирования одна поправка на срез в 1700 м по
#: дальности (R от 2650 до 4350) ошибается на +-25 % по краям.
RANGE_POWER_OFFSET = 1


@dataclass
class SearchResult:
    """Что нашёл поиск и по каким числам."""

    phi: np.ndarray                              # phi^(0): длина M, или (M, N) с дальностью
    coefficients: dict[int, float]               # степень -> коэффициент, рад на краю
    entropy_start: float                         # (5-6) при phi = 0
    entropy_final: float                         # (5-6) при найденной phi
    n_evaluations: int = 0                       # сколько раз считалась энтропия
    history: list[tuple[int, int, float, float]] = field(default_factory=list)
    band_edge: float = 0.0                       # край полосы, доли частоты повторения
    n_band_bins: int = 0


def band_axis(backend: Backend, h, floor_factor: float = C.SIGNAL_FLOOR_FACTOR):
    """Нормированная доплеровская частота u для полинома.

    Принимает бэкенд, данные сцены h(k,n) и порог решения №7; возвращает
    (u, край полосы, число бинов в полосе). u = f / f_края внутри полосы,
    ноль вне её: там сигнала нет, и фаза там ничего не значит, а полином,
    продолженный за край, уходил бы в десятки радиан и вносил бы в пустые
    бины бессмысленную фазу (проверено пробой в run_real: «снято 1 %»).

    Край берётся по тем же живым бинам, что и решение №7, — одно
    определение полосы на весь конвейер.
    """
    h_abs2 = backend.xp.abs(backend.asarray(h)) ** 2
    live = backend.to_numpy(C.signal_bins(backend, h_abs2, floor_factor)).astype(bool)
    M = live.size
    f = np.fft.fftfreq(M)
    edge = float(np.abs(f[live]).max()) if live.any() else 0.5
    u = np.where(live, f / edge, 0.0)
    return u, edge, int(live.sum())


def legendre(u: np.ndarray, degree: int) -> np.ndarray:
    """Многочлен Лежандра P_d(u) по рекуррентной формуле Бонне."""
    p_prev, p = np.ones_like(u), u.copy()
    if degree == 0:
        return p_prev
    for d in range(1, degree):
        p_prev, p = p, ((2 * d + 1) * u * p - d * p_prev) / (d + 1)
    return p


def polynomial_phase(u: np.ndarray, coefficients: dict[int, float],
                     range_scale: np.ndarray | None = None) -> np.ndarray:
    """phi(u) = sum_d c_d P_d(u), при range_scale — ещё и по дальности.

    Принимает ось u, словарь степень -> коэффициент и, если есть, вектор
    R_n / R_B0 по стробам; возвращает фазу длины M (без дальности) или
    массив (M, N): phi(k, n) = sum_d c_d P_d(u_k) (R_n / R_B0)^(d - 1),
    см. RANGE_POWER_OFFSET. Коэффициент c_d — фаза на краю полосы В ЦЕНТРЕ
    среза по дальности, R = R_B0.

    Вне полосы u = 0, и P_d(0) для чётных d НЕ ноль — поэтому фаза вне
    полосы обнуляется явно: там сигнала нет, и фаза там ничего не значит.
    """
    if range_scale is None:
        phi = np.zeros_like(u)
        for degree, c in coefficients.items():
            if c != 0.0:
                phi += c * legendre(u, degree)
        phi[u == 0.0] = 0.0
        return phi
    phi = np.zeros((u.size, range_scale.size), dtype=np.float64)
    for degree, c in coefficients.items():
        if c != 0.0:
            phi += (c * legendre(u, degree))[:, None] * (
                range_scale[None, :] ** (degree - RANGE_POWER_OFFSET))
    phi[u == 0.0, :] = 0.0
    return phi


def range_scale_axis(R_B0: float, r_b: float, N: int) -> np.ndarray:
    """R_n / R_B0 по стробам: R_B0 в середине среза, шаг r_b метров на строб."""
    return (R_B0 + (np.arange(N) - N / 2.0) * r_b) / R_B0


def scene_entropy(backend: Backend, h, phi, S_g: float, floor: float) -> float:
    """(5-6) всей сцены при фазе phi — одно азимутальное БПФ.

    Принимает бэкенд, данные сцены на устройстве, фазу, энергию (5-4) и
    порог решения №1; возвращает нормированную энтропию S.

    Считается теми же функциями, что и в этапе C, — это то же самое
    число, которое этап C будет минимизировать, только по всей сцене
    и без маски.
    """
    phi = backend.asarray(phi)
    if phi.ndim == 1:
        _, g = C.image_from_phase(backend, h, phi)
    else:  # фаза зависит и от дальности: (5-3) построчно, множитель (M, N)
        g = backend.fft_kernel_minus(h * backend.xp.exp(1j * phi))
    P, _, _ = C.image_power(backend, g, floor)
    return C.entropy(backend, P, S_g, None)[1]


def refine_coefficient(evaluate, coefficients: dict[int, float], degree: int,
                       step: float) -> tuple[float, float]:
    """Уточнить один коэффициент: перебор PROBE_OFFSETS, затем парабола.

    Принимает функцию S(коэффициенты), текущие коэффициенты, степень и
    шаг; возвращает (новое значение, S в нём). Коэффициенты меняются на
    месте.

    Парабола строится по лучшей точке перебора и двум её соседям, и
    берётся только если её вершина лежит между соседями и энтропия там
    действительно ниже: вершина параболы вне отрезка — не минимум, а
    экстраполяция. Если лучшая точка — крайняя, перебор продолжается в ту
    сторону (EXTEND_LIMIT), иначе поиск не достаёт больших ошибок.
    """
    c0 = coefficients[degree]
    probes = []
    for off in PROBE_OFFSETS:
        coefficients[degree] = c0 + off * step
        probes.append((evaluate(coefficients), coefficients[degree]))
    best = min(range(len(probes)), key=lambda i: probes[i][0])
    # минимум на краю перебора — впадина дальше, чем достаёт сетка; идём
    # туда шаг за шагом, пока не перевалим через минимум или не упрёмся
    # в EXTEND_LIMIT
    extended = 0
    while best in (0, len(probes) - 1) and extended < EXTEND_LIMIT:
        direction = -1.0 if best == 0 else 1.0
        coefficients[degree] = probes[best][1] + direction * step
        probe = (evaluate(coefficients), coefficients[degree])
        if best == 0:
            probes.insert(0, probe)
        else:
            probes.append(probe)
            best = len(probes) - 1
        best = min(range(len(probes)), key=lambda i: probes[i][0])
        extended += 1
    S_best, c_best = probes[best]
    if 0 < best < len(probes) - 1:
        (S_l, c_l), (S_r, c_r) = probes[best - 1], probes[best + 1]
        denominator = S_l - 2.0 * S_best + S_r
        if denominator > 0.0:
            c_v = c_best + 0.5 * step * (S_l - S_r) / denominator
            if c_l < c_v < c_r:
                coefficients[degree] = c_v
                S_v = evaluate(coefficients)
                if S_v < S_best:
                    return c_v, S_v
    coefficients[degree] = c_best
    return c_best, S_best


def search(backend: Backend, h, degrees=DEGREES, step_initial: float = STEP_INITIAL_RAD,
           rounds: int = ROUNDS, floor_factor: float = C.SIGNAL_FLOOR_FACTOR,
           floor_relative: float = C.POWER_FLOOR_RELATIVE,
           range_scale: np.ndarray | None = None,
           prescan_half: float = PRESCAN_HALF_RAD,
           prescan_step: float = PRESCAN_STEP_RAD,
           cubic_prescan_half: float = 0.0,
           grid_half_steps: int = GRID_HALF_STEPS,
           band_override: np.ndarray | None = None) -> SearchResult:
    """Поиск полинома по энтропии сцены: широкий скан, сетка, уточнение.

    Принимает бэкенд, данные сцены h(k,n) и, если есть, вектор R_n / R_B0
    по стробам (range_scale_axis); возвращает SearchResult с phi^(0):
    вектор длины M без дальности или массив (M, N) с ней.

    Три ступени:
      1. широкий скан одной квадратичной, +-prescan_half с шагом
         prescan_step, по каждому PRESCAN_RANGE_STRIDE-му стробу — чтобы
         достать ошибку скорости в сотни радиан;
      2. грубая сетка по двум первым степеням вокруг найденного
         (GRID_HALF_STEPS шагов step_initial);
      3. покоординатное уточнение, rounds кругов, шаг делится пополам.

    Стоимость: одно БПФ сцены на оценку; скан 81 (на четверти стробов),
    сетка 81, уточнение около 50. У этапа C на 22 блоках по 200 итераций
    БПФ 8800.
    """
    h_device = backend.asarray(h)
    M, N = h_device.shape
    u, edge, n_band = band_axis(backend, h_device, floor_factor)
    if band_override is not None:  # сегмент: полоса задана по всей сцене
        u = band_override
        n_band = int(np.count_nonzero(u))
    h_abs2 = backend.xp.abs(h_device) ** 2
    S_g = C.total_energy(backend, h_abs2, M)
    floor = C.power_floor(S_g, M, N, floor_relative)

    counter = {"n": 0}

    def evaluate(coefficients: dict[int, float]) -> float:
        counter["n"] += 1
        return scene_entropy(backend, h_device,
                             polynomial_phase(u, coefficients, range_scale),
                             S_g, floor)

    # прореженная по дальности сцена для широкого скана: свои S_g и порог
    stride = PRESCAN_RANGE_STRIDE
    h_thin = backend.asarray(h_device[:, ::stride])
    scale_thin = None if range_scale is None else range_scale[::stride]
    S_g_thin = C.total_energy(backend, backend.xp.abs(h_thin) ** 2, M)
    floor_thin = C.power_floor(S_g_thin, M, h_thin.shape[1], floor_relative)

    def evaluate_thin(coefficients: dict[int, float]) -> float:
        counter["n"] += 1
        return scene_entropy(backend, h_thin,
                             polynomial_phase(u, coefficients, scale_thin),
                             S_g_thin, floor_thin)

    coefficients = {d: 0.0 for d in degrees}
    S_start = evaluate(coefficients)
    history = []

    # ступень 1: широкий скан квадратичной. Минимум ищется на прореженной
    # сцене; берётся только положение, энтропия потом пересчитывается на всей
    d_a = degrees[0]
    if prescan_half > 0.0:
        best_thin, best_c = None, 0.0
        n_steps = int(round(prescan_half / prescan_step))
        for k in range(-n_steps, n_steps + 1):
            coefficients[d_a] = k * prescan_step
            S_thin = evaluate_thin(coefficients)
            if best_thin is None or S_thin < best_thin:
                best_thin, best_c = S_thin, coefficients[d_a]
        coefficients[d_a] = best_c
        history.append((-1, d_a, best_c, float(best_thin)))

    # ступень 1б: то же для кубической, если просили (сегменты края кадра)
    if cubic_prescan_half > 0.0 and len(degrees) > 1:
        d_c = degrees[1]
        best_thin, best_c = None, 0.0
        n_steps = int(round(cubic_prescan_half / prescan_step))
        for k in range(-n_steps, n_steps + 1):
            coefficients[d_c] = k * prescan_step
            S_thin = evaluate_thin(coefficients)
            if best_thin is None or S_thin < best_thin:
                best_thin, best_c = S_thin, coefficients[d_c]
        coefficients[d_c] = best_c
        history.append((-1, d_c, best_c, float(best_thin)))

    # ступень 2: грубая сетка по ПЕРВЫМ ДВУМ степеням вокруг найденного
    # (остальные в нуле): ищет ложбину, в которой потом уточняться
    grid = [k * step_initial for k in range(-grid_half_steps, grid_half_steps + 1)]
    centre_a = coefficients[d_a]
    d_b = degrees[1] if len(degrees) > 1 else degrees[0]
    centre_b = coefficients[d_b] if d_b != d_a else 0.0
    S_now, best = evaluate(coefficients), dict(coefficients)
    for c_a in grid:
        for c_b in (grid if d_b != d_a else [0.0]):
            coefficients[d_a], coefficients[d_b] = centre_a + c_a, centre_b + c_b
            S_here = evaluate(coefficients)
            if S_here < S_now:
                S_now, best = S_here, dict(coefficients)
    coefficients = best
    history.append((0, d_a, coefficients[d_a], S_now))
    history.append((0, d_b, coefficients[d_b], S_now))

    # ступень 3: покоординатный спуск с убывающим шагом, от шага сетки
    step = step_initial
    for round_index in range(rounds):
        for degree in degrees:
            _, S_now = refine_coefficient(evaluate, coefficients, degree, step)
            history.append((round_index + 1, degree, coefficients[degree], S_now))
        step /= 2.0

    return SearchResult(
        phi=polynomial_phase(u, coefficients, range_scale),
        coefficients=dict(coefficients),
        entropy_start=S_start,
        entropy_final=S_now,
        n_evaluations=counter["n"],
        history=history,
        band_edge=edge,
        n_band_bins=n_band,
    )


# ------------------------------------------------------------- вдоль азимута

#: Длина сегмента по азимуту для посегментной поправки, строк. Сегмент
#: обязан быть длиннее размаза: при a рад квадратичной на краю полосы
#: точка размазана на 2a/(pi*край) отсчётов — при a = 300 и крае 0,12 это
#: 1600 строк. 2048 с нулевыми полями по половине с каждой стороны
#: (SEGMENT_PAD_FRACTION) накрывают такой размаз без кругового наложения.
#: Ошибка внутри сегмента считается одной: замер профиля на крае кадра
#: владельца — 120, 140, 220, 540, 630, 930 рад по окнам через 768 строк,
#: то есть за 2048 строк она меняется до двух раз; остаток от этого
#: снимают блоки этапа C.
SEGMENT_ROWS = 2048

#: Сдвиг между сегментами — четверть длины. При половине соседние
#: поправки на крае кадра (150 и 510 рад через 1024 строки) смешивались
#: окнами sin^2 в широкой зоне, и на картинке шли горизонтальные полосы
#: по стыкам. При четверти в каждой строке складываются четыре сегмента,
#: поправка вдоль азимута меняется плавнее; сумма окон нормируется явно.
SEGMENT_HOP = SEGMENT_ROWS // 4

#: Нули с каждой стороны сегмента ПРИ ПРИМЕНЕНИИ поправки, в долях длины,
#: против кругового наложения.
SEGMENT_PAD_FRACTION = 0.5

#: Нули при ПОИСКЕ — ноль, сегмент кольцевой. Замер на крае кадра
#: владельца, скан квадратичной ±1200 рад:
#:
#:   сегмент       с полями 0,5     без полей
#:   2048…4096     0 (dS 0)         +510 (dS -0,085)
#:   3072…5120     0 (dS 0)         +690 (dS -0,113)
#:   2048…6144     0 (dS 0)         +570 (dS -0,045)
#:
#: При размазе в +-1400…1800 строк поправка выталкивает энергию краёв
#: сегмента в поля, энтропия буфера от этого растёт и прячет впадину. В
#: кольцевом окне энергия остаётся внутри, и впадина видна. Применять
#: поправку кольцевым окном нельзя — края наложатся, — поэтому поиск и
#: применение идут с разными полями.
SEGMENT_SEARCH_PAD_FRACTION = 0.0

#: Сегмент, у которого полоса -10 дБ уже этой доли от самой широкой по
#: сцене, не ищется и не правится: там апертура неполная (край кадра),
#: поправки нет по построению, а поиск на обрезанной полосе вернул бы шум.
SEGMENT_BAND_MIN_FRACTION = 0.6

#: Что делать с поправкой ТАМ, где её нельзя измерить (полоса урезана):
#: 'hold' — держать последнее надёжное значение, 'zero' — сводить к нулю.
#: Физически ошибка опорной функции у края кадра никуда не девается —
#: теряется лишь возможность её измерить, — поэтому по умолчанию 'hold'.
#: Выбор замерен на крае кадра владельца, см. coefficients_at.
SEGMENT_EDGE_POLICY = "hold"

#: Широкий скан по сегменту: полуширина и шаг для квадратичной и для
#: кубической. Край кадра владельца — до 930 рад квадратичной.
SEGMENT_PRESCAN_HALF_RAD = 1200.0
SEGMENT_PRESCAN_STEP_RAD = 15.0
SEGMENT_CUBIC_HALF_RAD = 300.0

#: Сетка вокруг найденного при посегментном поиске: +-2 шага, 5 x 5.
#: После скана с шагом 15 этого хватает, а 9 x 9 на семи сегментах
#: обошлась бы в 570 БПФ сегмента.
SEGMENT_GRID_HALF_STEPS = 2


@dataclass
class SegmentResult:
    """Поправка одного сегмента."""

    row_start: int
    row_stop: int
    coefficients: dict[int, float]
    entropy_start: float
    entropy_final: float
    n_evaluations: int


def segment_bounds(M: int, rows: int = SEGMENT_ROWS, hop: int = SEGMENT_HOP) -> list[tuple[int, int]]:
    """Границы сегментов: длина rows, шаг hop, последний прижат к концу.

    Принимает число строк сцены; возвращает список (start, stop). Сцена
    короче сегмента — один сегмент на всю.
    """
    if M <= rows:
        return [(0, M)]
    starts = list(range(0, M - rows + 1, hop))
    if starts[-1] + rows < M:
        starts.append(M - rows)
    return [(s, s + rows) for s in starts]


def segment_data(backend: Backend, g_scene, start: int, stop: int,
                 pad_fraction: float = SEGMENT_PAD_FRACTION):
    """Сегмент сцены с нулевыми полями, в области дальность-доплер.

    Принимает бэкенд, изображение сцены и границы строк; возвращает
    (h_seg, pad) — данные длины (stop - start) + 2 pad и ширину поля.
    """
    rows = stop - start
    pad = int(round(pad_fraction * rows))
    buffer = backend.xp.zeros((rows + 2 * pad, g_scene.shape[1]), dtype=g_scene.dtype)
    buffer[pad : pad + rows] = g_scene[start:stop]
    return backend.xp.fft.ifft(buffer, axis=0) / backend.alpha, pad


def search_segments(backend: Backend, h, range_scale: np.ndarray | None = None,
                    rows: int = SEGMENT_ROWS, hop: int = SEGMENT_HOP,
                    floor_factor: float = C.SIGNAL_FLOOR_FACTOR,
                    floor_relative: float = C.POWER_FLOOR_RELATIVE,
                    prescan_half: float = SEGMENT_PRESCAN_HALF_RAD,
                    cubic_half: float = SEGMENT_CUBIC_HALF_RAD,
                    degrees=DEGREES) -> list[SegmentResult]:
    """Посегментный поиск полинома: своя (P2, P3) на каждый сегмент азимута.

    Принимает бэкенд, данные сцены h(k,n) и масштаб дальности; возвращает
    список SegmentResult по сегментам (segment_bounds).

    Зачем не одна фаза на сцену: на крае кадра владельца квадратичная
    ошибка растёт вдоль азимута от 120 до 930 рад — одна поправка на
    сцену подошла бы одному месту и испортила бы остальные. Полоса
    доплера (край u = 1) берётся по ВСЕЙ сцене, одна для всех сегментов:
    у сегмента она та же, только сетка реже.

    Каждый сегмент: широкий скан квадратичной (SEGMENT_PRESCAN_*), скан
    кубической (SEGMENT_CUBIC_HALF_RAD), сетка 5 x 5, уточнение — то же
    search(), только с полями и на сетке сегмента.
    """
    h_device = backend.asarray(h)
    M, N = h_device.shape
    _, edge, _ = band_axis(backend, h_device, floor_factor)
    g_scene = backend.fft_kernel_minus(h_device)
    bounds = segment_bounds(M, rows, hop)
    widths = [segment_band_width(backend, g_scene, s, e)[0] for s, e in bounds]
    widest = max(widths) if widths else 1.0
    results = []
    for (start, stop), width in zip(bounds, widths):
        if width < SEGMENT_BAND_MIN_FRACTION * widest:
            results.append(SegmentResult(start, stop, {2: 0.0, 3: 0.0}, 0.0, 0.0, 0))
            continue
        h_seg, pad = segment_data(backend, g_scene, start, stop,
                                  SEGMENT_SEARCH_PAD_FRACTION)
        L = h_seg.shape[0]
        f_seg = np.fft.fftfreq(L)
        u_seg = np.where(np.abs(f_seg) <= edge, f_seg / edge, 0.0)
        result = search(backend, h_seg, degrees=degrees, range_scale=range_scale,
                        prescan_half=prescan_half,
                        prescan_step=SEGMENT_PRESCAN_STEP_RAD,
                        cubic_prescan_half=cubic_half,
                        grid_half_steps=SEGMENT_GRID_HALF_STEPS,
                        band_override=u_seg)
        coefficients = {d: result.coefficients.get(d, 0.0) for d in DEGREES}
        results.append(SegmentResult(start, stop, coefficients,
                                     result.entropy_start, result.entropy_final,
                                     result.n_evaluations))
    return results


#: Плитка выдачи при применении поправки, строк. Поправка кусочно-постоянна
#: по плитке: на самом крутом участке края кадра (577 -> 767 рад за 1024
#: строки) соседние плитки по 128 строк отличаются на 24 рад — цель на
#: стыке видит скачок в 24 рад между половинами, это терпимо.
TILE_ROWS = 128

#: Окно вокруг плитки при применении, строк: плитка берётся из его середины,
#: и края окна (где поправка портит содержимое) до неё не достают, пока
#: размаз меньше половины окна минус полплитки — при 4096 это +-1980 строк,
#: то есть до ~770 рад квадратичной на этой записи.
APPLY_WINDOW_ROWS = 4096


def coefficients_at(segments: list[SegmentResult], m: float,
                    edge_policy: str = SEGMENT_EDGE_POLICY) -> dict[int, float]:
    """Коэффициенты поправки в строке m — линейная интерполяция по центрам сегментов.

    Принимает список поправок сегментов и строку; возвращает словарь
    степень -> коэффициент. За крайними центрами — постоянно.

    Сегменты, где поиск не делался (полоса урезана, n_evaluations = 0),
    при edge_policy 'hold' в интерполяцию не входят: поправка за последним
    измеренным центром держится постоянной. При 'zero' они участвуют
    нулями, и поправка сходит к нулю за полсегмента.
    """
    if edge_policy == "hold":
        used = [s for s in segments if s.n_evaluations > 0] or segments
    else:
        used = segments
    centres = np.array([(s.row_start + s.row_stop) / 2.0 for s in used])
    order = np.argsort(centres)
    degrees = sorted({d for s in used for d in s.coefficients})
    return {d: float(np.interp(m, centres[order],
                               np.array([used[i].coefficients.get(d, 0.0) for i in order])))
            for d in degrees}


def apply_segments(backend: Backend, h, segments: list[SegmentResult],
                   range_scale: np.ndarray | None = None,
                   floor_factor: float = C.SIGNAL_FLOOR_FACTOR,
                   tile_rows: int = TILE_ROWS, window_rows: int = APPLY_WINDOW_ROWS,
                   edge_policy: str = SEGMENT_EDGE_POLICY):
    """Исправленная сцена: перекрытие с сохранением, одна поправка на плитку.

    Принимает бэкенд, данные сцены и список поправок; возвращает
    изображение сцены (M, N) на устройстве, готовое к нарезке на блоки.

    Для каждой плитки в tile_rows строк берётся окно window_rows строк
    вокруг неё (с нулевыми полями SEGMENT_PAD_FRACTION), исправляется
    ОДНОЙ поправкой — коэффициентами, интерполированными на центр плитки,
    — и в сцену идёт только плитка из середины окна.

    ПОЧЕМУ НЕ СКЛЕЙКА ОКНАМИ. Первая версия складывала сегменты, каждый
    со своей поправкой, с весами sin^2. Замер на крае кадра: цели в
    строках 1368…1798 после такой склейки стали ШИРЕ (8,0 -> 20 элементов
    по -20 дБ), хотя одна поправка на всю сцену давала там 4,0. В одной
    строке складывались четыре версии цели, исправленные фазами 125, 150,
    174 и 263 рад, — комплексные картинки с разной остаточной фазой
    интерферируют, и сумма резкой не бывает. Здесь на строку приходится
    ровно одна поправка, складывать нечего.
    """
    h_device = backend.asarray(h)
    M, N = h_device.shape
    _, edge, _ = band_axis(backend, h_device, floor_factor)
    g_scene = backend.fft_kernel_minus(h_device)
    xp = backend.xp
    out = xp.zeros_like(g_scene)
    for t0 in range(0, M, tile_rows):
        t1 = min(t0 + tile_rows, M)
        coefficients = coefficients_at(segments, (t0 + t1) / 2.0, edge_policy)
        if all(abs(c) < 1e-9 for c in coefficients.values()):
            out[t0:t1] = g_scene[t0:t1]
            continue
        w0 = max(0, min((t0 + t1) // 2 - window_rows // 2, M - window_rows))
        w1 = min(M, w0 + window_rows)
        h_seg, pad = segment_data(backend, g_scene, w0, w1)
        L = h_seg.shape[0]
        f_seg = np.fft.fftfreq(L)
        u_seg = np.where(np.abs(f_seg) <= edge, f_seg / edge, 0.0)
        phi_device = backend.asarray(polynomial_phase(u_seg, coefficients, range_scale))
        if phi_device.ndim == 1:
            phi_device = phi_device[:, None]
        g_win = backend.fft_kernel_minus(h_seg * xp.exp(1j * phi_device))
        out[t0:t1] = g_win[pad + (t0 - w0) : pad + (t1 - w0)]
    return out


#: Проходов посегментной поправки. Внутри сегмента ошибка не одна, а плавно
#: меняется, и первый проход находит её с перебором: замер с внесённой
#: растущей 100…500 рад — найдено 237/319/371 при 168/237/305. Второй
#: проход идёт по уже исправленной сцене, где остаток в разы меньше и
#: внутри сегмента почти постоянен. Третий проход на том же замере
#: ничего не добавлял.
SEGMENT_PASSES = 3

#: ПЕРВЫЙ ПРОХОД — ТОЛЬКО КВАДРАТИЧНАЯ. Замер на крае кадра владельца,
#: скан по окнам с центрами 1024 / 1536 / 2048 строк:
#:
#:   только квадратичная            120 / 130 / 160 рад
#:   квадратичная + кубическая      125 / 150 / 174 рад, кубическая -24…-53
#:
#: Цели в этих строках резки при 125…135: совместная подгонка завышает
#: квадратичную и добавляет кубическую, которая одна ухудшает цели с 8,0
#: до 12,2 элемента по -20 дБ (вместе — до 19,2). При квадратичной в
#: сотни радиан кубическая в кольцевом окне ловит шум. Она ищется на
#: втором проходе, когда остаток мал, и в узких пределах.
FIRST_PASS_DEGREES = (2,)

#: ВТОРОЙ ПРОХОД — окна вдвое короче и шаг в четверть: остаток после
#: первого прохода — десятки радиан, размаз +-100…300 строк, и окно в 1024
#: строки его накрывает, а профиль видит вдвое мельче. Замер: окна 1024
#: и 512 строк на крае кадра дают тот же профиль, что 2048 (dS -0,11 /
#: -0,09 против -0,12), пока ошибка не больше ~300 рад.
SECOND_PASS_ROWS = 1024
SECOND_PASS_HOP = 256
SECOND_PASS_PRESCAN_HALF_RAD = 150.0

#: Кубическая во втором проходе ОТКЛЮЧЕНА (ноль — не ищется). Замер на
#: крае кадра: второй проход находил кубическую -6…-140 рад с dS всего
#: -0,003…-0,026 на окно, и цели в строках 1368…1798 от неё расплывались:
#: -20 дБ 8,0 -> 8,4 при квадратичной с кубической против 8,0 -> 4,0 при
#: одной квадратичной на сцену. Энтропия леса кубическую хочет, точечные
#: цели — нет; верим целям. Остаток выше квадратичной снимает этап C по
#: бинам блока.
SECOND_PASS_CUBIC_HALF_RAD = 0.0
SECOND_PASS_DEGREES = (2,)


def segment_band_width(backend: Backend, g_scene, start: int, stop: int) -> tuple[float, float]:
    """Ширина полосы -10 дБ и её центр в сегменте, в долях частоты повторения.

    Принимает бэкенд, изображение сцены и строки; возвращает (ширина, центр).
    Зачем: у края кадра апертура неполная, и полоса сужается — там
    поправка не поможет, и об этом надо печатать, а не молчать.
    """
    xp = backend.xp
    rows = stop - start
    window = xp.asarray(np.hanning(rows), dtype=backend.real_dtype)[:, None]
    spectrum = backend.to_numpy(
        (xp.abs(xp.fft.fft(g_scene[start:stop] * window, axis=0)) ** 2).sum(axis=1))
    f = np.fft.fftfreq(rows)
    band = spectrum > spectrum.max() / 10.0
    width = float(band.sum()) / rows
    centre = float(np.sum(f[band] * spectrum[band]) / max(np.sum(spectrum[band]), 1e-300))
    return width, centre


def correct_segments(backend: Backend, h, range_scale: np.ndarray | None = None,
                     passes: int = SEGMENT_PASSES,
                     rows: int = SEGMENT_ROWS, hop: int = SEGMENT_HOP):
    """Посегментная поправка в несколько проходов.

    Принимает бэкенд, данные сцены и масштаб дальности; возвращает
    (исправленное изображение сцены на устройстве, список проходов; проход
    — словарь: segments, bands (полоса по сегментам), rows, hop).
    Первый проход — только квадратичная в длинных окнах (FIRST_PASS_*),
    второй — квадратичная и малая кубическая в коротких (SECOND_PASS_*).

    Каждый проход: search_segments на текущих данных, apply_segments,
    и следующий проход идёт по исправленной сцене (обратное ПФ по азимуту
    возвращает её в область дальность-доплер).
    """
    h_current = backend.asarray(h)
    passes_out = []
    scene = None
    for index in range(max(passes, 1)):
        if index == 0:
            # грубо: длинные окна, только квадратичная, широкий скан
            segments = search_segments(backend, h_current, range_scale, rows, hop,
                                       prescan_half=SEGMENT_PRESCAN_HALF_RAD,
                                       cubic_half=0.0, degrees=FIRST_PASS_DEGREES)
            rows_used, hop_used = rows, hop
        else:
            # остаток: короткие окна, квадратичная и малая кубическая
            segments = search_segments(backend, h_current, range_scale,
                                       SECOND_PASS_ROWS, SECOND_PASS_HOP,
                                       prescan_half=SECOND_PASS_PRESCAN_HALF_RAD,
                                       cubic_half=SECOND_PASS_CUBIC_HALF_RAD,
                                       degrees=SECOND_PASS_DEGREES)
            rows_used, hop_used = SECOND_PASS_ROWS, SECOND_PASS_HOP
        scene = apply_segments(backend, h_current, segments, range_scale)
        g_now = backend.fft_kernel_minus(h_current)
        bands = [segment_band_width(backend, g_now, s.row_start, s.row_stop)
                 for s in segments]
        passes_out.append({"segments": segments, "bands": bands,
                           "rows": rows_used, "hop": hop_used})
        h_current = backend.xp.fft.ifft(scene, axis=0) / backend.alpha
    return scene, passes_out

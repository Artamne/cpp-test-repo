"""Договор о входных данных и его проверка.

Файл не считает алгоритм. Он отвечает на один вопрос: годятся ли
подготовленные данные, и если нет — что именно не так. Нужен потому, что
почти всякий отказ MN-MEA на реальной записи объясняется не алгоритмом, а
входом: не та раскладка по доплеру, не те единицы, не сошедшаяся геометрия.

Порядок функций — порядок применения:

    aperture_plan       сколько кадров выйдет из записи и какое M в каждом
    check_data          h(k,n): форма, тип, конечность, раскладка по k
    check_geometry      Geometry: единицы, знаки, согласованность
    derived_numbers     что из них получится: PRF, размер сцены, число блоков
    check_inputs        всё сразу, одним вызовом

Каждая проверка возвращает список замечаний строками. Пустой список значит
«годится». Ничего не печатается и не выбрасывается: решает вызывающий.
"""

from __future__ import annotations

import math

import numpy as np

from stage_a_blocks import (
    Geometry,
    block_counts,
    block_half_sizes,
    block_sample_sizes,
    linearisation_coefficients,
)

#: Меньше этого числа доплеровских бинов этап C не имеет смысла: вектор фазы
#: становится короче, чем разумная модель ошибки. Замерено на краях (§7.5):
#: при M = 8 итерации расходятся, при M = 16 сходятся на порядок хуже.
MIN_AZIMUTH_BINS = 16

#: Во сколько раз энергия в первой половине доплеровской оси должна
#: превышать энергию у её краёв, чтобы считать раскладку numpy.fft
#: подтверждённой. Не строгий критерий, а предупреждение: у сцены,
#: занимающей всю полосу, обе половины равны, и проверка промолчит.
BAND_CENTRE_RATIO = 2.0


def aperture_plan(
    wavelength_m: float,
    range_near_m: float,
    range_far_m: float,
    speed_ms: float,
    prf_hz: float,
    azimuth_resolution_m: float,
    record_seconds: float,
    overlap: float = 0.0,
) -> dict[str, float]:
    """План нарезки записи на апертуры — ДО всякого автофокуса.

    Принимает длину волны, ближнюю и дальнюю границы полосы дальностей,
    путевую скорость, частоту повторения, нужное разрешение по азимуту,
    длительность записи и перекрытие апертур (0 — встык, 0,5 — половинное);
    возвращает словарь чисел.

    Зачем это здесь. Запись длиной в секунды — НЕ одна апертура. Апертура
    задаётся нужным разрешением:

        T_a = lambda * R / (2 * v * r_a)

    и растёт с дальностью, потому что скорость изменения доплера
    K_a = 2 v^2 / (lambda R) с дальностью падает. Полоса доплера при этом от
    дальности НЕ зависит вовсе: B_a = v / r_a.

    Отсюда правило: T_a берётся по ДАЛЬНЕЙ кромке. Тогда число импульсов
    M = PRF * T_a одно для всех стробов, и массив h выходит прямоугольным —
    а другого алгоритм не принимает. Ценой будет переразрешение ближней
    кромки ровно в range_far / range_near раз: там апертура длиннее, чем
    нужно. Это не ошибка; хотите ровное разрешение по полосе — отфильтруйте
    ближние стробы до той же полосы доплера, но помните, что опустевшие бины
    дадут E'' = 0 и фаза в них не определится.

    Возвращаются и границы разумной сетки блоков: замером (см. README,
    «Если ИНС нет») плато лежит на блоках в 16-32 отсчёта по азимуту.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"перекрытие должно быть в [0, 1), а не {overlap!r}")

    aperture_s = wavelength_m * range_far_m / (2.0 * speed_ms * azimuth_resolution_m)
    pulses = int(math.ceil(prf_hz * aperture_s))
    step_s = aperture_s * (1.0 - overlap)
    frames = int(math.floor((record_seconds - aperture_s) / step_s)) + 1 if record_seconds >= aperture_s else 0

    return {
        "aperture_seconds": aperture_s,
        "aperture_length_m": speed_ms * aperture_s,
        "pulses_per_aperture": pulses,
        "doppler_bandwidth_hz": speed_ms / azimuth_resolution_m,
        "frames_in_record": frames,
        "frame_step_seconds": step_s,
        "near_edge_resolution_m": wavelength_m * range_near_m / (2.0 * speed_ms * aperture_s),
        "near_edge_oversampling": range_far_m / range_near_m,
        "blocks_azimuth_min": int(math.ceil(pulses / 32)),
        "blocks_azimuth_max": int(math.ceil(pulses / 16)),
    }


def check_data(h) -> list[str]:
    """Проверка массива h(k,n) — единственного обязательного входа.

    Принимает что угодно, что притворяется массивом; возвращает список
    замечаний. Пустой список — годится.

    Что требуется:

      * два измерения, (M, N): k — доплеровский бин, он же номер импульса в
        апертуре; n — строб дальности. Именно в таком порядке: сумма (5-3)
        идёт по ПЕРВОМУ индексу;
      * комплексный тип. Вещественный массив означает, что фаза потеряна, а
        искать нечего: весь алгоритм работает с фазой;
      * ни одного NaN и ни одной бесконечности: (5-5) берёт логарифм, и одна
        такая ячейка испортит сумму по всему блоку;
      * M >= MIN_AZIMUTH_BINS;
      * раскладка по k как у numpy.fft: нулевая частота в индексе 0,
        положительные дальше, отрицательные с конца. Это ПРЕДУПРЕЖДЕНИЕ, а не
        отказ: строго проверить нельзя, но если энергия собрана в середине
        оси, значит данные пришли центрированными и нужен ifftshift.
    """
    notes: list[str] = []
    a = np.asarray(h)

    if a.ndim != 2:
        notes.append(f"h должен быть двумерным (M, N), а у него {a.ndim} измерений")
        return notes
    M, N = a.shape

    if not np.iscomplexobj(a):
        notes.append("h должен быть КОМПЛЕКСНЫМ: у вещественного массива фазы нет, "
                     "а искать в нём нечего")
    if not np.all(np.isfinite(a)):
        bad = int(np.count_nonzero(~np.isfinite(a)))
        notes.append(f"в h есть {bad} неконечных значений (NaN или inf): (5-5) берёт "
                     "логарифм, и одна такая ячейка испортит сумму по всему блоку")
    if M < MIN_AZIMUTH_BINS:
        notes.append(f"M = {M} доплеровских бинов — слишком мало, нужно хотя бы "
                     f"{MIN_AZIMUTH_BINS} (см. §7.5, края)")
    if N < 1:
        notes.append(f"N = {N} стробов дальности")
    if M < N:
        notes.append(f"M = {M} меньше N = {N}: похоже, что оси перепутаны местами. "
                     "Первый индекс — доплер, второй — дальность")

    if np.iscomplexobj(a) and np.all(np.isfinite(a)) and M >= MIN_AZIMUTH_BINS:
        power = np.abs(a) ** 2
        по_k = power.sum(axis=1)
        quarter = max(1, M // 8)
        у_нуля = float(по_k[:quarter].sum() + по_k[-quarter:].sum())
        у_середины = float(по_k[M // 2 - quarter : M // 2 + quarter].sum())
        if у_середины > BAND_CENTRE_RATIO * у_нуля:
            notes.append(
                "энергия собрана в СЕРЕДИНЕ доплеровской оси, а ожидается у её краёв: "
                "похоже, h пришёл центрированным. Нужна раскладка numpy.fft — "
                "примените numpy.fft.ifftshift(h, axes=0)"
            )
    return notes


def check_geometry(geom: Geometry) -> list[str]:
    """Проверка Geometry — того, что нужно этапам A и B.

    Принимает Geometry; возвращает список замечаний.

    Этапам C и D геометрия НЕ НУЖНА вовсе: имея один только h, можно считать
    сцену одним блоком. Эта проверка нужна лишь тогда, когда нужно разбиение.

    Проверяется то, что можно проверить, не зная съёмки:

      * положительность там, где величина положительна по смыслу;
      * согласованность R_B0 с (x_m, y_m, z_m): наклонная дальность до центра
        сцены обязана равняться длине этого вектора. Расхождение означает, что
        координаты и дальность взяты из разных мест;
      * k_20 != 0. Этап B делит на него в (5-27)/(5-28), и при нулевом
        квадратичном члене хода дальности он не определён. Ноль получается
        при нулевой скорости.
    """
    notes: list[str] = []

    for name, value in (("R_B0", geom.R_B0), ("f_0", geom.f_0), ("r_a", geom.r_a),
                        ("r_b", geom.r_b), ("T_a", geom.T_a), ("c", geom.c)):
        if not (value > 0.0):
            notes.append(f"{name} = {value!r}, а должно быть больше нуля")

    speed = math.sqrt(geom.v_x0**2 + geom.v_y0**2 + geom.v_z0**2)
    if speed <= 0.0:
        notes.append("скорость нулевая: ход дальности не имеет квадратичного члена, "
                     "и (5-27)/(5-28) не определены — этап B работать не будет")

    distance = math.sqrt(geom.x_m**2 + geom.y_m**2 + geom.z_m**2)
    if distance > 0.0 and geom.R_B0 > 0.0:
        mismatch = abs(distance - geom.R_B0) / geom.R_B0
        if mismatch > 0.01:
            notes.append(
                f"R_B0 = {geom.R_B0:.1f} м не сходится с |(x_m, y_m, z_m)| = "
                f"{distance:.1f} м, расхождение {100 * mismatch:.1f} %. "
                "Наклонная дальность до центра сцены и его координаты должны быть "
                "об одной и той же точке"
            )
    return notes


def derived_numbers(geom: Geometry, M: int, N: int) -> dict[str, float]:
    """Что получится из этих данных, ещё до всякого счёта.

    Принимает Geometry и размеры h; возвращает словарь чисел.

    Смотреть надо на blocks_total. Единица означает, что сцена целиком
    укладывается в один блок — разбиение не нужно и ничего не даст. Сотни
    означают, что блок вышел крошечным: скорее всего перепутаны единицы в
    r_a или в ускорении.
    """
    coefficients = linearisation_coefficients(geom)
    x_p, y_p = block_half_sizes(coefficients, geom)
    m_p, n_p = block_sample_sizes(x_p, y_p, geom)
    M_k, N_k, q = block_counts(M, N, m_p, n_p)
    return {
        "prf_hz": M / geom.T_a,
        "scene_azimuth_m": M * geom.r_a,
        "scene_range_m": N * geom.r_b,
        "block_half_azimuth_m": x_p,
        "block_half_range_m": y_p,
        "block_azimuth_samples": m_p,
        "block_range_samples": n_p,
        "blocks_azimuth": M_k,
        "blocks_range": N_k,
        "blocks_total": q,
    }


def check_inputs(h, geom: Geometry | None = None) -> list[str]:
    """Всё сразу: данные, геометрия и здравость того, что из них выйдет.

    Принимает h и, если нужно разбиение на блоки, Geometry; возвращает
    список замечаний. Пустой список — можно запускать.

    Geometry можно не давать: тогда проверяется только h, и работать можно
    этапами C и D, одним блоком на всю сцену.
    """
    notes = check_data(h)
    if geom is None:
        return notes
    notes += check_geometry(geom)
    if notes:
        return notes  # считать производные по негодным числам бессмысленно

    a = np.asarray(h)
    numbers = derived_numbers(geom, a.shape[0], a.shape[1])
    if numbers["blocks_total"] == 1:
        notes.append("разбиение даёт ОДИН блок на всю сцену: при таких параметрах "
                     "движения изменчивость мала и делить нечего — это не ошибка, "
                     "просто этапы A и B ничего не добавят")
    if numbers["block_azimuth_samples"] < MIN_AZIMUTH_BINS:
        notes.append(
            f"блок вышел в {numbers['block_azimuth_samples']:.1f} отсчёта по азимуту — "
            f"меньше {MIN_AZIMUTH_BINS}. Так бывает при перепутанных единицах в r_a "
            "или в ускорении; проверьте, что r_a в МЕТРАХ на отсчёт, а a_* в м/с²"
        )
    return notes

"""Опыты. §7 и §10 задания.

Порядок функций — порядок выполнения: сначала меры, потом опыты, снизу —
полный прогон A->B->C->D.

Меры:
    remove_first_degree_polynomial   §7.2 — снятие постоянной и линейной по k
    phase_residual_rms               §7.2 — главное число, СКО в радианах
    point_target_metrics             §8   — пик и ширина отклика по азимуту

Опыты (каждый возвращает словарь чисел, ничего не печатает и не рисует):
    experiment_stage_c_alone         §7.1, §10.1  этап C в одиночку
    experiment_error_kinds           §7.3, §10.2  четыре вида ошибки
    experiment_scenes                §7.4, §10.3  три вида сцены
    experiment_magnitude_sweep       §7.3         остаточная против размаха
    experiment_stop_threshold        §6.4         выбор mu замером
    experiment_power_floor           решение №1   влияние порога на энтропию
    experiment_normalisation         решение №3   шаг не зависит от нормировки
    experiment_curvature_policy      решение №2   сторож на E'' <= 0
    experiment_step_limit            решение №2   предел шага (5-8) замером
    experiment_edge_cases            §7.5         края
    experiment_backends              §3, §10.5    сверка двух бэкендов
    experiment_iteration_cost        §10.6,§10.7  два БПФ и один W, счётчиком
    experiment_timing                §8           время на итерацию
    experiment_straight_flight       §4           прямолинейный полёт
    experiment_linearisation         этап B       восстановление k_10,k_20,k_30
    experiment_block_criterion       §4           (5-20) выполняется
    experiment_block_transform       §5           длина (5-3): блок или сцена
    experiment_full_run              §10.4        A->B->C->D целиком
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

import stage_a_blocks as A
import stage_b_initial as B
import stage_c_iterate as C
import stage_d_assemble as D
import synthetic as SY
from backend import get_backend

#: Геометрия для полного прогона A->B->C->D: манёвренный БПЛА, случай книги.
#: Все числа — правдоподобные для малого носителя; поперечное ускорение
#: 45 м/с^2 (~4,6 g) — это и есть «高机动», ради которого книга и написана.
#: Разбиение на блоки из этих чисел ВЫЧИСЛЯЕТСЯ, а не задаётся.
DEMO_GEOMETRY = A.Geometry(
    v_x0=75.0, v_y0=8.0, v_z0=-2.0,
    a_x=4.0, a_y=45.0, a_z=-6.0,
    x_m=0.0, y_m=800.0, z_m=2900.0,
    R_B0=math.sqrt(800.0**2 + 2900.0**2),
    lambda_=0.03, f_0=1.0e10,
    r_a=0.4, r_b=1.2, T_a=1.6,
)

#: Порог останова, выбранный ЗАМЕРОМ (см. experiment_stop_threshold), в
#: отличие от книжного mu = 0,1: книжный останавливает на 7-8 итерации, но
#: оставляет остаточную ошибку около десятой радиана.
MU_MEASURED = 1.0e-3


def _floor_for(backend, h) -> float:
    """Порог решения №1 для блока h — (5-4) плюс power_floor, как в iterate_block.
    Принимает бэкенд и данные блока; возвращает порог мощности."""
    M, N = h.shape
    return C.power_floor(C.total_energy(backend, backend.xp.abs(h) ** 2, M), M, N)


def centred_bin_index(M: int) -> np.ndarray:
    """Номер доплеровского бина в ЦЕНТРИРОВАННОМ представлении:
    k~ = -M/2 … M/2-1 вместо 0 … M-1.

    Принимает M; возвращает вектор длины M.

    Зачем. Ядро (5-3) e^{-j2*pi*k*m/M} периодично по k с периодом M, поэтому
    бины k и k-M — ОДИН И ТОТ ЖЕ бин, и «линейная по k» обязана браться по
    тому представителю, который непрерывен вдоль доплеровской оси. Массив h
    лежит в раскладке numpy.fft (сначала положительные частоты, потом
    отрицательные), и в ней непрерывный представитель — это k~, а не 0…M-1.

    Замером: при ЧЁТНОЙ вносимой ошибке (квадратичной) оба представления дают
    одно и то же, а при НЕЧЁТНОЙ (кубической) снятие линии по 0…M-1 оставляет
    1,58 рад вместо 0,046 — то есть показывает ошибку там, где её нет.
    См. опыт experiment_ambiguity_basis.
    """
    return np.fft.fftfreq(M) * M


def remove_first_degree_polynomial(phi: np.ndarray) -> np.ndarray:
    """§7.2 задания: снять полином первой степени по k.

    Принимает вектор фазы длины M; возвращает вектор той же длины без
    постоянной и линейной по k составляющих.

    Энтропия слепа к постоянной фазе и к линейной по k: постоянная — общий
    поворот, линейная по k — сдвиг картинки по азимуту m. Ни то, ни другое
    изображение не портит, и алгоритм их не обязан находить. Не вычесть —
    и правильная работа будет выглядеть ошибкой.

    Номер бина берётся центрированным (см. centred_bin_index).
    """
    phi = np.asarray(phi, dtype=np.float64)
    k = centred_bin_index(phi.size)
    basis = np.vstack([np.ones_like(k), k]).T
    coeffs, *_ = np.linalg.lstsq(basis, phi, rcond=None)
    return phi - basis @ coeffs


def ambiguity_fit(truth: np.ndarray, estimate: np.ndarray) -> tuple[float, float, np.ndarray]:
    """§7.2 задания: подобрать тот самый полином первой степени по k, который
    вычитается из истины и из оценки, и вернуть остаток.

    Принимает истинную и оценённую фазу длины M; возвращает (a, b, остаток),
    где остаток_k = обёрнутая в (-pi, pi] разность
    estimate_k - truth_k - a - b * k~.

    Подгонка КРУГОВАЯ, а не обычная по методу наименьших квадратов, по двум
    причинам, обе из документа:

      * фаза определена с точностью до 2 pi и разворачивать её не требуется
        (§5.7 документа — там же по этой причине выбрана форма (5-9) с
        экспонентами). Значит сравнивать надо e^{j phi}, а не phi: бин, в
        котором оценка отличается от истины ровно на 2 pi, изображению
        тождествен, и обычное СКО объявило бы его ошибкой в 6,28 рад;
      * номер бина берётся центрированным (см. centred_bin_index).

    Максимум |sum_k e^{j(d_k - b k~)}| по b — это круговой аналог наименьших
    квадратов, и он же модуль ДПФ от e^{j d} по переменной b, поэтому
    ищется БПФ с передискретизацией и уточняется мелким перебором.
    """
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    M = truth.size
    k = centred_bin_index(M)
    z = np.exp(1j * (estimate - truth))

    oversample = 64
    grid = np.fft.fftfreq(M * oversample) * (2 * np.pi)
    spectrum = np.abs(np.fft.fft(np.exp(1j * (estimate - truth))[np.argsort(k)], n=M * oversample))
    b_coarse = grid[int(np.argmax(spectrum))]

    step = 2 * np.pi / (M * oversample)
    fine = b_coarse + np.linspace(-step, step, 129)
    resultant = np.abs(np.exp(-1j * np.outer(fine, k)) @ z)
    b = float(fine[int(np.argmax(resultant))])

    a = float(np.angle(np.sum(z * np.exp(-1j * b * k))))
    residual = np.angle(z * np.exp(-1j * (a + b * k)))
    return a, b, residual


def phase_residual_rms(truth: np.ndarray, estimate: np.ndarray) -> float:
    """§7.2 задания: ГЛАВНОЕ ЧИСЛО — среднеквадратичное расхождение внесённой
    и найденной фазы в радианах, после снятия одного и того же полинома
    первой степени по k из истины и из оценки.

    Принимает истинную и оценённую фазу одной длины; возвращает СКО в радианах.
    """
    return float(np.sqrt(np.mean(ambiguity_fit(truth, estimate)[2] ** 2)))


def truth_on_block_grid(phi_true: np.ndarray, m_block: int) -> np.ndarray:
    """Истинная фаза, снятая на доплеровскую сетку блока.

    Принимает истину длины M (сетка апертуры) и число бинов блока m_block;
    возвращает вектор длины m_block.

    Сравнивать найденную блоком фазу с ОТРЕЗКОМ истины нельзя, и это не
    придирка, а источник неверного вывода: блок из m_block бинов накрывает
    ту же полосу доплера, что и вся апертура, только реже — его бину j
    отвечает частота j/m_block от полосы, а не j-й бин апертуры. Поэтому
    истина не режется, а пересчитывается на сетку блока по нормированной
    частоте (край апертуры = +-1). Фаза для этого достаточно плавная: (5-27)
    квадратична по f_a, а внесённое в стенде дрожание ограничено гармониками
    JITTER_HARMONICS.
    """
    M = np.asarray(phi_true).size
    u_full = np.fft.fftfreq(M) * 2.0
    u_block = np.fft.fftfreq(m_block) * 2.0
    order = np.argsort(u_full)
    return np.interp(u_block, u_full[order], np.asarray(phi_true)[order])


def image_sharpness(backend, image) -> dict[str, float]:
    """Резкость ГОТОВОГО изображения тремя числами, без обращения к истине.

    Принимает бэкенд и комплексное изображение; возвращает
    {'S': нормированная энтропия (5-6), 'contrast': ..., 'peak': ...}.

    Истина для этих чисел не нужна, и в этом всё дело: сравнивать два способа
    обработки по остатку фазы можно лишь когда у них ОДНА сетка, а у блочного
    и сценного преобразований она разная. Готовая картинка — общий знаменатель.

    Все три числа не меняются при циклическом сдвиге изображения, а сдвиг —
    это ровно та линейная по k составляющая, к которой энтропия слепа (§7.2).
    Мера, чувствительная к сдвигу, объявила бы правильную работу провалом.
    """
    g = backend.asarray(np.asarray(image))
    M, N = g.shape
    S_g = float(backend.sum_real(backend.xp.abs(g) ** 2))  # (5-4) впрямую
    P, _, _ = C.image_power(backend, g, C.power_floor(S_g, M, N))
    _, S = C.entropy(backend, P, S_g)
    intensity = np.abs(backend.to_numpy(g)) ** 2
    return {
        "S": S,
        "contrast": float(np.sqrt((intensity**2).mean()) / intensity.mean()),
        "peak": float(np.sqrt(intensity.max())),
    }


def point_target_metrics(image: np.ndarray, row: int, col: int) -> dict[str, float]:
    """§8 задания: выигрыш по пику точечной цели и ширина отклика по азимуту.

    Принимает комплексное изображение, строку и столбец цели; возвращает
    {'peak_db': пик в дБ, 'width_samples': ширина по уровню -3 дБ в отсчётах}.

    Ширина меряется по срезу |g(:, col)|^2 вокруг пика линейной интерполяцией
    точек пересечения уровня -3 дБ. Если срез не опускается до -3 дБ (отклик
    шире окна), возвращается nan, и отчёт скажет об этом прямо.
    """
    cut = np.abs(image[:, col]) ** 2
    peak_index = int(np.argmax(cut))
    peak = float(cut[peak_index])
    if peak <= 0.0:
        return {"peak_db": -np.inf, "width_samples": float("nan")}

    half = peak / 2.0
    M = cut.size

    def _edge(direction: int) -> float:
        previous = peak
        for step in range(1, M):
            index = (peak_index + direction * step) % M
            value = float(cut[index])
            if value <= half:
                span = previous - value
                fraction = (previous - half) / span if span > 0 else 0.0
                return step - 1 + fraction
            previous = value
        return float("nan")

    width = _edge(+1) + _edge(-1)
    return {
        "peak_db": 10.0 * math.log10(peak),
        "width_samples": float(width),
        "row_found": peak_index,
        "row_true": row,
    }


def _stage_c_case(
    scene_kind: str,
    error_kind: str,
    edge_rad: float,
    M: int = 256,
    N: int = 64,
    seed: int = 20250915,
    mu: float = MU_MEASURED,
    backend_name: str = "auto",
    max_iterations: int = C.MAX_ITERATIONS,
    floor_relative: float = C.POWER_FLOOR_RELATIVE,
    curvature_policy: str = "freeze",
    alpha: float = 1.0,
) -> dict:
    """Один опыт этапа C в одиночку (§7.1): сцена, внесённая ошибка, итерации,
    сравнение с истиной. Возвращает словарь чисел; служит телом почти всех
    опытов ниже."""
    backend = get_backend(backend_name, alpha=alpha)
    rng = np.random.default_rng(seed)
    sc = SY.scene(scene_kind, M, N, rng)
    phi_err = SY.phase_error(error_kind, M, edge_rad, rng)
    h = SY.range_doppler_from_scene(backend, sc.image, phi_err)

    # Этап C НЕ ИСПОЛЬЗУЕТ параметров движения: начальная фаза здесь нулевая,
    # геометрия в опыт не входит вовсе (§7.1 задания).
    started = time.perf_counter()
    result = C.iterate_block(
        backend, h, np.zeros(M),
        mu=mu, max_iterations=max_iterations,
        floor_relative=floor_relative, curvature_policy=curvature_policy,
    )
    elapsed = time.perf_counter() - started

    before = backend.to_numpy(D.block_image(backend, h, np.zeros(M)))
    after = backend.to_numpy(D.block_image(backend, h, result.phi))

    out = {
        "scene": scene_kind, "error": error_kind, "edge_rad": edge_rad,
        "M": M, "N": N, "mu": mu, "backend": backend.name,
        "iterations": result.n_iterations, "converged": result.converged,
        "stop_reason": result.stop_reason,
        "entropy_before": result.normalised_entropy_history[0],
        "entropy_after": result.normalised_entropy_final,
        "entropy_raw_before": result.entropy_history[0],
        "entropy_raw_after": result.entropy_final,
        "residual_rms_rad": phase_residual_rms(phi_err, result.phi),
        "initial_rms_rad": phase_residual_rms(phi_err, np.zeros(M)),
        "frozen_bins_total": int(sum(result.frozen_bins_history)),
        "clipped_bins_total": int(sum(result.clipped_bins_history)),
        "floored_fraction": result.floored_fraction,
        "fft_per_iteration": sorted(set(result.fft_per_iteration)),
        "w_per_iteration": sorted(set(result.w_per_iteration)),
        "seconds_total": elapsed,
        "seconds_per_iteration": elapsed / max(1, result.n_iterations),
        "_phi_true": phi_err, "_phi_est": result.phi, "_result": result,
        "_image_before": before, "_image_after": after, "_scene": sc,
    }
    if sc.points:
        row, col = sc.points[0]
        m_before = point_target_metrics(before, row, col)
        m_after = point_target_metrics(after, row, col)
        out["peak_gain_db"] = m_after["peak_db"] - m_before["peak_db"]
        out["width_before"] = m_before["width_samples"]
        out["width_after"] = m_after["width_samples"]
        out["_cut_col"] = col
        out["_cut_row"] = row
    return out


def experiment_stage_c_alone() -> dict:
    """§7.1 и §10.1: первый и главный опыт. Этап C без геометрии вообще —
    внесена известная квадратичная ошибка, найдена, остаточное СКО названо
    числом. Если этап C неверен, остальное смысла не имеет."""
    return _stage_c_case("points_and_clutter", "quadratic", 3.0)


def experiment_error_kinds() -> list[dict]:
    """§7.3 и §10.2: то же для квадратичной, кубической, дрожания и смеси —
    таблицей."""
    return [
        _stage_c_case("points_and_clutter", kind, 3.0) for kind in SY.ERROR_KINDS
    ]


def experiment_scenes() -> list[dict]:
    """§7.4 и §10.3: три вида сцены, в том числе ОБЯЗАТЕЛЬНАЯ сцена без
    точечных целей. Энтропийный критерий на однородном спекле может не иметь
    выраженного минимума — что именно алгоритм там делает, говорится замером."""
    return [_stage_c_case(kind, "quadratic", 3.0) for kind in SY.SCENES]


def experiment_magnitude_sweep() -> list[dict]:
    """§7.3: размах перебором — от долей радиана до нескольких радиан на краю
    апертуры. Даёт график 7: где алгоритм ломается."""
    amplitudes = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    rows = []
    for kind in ("quadratic", "cubic", "jitter"):
        for edge in amplitudes:
            rows.append(_stage_c_case("points_and_clutter", kind, edge))
    return rows


def experiment_stop_threshold() -> list[dict]:
    """§6.4 задания: книга называет mu = 0,1 (§5.7 документа) и обещает
    сходимость за 7-8 итераций. Опыт показывает, что оба числа книги
    воспроизводятся, и чем за них платят в остаточной ошибке."""
    return [
        _stage_c_case("points_and_clutter", "quadratic", 3.0, mu=mu)
        for mu in (1.0e-1, 3.0e-2, 1.0e-2, 1.0e-3, 1.0e-4)
    ]


def experiment_power_floor() -> list[dict]:
    """Решение реализации №1: как выбор порога под логарифмом влияет на
    энтропию и на результат. Меряется на сцене 'points' — единственной, где
    есть ТОЧНЫЕ нули, то есть где порог вообще срабатывает."""
    rows = []
    for scene_kind in ("points", "points_and_clutter"):
        for floor in (1e-16, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2):
            row = _stage_c_case(
                scene_kind, "quadratic", 3.0, floor_relative=floor
            )
            row["floor_relative"] = floor
            rows.append(row)
    return rows


def experiment_normalisation() -> dict:
    """Решение реализации №3. §8 документа: множитель нормировки ПФ в книге не
    указан; на положение минимума не влияет, пока согласован между (5-3) и
    (5-14), но входит в абсолютное значение E'' и значит в длину шага.

    Опыт проверяет ровно это утверждение и уточняет его: «согласован» — это
    ВЗАИМНО ОБРАТНАЯ пара alpha * beta = 1. При ней и E', и E'' выходят
    численно теми же, а значит и шаг (5-8) тот же. При несогласованной паре
    (например alpha = 1/M при beta = 1, то есть просто numpy.fft.ifft и
    numpy.fft.fft) E'' меняется, и шаг вместе с ним.
    """
    M, N = 256, 64
    rng = np.random.default_rng(20250915)
    sc = SY.scene("points_and_clutter", M, N, rng)
    phi_err = SY.phase_error("quadratic", M, 3.0, rng)

    def derivatives(alpha: float) -> tuple[np.ndarray, np.ndarray]:
        backend = get_backend("numpy", alpha=alpha)
        h = SY.range_doppler_from_scene(backend, sc.image, phi_err)
        phi = backend.asarray(np.zeros(M))
        h_phi, g = C.image_from_phase(backend, h, phi)
        h_abs2 = backend.xp.abs(h) ** 2
        floor = C.power_floor(C.total_energy(backend, h_abs2, M), M, N)
        P, ln_P, _ = C.image_power(backend, g, floor)
        G = C.auxiliary_array(backend, ln_P, g)
        W = C.w_product(backend, G, h_phi)
        E_1 = C.first_derivative(backend, W)
        E_2 = C.second_derivative(backend, W, backend.xp.abs(h) ** 2, ln_P)
        return backend.to_numpy(E_1), backend.to_numpy(E_2)

    E1_ref, E2_ref = derivatives(1.0)
    step_ref = -E1_ref / E2_ref
    consistent = []
    for alpha in (1.0, 1.0 / M, 1.0 / math.sqrt(M), float(M), 1234.5):
        E1, E2 = derivatives(alpha)
        step = -E1 / E2
        consistent.append({
            "alpha": alpha,
            "E2_max_abs_rel_diff": float(
                np.max(np.abs(E2 - E2_ref)) / np.max(np.abs(E2_ref))
            ),
            "step_max_abs_rel_diff": float(
                np.max(np.abs(step - step_ref)) / np.max(np.abs(step_ref))
            ),
        })

    # Несогласованная пара: g = ОПФ numpy (alpha = 1/M), G = ПФ numpy (beta = 1),
    # то есть alpha * beta = 1/M, а не 1.
    backend = get_backend("numpy", alpha=1.0)
    h = SY.range_doppler_from_scene(backend, sc.image, phi_err)
    xp = backend.xp
    h_phi = h * xp.exp(1j * xp.zeros(M))[:, None]
    g = xp.fft.ifft(h_phi, axis=0)
    P = xp.maximum(xp.abs(g) ** 2, C.POWER_FLOOR_RELATIVE * float(backend.sum_real(xp.abs(g) ** 2)) / (M * N))
    ln_P = xp.log(P)
    G = xp.fft.fft((1.0 + ln_P) * g, axis=0)
    W = xp.conj(G) * h_phi
    E1_bad = 2.0 * backend.sum_real(W.imag, axis=1)
    E2_bad = -2.0 * ((xp.abs(h) ** 2) @ backend.sum_real(2.0 + ln_P, axis=0)) + 2.0 * backend.sum_real(W.real, axis=1)
    step_bad = -np.asarray(E1_bad) / np.asarray(E2_bad)

    return {
        "consistent": consistent,
        "mismatched_step_max_abs_rel_diff": float(
            np.max(np.abs(step_bad - step_ref)) / np.max(np.abs(step_ref))
        ),
        "mismatched_E2_max_abs_rel_diff": float(
            np.max(np.abs(np.asarray(E2_bad) - E2_ref)) / np.max(np.abs(E2_ref))
        ),
    }


def experiment_curvature_policy() -> dict:
    """Решение реализации №2: сторож на E'' <= 0. Обе политики показываются на
    сцене без точечных целей — там (5-19) заведомо даёт неположительную
    кривизну в части бинов."""
    frozen = _stage_c_case("clutter_only", "quadratic", 3.0, curvature_policy="freeze")
    refused = None
    try:
        _stage_c_case("clutter_only", "quadratic", 3.0, curvature_policy="raise")
    except C.CurvatureRefusal as exc:
        refused = str(exc)
    return {"freeze": frozen, "refusal_message": refused}


def experiment_step_limit() -> list[dict]:
    """Решение реализации №2, часть вторая: чем ограничивать шаг (5-8).

    Предел не выбирается по вкусу, а замеряется: полный набор блоков полного
    прогона считается при нескольких значениях, и смотрят, сколько блоков
    доходит до (5-9) и какой остаётся остаток.

    Отдельной колонкой — сколько раз критерий останова встал РОВНО на 2. Это
    подпись предельного цикла: (5-9) для шага d равен 2|sin(d/2)|, значит при
    d = pi он равен 2, максимуму меры, и упёршийся в такой предел бин не даёт
    (5-9) выполниться никогда, сколько ни итерируй. Поэтому предел обязан быть
    строго меньше pi — это не вкус, а свойство самой (5-9).
    """
    backend = get_backend("auto")
    geom = DEMO_GEOMETRY
    M, N = 256, 96
    rng = np.random.default_rng(20250915)
    sc = SY.scene("points_and_clutter", M, N, rng, n_points=12)
    phi_err = SY.phase_error("mixture", M, 3.0, rng, T_a=geom.T_a)
    h = SY.range_doppler_from_scene(backend, sc.image, phi_err)
    coef = A.linearisation_coefficients(geom)
    x_p, y_p = A.block_half_sizes(coef, geom)
    m_p, n_p = A.block_sample_sizes(x_p, y_p, geom)
    M_k, N_k, q = A.block_counts(M, N, m_p, n_p)
    blocks = A.block_grid(M, N, M_k, N_k, geom)
    g_scene = A.scene_image(backend, h)
    prepared = [
        (A.block_data(backend, g_scene, b), b.shape[0],
         truth_on_block_grid(phi_err, b.shape[0]))
        for b in blocks
    ]

    rows = []
    for step_max in (math.pi, 1.5, 1.0, 0.5, 0.3, 0.1, 0.05, 0.02):
        converged, iterations, residuals, pinned = 0, [], [], 0
        for h_block, m_b, truth in prepared:
            r = C.iterate_block(backend, h_block, np.zeros(m_b),
                                mu=MU_MEASURED, step_max=step_max)
            converged += int(r.converged)
            iterations.append(r.n_iterations)
            residuals.append(phase_residual_rms(truth, r.phi))
            pinned += sum(1 for c in r.criterion_history if c > 1.999)
        rows.append({
            "step_max_rad": step_max,
            "is_pi": step_max == math.pi,
            "converged_blocks": converged,
            "blocks": len(prepared),
            "mean_iterations": float(np.mean(iterations)),
            "mean_residual_rms_rad": float(np.mean(residuals)),
            "criterion_pinned_at_two": pinned,
        })
    return rows


def experiment_edge_cases() -> list[dict]:
    """§7.5 задания: края. M не степень двойки; очень короткий блок; один блок
    на весь кадр (N_k = 1, разбиение чисто азимутальное, §3.7 документа);
    блок, попавший на границу и обрезанный."""
    rows = []
    for label, M, N in (
        ("M не степень двойки (M=250)", 250, 64),
        ("M простое (M=251)", 251, 64),
        ("короткий блок (M=16)", 16, 64),
        ("один строб дальности (N=1)", 128, 1),
        ("узкий по азимуту (M=8)", 8, 32),
    ):
        row = _stage_c_case("points_and_clutter", "quadratic", 3.0, M=M, N=N)
        row["case"] = label
        rows.append(row)

    # Блок, попавший на границу и обрезанный (§7.5). Номинальный размер блока
    # равен ceil(M/M_k), последнему достаётся остаток; насколько он меньше
    # номинального, здесь не утверждается, а СЧИТАЕТСЯ. Второй случай —
    # предельный: блок в один доплеровский бин, то есть короче любого разумного.
    backend = get_backend("auto")
    for M_total, M_k in ((201, 8), (61, 16)):
        blocks = A.block_grid(M_total, 48, M_k, 1, DEMO_GEOMETRY)
        nominal = max(b.shape[0] for b in blocks)
        smallest = min(blocks, key=lambda b: b.shape[0])
        rng = np.random.default_rng(20250915)
        sc = SY.scene("points_and_clutter", M_total, 48, rng)
        phi_err = SY.phase_error("quadratic", M_total, 3.0, rng)
        h = SY.range_doppler_from_scene(backend, sc.image, phi_err)
        h_block = A.block_data(backend, A.scene_image(backend, h), smallest)
        m_b = smallest.shape[0]  # §5: длина (5-3) — размер блока, а не сцены
        result = C.iterate_block(backend, h_block, np.zeros(m_b), mu=MU_MEASURED)
        rows.append({
            "case": f"обрезанный блок q_k={smallest.q_k}: {smallest.shape[0]} бинов "
                    f"против {nominal} у номинального (M={M_total}, M_k={M_k})",
            "iterations": result.n_iterations,
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "residual_rms_rad": phase_residual_rms(
                truth_on_block_grid(phi_err, m_b), result.phi
            ),
            "coverage_exact": sum(b.shape[0] * b.shape[1] for b in blocks)
            == M_total * 48,
        })

    # §3.7: один блок на весь кадр — разбиение чисто азимутальное, N_k = 1.
    geom = dataclasses.replace(DEMO_GEOMETRY, a_y=0.0)
    coef = A.linearisation_coefficients(geom)
    x_p, y_p = A.block_half_sizes(coef, geom)
    m_p, n_p = A.block_sample_sizes(x_p, y_p, geom)
    M_k, N_k, q = A.block_counts(256, 96, m_p, n_p)
    rows.append({
        "case": f"§3.7 без поперечного ускорения: M_k={M_k}, N_k={N_k}, q={q}",
        "M_k": M_k, "N_k": N_k, "q": q,
        "purely_azimuth": N_k == 1,
        "block_numbers": [A.block_number(m, M_k, 1) for m in range(1, M_k + 1)],
    })
    return rows


def experiment_backends() -> dict:
    """§3 и §10.5 задания: один и тот же вход, оба пути, названное число
    расхождения. Сверка обязательна, и она здесь с самого начала.

    Если видеокарты на машине нет, сверяются два доступных пути процессора —
    двойная и одинарная точность. Это не подмена: одинарная точность прямо
    проверяет требование §3 о накоплении сумм по M N ячейкам в двойной
    точности независимо от того, в чём лежит массив. Какие пути сверялись,
    отчёт называет явно.
    """
    from backend import available_backends

    names = available_backends()
    reference = names[0] if names[0].startswith("cupy") else "numpy"
    others = [n for n in names if n != reference]

    M, N = 256, 64
    rng = np.random.default_rng(20250915)
    sc = SY.scene("points_and_clutter", M, N, rng)
    phi_err = SY.phase_error("quadratic", M, 3.0, rng)

    runs: dict[str, dict] = {}
    for name in [reference] + others:
        backend = get_backend(name)
        h = SY.range_doppler_from_scene(backend, sc.image, phi_err)
        started = time.perf_counter()
        result = C.iterate_block(backend, h, np.zeros(M), mu=MU_MEASURED)
        elapsed = time.perf_counter() - started
        runs[name] = {
            "phi": result.phi,
            "iterations": result.n_iterations,
            "converged": result.converged,
            "entropy_after": result.normalised_entropy_final,
            "residual_rms_rad": phase_residual_rms(phi_err, result.phi),
            "seconds_per_iteration": elapsed / max(1, result.n_iterations),
            "on_gpu": backend.on_gpu,
        }

    comparisons = []
    for name in others:
        d_phi = runs[name]["phi"] - runs[reference]["phi"]
        comparisons.append({
            "reference": reference,
            "other": name,
            "phi_max_abs_diff_rad": float(np.max(np.abs(d_phi))),
            "phi_rms_diff_rad": float(np.sqrt(np.mean(d_phi**2))),
            "entropy_abs_diff": abs(
                runs[name]["entropy_after"] - runs[reference]["entropy_after"]
            ),
            "iterations": (runs[reference]["iterations"], runs[name]["iterations"]),
        })

    return {
        "available": names,
        "gpu_present": any(n.startswith("cupy") for n in names),
        "runs": runs,
        "comparisons": comparisons,
    }


def experiment_iteration_cost() -> dict:
    """§10.6 и §10.7 задания: стоимость итерации — РОВНО два азимутальных БПФ,
    и W считается один раз за итерацию. Показывается счётчиком вызовов, а не
    утверждением."""
    backend = get_backend("numpy")
    M, N = 256, 64
    rng = np.random.default_rng(20250915)
    sc = SY.scene("points_and_clutter", M, N, rng)
    phi_err = SY.phase_error("quadratic", M, 3.0, rng)
    h = SY.range_doppler_from_scene(backend, sc.image, phi_err)

    start = backend.counters.snapshot()
    result = C.iterate_block(backend, h, np.zeros(M), mu=MU_MEASURED)
    end = backend.counters.snapshot()

    return {
        "iterations": result.n_iterations,
        "fft_per_iteration_set": sorted(set(result.fft_per_iteration)),
        "w_per_iteration_set": sorted(set(result.w_per_iteration)),
        "fft_kernel_minus_in_loop": sum(1 for _ in result.fft_per_iteration),
        "fft_total_incl_final_image": (end[0] - start[0]) + (end[1] - start[1]),
        "fft_outside_loop": ((end[0] - start[0]) + (end[1] - start[1]))
        - 2 * result.n_iterations,
        "w_total": end[2] - start[2],
        "exactly_two_ffts": set(result.fft_per_iteration) == {2},
        "exactly_one_w": set(result.w_per_iteration) == {1},
    }


def experiment_timing() -> list[dict]:
    """§8 задания: время на итерацию, видеокарта и процессор. Замер на одном и
    том же входе фиксированным числом итераций, чтобы числа были сравнимы."""
    from backend import available_backends

    M, N = 512, 128
    rng = np.random.default_rng(20250915)
    sc = SY.scene("points_and_clutter", M, N, rng)
    phi_err = SY.phase_error("quadratic", M, 3.0, rng)

    rows = []
    for name in available_backends():
        backend = get_backend(name)
        h = SY.range_doppler_from_scene(backend, sc.image, phi_err)
        C.iterate_block(backend, h, np.zeros(M), mu=0.0, max_iterations=2)  # прогрев
        started = time.perf_counter()
        result = C.iterate_block(backend, h, np.zeros(M), mu=0.0, max_iterations=10)
        elapsed = time.perf_counter() - started
        rows.append({
            "backend": name,
            "on_gpu": backend.on_gpu,
            "M": M, "N": N,
            "iterations": result.n_iterations,
            "ms_per_iteration": 1e3 * elapsed / result.n_iterations,
        })
    return rows


def experiment_straight_flight() -> dict:
    """§4 задания: прямолинейный полёт как отдельная проверяемая ветка.
    Подставили v_x=v, v_y=v_z=0, a=0, x_m=0 в ОБЩИЕ формулы (4-83)/(4-84) —
    обязаны получить три числа §2.2 и три нуля."""
    v, R_B0, y_m = 120.0, 8000.0, 5000.0
    geom = A.Geometry(
        v_x0=v, v_y0=0.0, v_z0=0.0, a_x=0.0, a_y=0.0, a_z=0.0,
        x_m=0.0, y_m=y_m, z_m=math.sqrt(max(R_B0**2 - y_m**2, 0.0)),
        R_B0=R_B0, lambda_=0.03, f_0=1.0e10, r_a=1.0, r_b=1.5, T_a=1.2,
    )
    general = A.linearisation_coefficients(geom)
    expected = A.straight_flight_coefficients(v, R_B0, y_m)
    A_1, A_2, mu_3 = A.motion_constants(geom)
    k_10, k_20, k_30 = B.range_coefficients_at_scene_centre(geom)
    D_x, D_y = B.spatially_variant_quadratic_coefficients(geom, general)
    return {
        "A_1": A_1, "A_2": A_2, "mu_3": mu_3,
        "A_1_is_zero": A_1 == 0.0,
        "A_2_equals_v2": abs(A_2 - v**2) <= 1e-9 * v**2,
        "mu_3_is_zero": mu_3 == 0.0,
        "general": general, "expected": expected,
        "max_abs_diff": max(abs(general[k] - expected[k]) for k in expected),
        "all_match": all(
            abs(general[k] - expected[k]) <= 1e-12 * max(1.0, abs(expected[k]))
            for k in expected
        ),
        "zeros_are_zero": all(general[k] == 0.0 for k in ("k2x", "k1y", "k3y")),
        "k_10": k_10, "k_20": k_20, "k_30": k_30,
        "k_20_expected": v**2 / (2 * R_B0),
        "D_x": D_x, "D_y": D_y,
    }


def experiment_linearisation() -> list[dict]:
    """Проверка восстановления k_10, k_20, k_30 (см. stage_b_initial): их
    производная по координате точки обязана воспроизводить напечатанные
    в (4-83)/(4-84) коэффициенты.

    Для k_1 совпадение точное; для k_2 и k_3 книга отбрасывает члены высшего
    порядка по A_1, поэтому расхождение есть, и оно называется числом.
    """
    geom = DEMO_GEOMETRY
    printed = A.linearisation_coefficients(geom)
    delta = 1.0e-2

    def shifted(d_x: float, d_y: float) -> A.Geometry:
        x_m, y_m = geom.x_m - d_x, geom.y_m - d_y
        return dataclasses.replace(
            geom, x_m=x_m, y_m=y_m,
            R_B0=math.sqrt(x_m**2 + y_m**2 + geom.z_m**2),
        )

    rows = []
    for j, (key_x, key_y) in enumerate((("k1x", "k1y"), ("k2x", "k2y"), ("k3x", "k3y"))):
        for axis, key in (("x", key_x), ("y", key_y)):
            plus = B.range_coefficients_at_scene_centre(
                shifted(delta if axis == "x" else 0.0, delta if axis == "y" else 0.0)
            )[j]
            minus = B.range_coefficients_at_scene_centre(
                shifted(-delta if axis == "x" else 0.0, -delta if axis == "y" else 0.0)
            )[j]
            numeric = (plus - minus) / (2 * delta)
            rows.append({
                "coefficient": key,
                "numeric": numeric,
                "printed": printed[key],
                "relative_difference": abs(numeric - printed[key])
                / max(abs(numeric), abs(printed[key]), 1e-300),
            })
    return rows


def experiment_block_criterion() -> dict:
    """§4 задания: проверить, что найденные полуразмеры блока действительно
    удовлетворяют (5-20) на краю апертуры.

    Проверяется ПООСЁВО, как и написано в §3.3 документа: критерий распадается
    на два независимых неравенства. В углу блока (x_p и y_p одновременно) два
    вклада могут частично погасить друг друга, и совместная проверка ничего
    не говорила бы о каждом из них.
    """
    geom = DEMO_GEOMETRY
    coef = A.linearisation_coefficients(geom)
    x_p, y_p = A.block_half_sizes(coef, geom)
    t_edge = geom.T_a / 2.0

    dR_azimuth = abs(coef["k1x"] * x_p) * t_edge
    dR_range = abs(coef["k2y"] * y_p) * t_edge**2
    dR_corner = abs(A.slant_range_change(coef, t_edge, x_p, y_p))
    dR_cubic = abs(coef["k3x"] * x_p + coef["k3y"] * y_p) * t_edge**3

    m_p, n_p = A.block_sample_sizes(x_p, y_p, geom)
    M_k, N_k, q = A.block_counts(256, 96, m_p, n_p)
    return {
        "x_p_m": x_p, "y_p_m": y_p, "m_p": m_p, "n_p": n_p,
        "M_k": M_k, "N_k": N_k, "q": q,
        "r_a_over_2": geom.r_a / 2.0,
        "dR_azimuth_only": dR_azimuth,
        "dR_range_only": dR_range,
        "dR_corner_both": dR_corner,
        "dR_cubic_term": dR_cubic,
        "azimuth_ok": dR_azimuth <= geom.r_a / 2.0 * (1 + 1e-12),
        "range_ok": dR_range <= geom.r_a / 2.0 * (1 + 1e-12),
        "cubic_is_smaller": dR_cubic < dR_azimuth,
    }


def experiment_full_run(
    M: int = 256, N: int = 96, seed: int = 20250915, eta_offset: bool = True,
    scene_transform: bool = False,
) -> dict:
    """§10.4 задания: полный прогон A->B->C->D на синтетике.

    Этап A по геометрии DEMO_GEOMETRY даёт сетку блоков; этап B — начальную
    фазу в каждом блоке; этап C — итерации; этап D — сборку. Возвращает всё,
    что нужно для карты блоков, энтропии по блокам и картинки до и после.

    eta_offset=False убирает из (5-29) неразъяснённый член -(q - q_k)/2 —
    НЕ как альтернативную реализацию, а как ЗАМЕР его цены: см.
    experiment_eta_term и §8 документа, расхождение пятое. Рабочая форма
    основного текста (eta_offset=True) — то, что считается по умолчанию.

    scene_transform=True берёт (5-3) длиной во всю сцену вместо длины блока —
    тоже ЗАМЕР, а не вторая реализация: см. experiment_block_transform. По
    умолчанию делается так, как написано в §5: «далее M, N — размеры блока».
    """
    geom = DEMO_GEOMETRY
    backend = get_backend("auto")
    rng = np.random.default_rng(seed)
    sc = SY.scene("points_and_clutter", M, N, rng, n_points=12)
    phi_err = SY.phase_error("mixture", M, 3.0, rng, T_a=geom.T_a)
    h = SY.range_doppler_from_scene(backend, sc.image, phi_err)

    # --- этап A ---
    coef = A.linearisation_coefficients(geom)
    x_p, y_p = A.block_half_sizes(coef, geom)
    m_p, n_p = A.block_sample_sizes(x_p, y_p, geom)
    M_k, N_k, q = A.block_counts(M, N, m_p, n_p)
    blocks = A.block_grid(M, N, M_k, N_k, geom)

    # --- этап B: D_x, D_y общие для всех блоков, считаются один раз ---
    D_x, D_y = B.spatially_variant_quadratic_coefficients(geom, coef)

    # Блок — участок СЦЕНЫ (§3.3 документа), поэтому изображение строится один
    # раз на всю сцену, а плитки вырезаются из него.
    g_scene = A.scene_image(backend, h)

    per_block = []
    images: dict[int, object] = {}
    images_before: dict[int, object] = {}
    for block in blocks:
        if scene_transform:  # замер другого прочтения, см. experiment_block_transform
            window = backend.xp.zeros_like(g_scene)
            window[block.m_start : block.m_stop, block.n_start : block.n_stop] = g_scene[
                block.m_start : block.m_stop, block.n_start : block.n_stop
            ]
            h_block = (backend.xp.fft.ifft(window, axis=0) / backend.alpha)[
                :, block.n_start : block.n_stop
            ]
            M_block = M
        else:
            h_block = A.block_data(backend, g_scene, block)
            M_block = block.m_stop - block.m_start  # §5: «M, N — размеры блока»

        # eps считается на сетке ЭТОГО блока: длина у блока, полоса у радара.
        f_a_block = B.azimuth_frequency_axis(M_block, geom.T_a, M)
        eps_reference = B.phase_model(D_x, D_y, x_p, y_p, f_a_block)

        # (5-29),(5-30): eta и phi^(0). eps берётся в опорной точке, той же,
        # что в знаменателе (5-29) — см. stage_b_initial.initial_phase.
        eta = B.block_scale_factor(
            D_x, D_y, block.x_centre, block.y_centre, x_p, y_p, q, block.q_k
        )
        if not eta_offset:  # замер цены расхождения §8, см. experiment_eta_term
            eta += (q - block.q_k) / 2.0
        phi_0 = B.initial_phase(eta, eps_reference)

        result = C.iterate_block(backend, h_block, phi_0, mu=MU_MEASURED)
        tile = D.block_image(backend, h_block, result.phi)
        tile_zero = D.block_image(backend, h_block, np.zeros(M_block))
        leak = 0.0  # у блочного (5-3) энергии некуда деться: свёртка круговая
        if scene_transform:  # изображение вышло во всю сцену, берётся полоса блока
            power = np.abs(backend.to_numpy(tile)) ** 2
            leak = 1.0 - float(
                power[block.m_start : block.m_stop].sum() / power.sum()
            )
            tile = tile[block.m_start : block.m_stop]
            tile_zero = tile_zero[block.m_start : block.m_stop]
        images[block.q_k] = tile
        images_before[block.q_k] = tile_zero

        truth_block = truth_on_block_grid(phi_err, M_block)
        per_block.append({
            "q_k": block.q_k, "m_k": block.m_k, "n_k": block.n_k,
            "shape": block.shape,
            "eta": eta,
            "phi0_rms_rad": float(np.sqrt(np.mean(phi_0**2))),
            "iterations": result.n_iterations,
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "entropy_before": result.normalised_entropy_history[0],
            "entropy_after": result.normalised_entropy_final,
            "residual_rms_rad": phase_residual_rms(truth_block, result.phi),
            "initial_rms_rad": phase_residual_rms(truth_block, phi_0),
            "frozen_bins_total": int(sum(result.frozen_bins_history)),
            "leak_fraction": leak,
            "_history": result.normalised_entropy_history,
            "_criterion": result.criterion_history,
        })

    # --- этап D ---
    assembled = backend.to_numpy(D.assemble(backend, images, blocks, M, N))
    assembled_before = backend.to_numpy(D.assemble(backend, images_before, blocks, M, N))

    return {
        "geometry": geom, "blocks": blocks, "M": M, "N": N, "eta_offset": eta_offset,
        "x_p_m": x_p, "y_p_m": y_p, "m_p": m_p, "n_p": n_p,
        "M_k": M_k, "N_k": N_k, "q": q,
        "D_x": D_x, "D_y": D_y,
        "per_block": per_block,
        "converged_blocks": sum(1 for b in per_block if b["converged"]),
        "mean_residual_rms_rad": float(np.mean([b["residual_rms_rad"] for b in per_block])),
        "worst_residual_rms_rad": float(np.max([b["residual_rms_rad"] for b in per_block])),
        "_image_before": assembled_before,
        "_image_after": assembled,
        "_scene": sc,
        "_phi_true": phi_err,
    }


def experiment_eta_term() -> dict:
    """§8 документа, расхождение пятое: член -(q - q_k)/2 в (5-29) в книге не
    выведен. Реализуется рабочая форма основного текста, то есть член оставлен,
    — но чего он стоит, сказано ЗАМЕРОМ, а не мнением.

    Полный прогон A->B->C->D считается дважды: с членом и без него.
    Сравнивается, насколько начальная фаза (5-30) близка к истине и сколько
    итераций уходит потом.
    """
    rows = {}
    for label, flag in (("с членом (рабочая форма)", True), ("без члена", False)):
        run = experiment_full_run(eta_offset=flag)
        rows[label] = {
            "eta_min": min(b["eta"] for b in run["per_block"]),
            "eta_max": max(b["eta"] for b in run["per_block"]),
            "phi0_rms_max_rad": max(b["phi0_rms_rad"] for b in run["per_block"]),
            "mean_initial_rms_rad": float(np.mean([b["initial_rms_rad"] for b in run["per_block"]])),
            "mean_iterations": float(np.mean([b["iterations"] for b in run["per_block"]])),
            "converged_blocks": run["converged_blocks"],
            "q": run["q"],
            "mean_residual_rms_rad": run["mean_residual_rms_rad"],
        }
    return rows


def experiment_block_transform() -> dict:
    """§5 документа: «далее M, N — размеры блока, h(k,n) — его данные».

    Сколько бинов у преобразования (5-3) в блоке — вопрос не праздный, и
    книга отвечает на него дважды по-разному. §4.5 и блок-схема §7 называют
    phi^(0) «вектором длины M», где M — размер сцены; §5, открывая этап C,
    объявляет M размером блока. В список §8 книги это не попало — расхождение
    седьмое, найдено при разборе.

    Реализовано второе — так, как написано у (5-3): длина преобразования есть
    длина суммы по k, то есть размер блока. Чего это стоит, сказано ЗАМЕРОМ:
    полный прогон считается дважды, блочным преобразованием и сценным.

    Сравнение идёт по ГОТОВОЙ картинке (image_sharpness), потому что остаток
    фазы у двух способов живёт на разных сетках и прямо не сопоставим.

    Два довода в пользу блочного, помимо буквы книги, тоже замеряются здесь:
    энергия за границу блока не выходит вовсе (свёртка круговая), и итерация
    дешевле. У сценного и то, и другое хуже — утечка ненулевая, БПФ длиннее.
    """
    backend = get_backend("auto")
    rows: dict[str, dict] = {}
    ideal_scene = None
    for label, flag in (("блочное ПФ (как в §5)", False), ("сценное ПФ", True)):
        run = experiment_full_run(scene_transform=flag)
        ideal_scene = run["_scene"].image
        rows[label] = {
            "transform_length": "M сцены" if flag else "m_b блока",
            "converged_blocks": run["converged_blocks"],
            "mean_residual_rms_rad": run["mean_residual_rms_rad"],
            "leak_percent": 100.0 * float(
                np.mean([b["leak_fraction"] for b in run["per_block"]])
            ),
            **image_sharpness(backend, run["_image_after"]),
        }
    rows["идеальная сцена"] = {
        "transform_length": "—", "converged_blocks": None,
        "mean_residual_rms_rad": 0.0, "leak_percent": 0.0,
        **image_sharpness(backend, ideal_scene),
    }
    return rows


def experiment_ambiguity_basis() -> dict:
    """§7.2 задания, проверка самой меры. Показывает замером, что:

      * энтропия ТОЧНО слепа к постоянной фазе;
      * к линейной по k она слепа лишь при сдвиге на ЦЕЛОЕ число отсчётов,
        а при дробном сдвиге меняется — то есть «слепота» из §7.2 верна не
        буквально, и снятие линии остаётся консервативной (в пользу
        алгоритма) поправкой, а не тождеством;
      * снимать линию надо по ЦЕНТРИРОВАННОМУ номеру бина: при нечётной
        (кубической) ошибке снятие по 0…M-1 показывает ошибку там, где её нет.
    """
    from backend import get_backend as _gb

    backend = _gb("numpy")
    M, N = 256, 64
    rng = np.random.default_rng(20250915)
    sc = SY.scene("points_and_clutter", M, N, rng)
    h = SY.range_doppler_from_scene(backend, sc.image, np.zeros(M))

    def normalised_entropy(phi: np.ndarray) -> float:
        _, g = C.image_from_phase(backend, h, backend.asarray(phi))
        P, _, _ = C.image_power(backend, g, _floor_for(backend, h))
        return C.entropy(backend, P, float(backend.sum_real(P)))[1]

    k_centred = centred_bin_index(M)
    S_0 = normalised_entropy(np.zeros(M))
    integer_shifts = [
        abs(normalised_entropy(2 * np.pi * s / M * k_centred) - S_0) for s in (1, 2, 5)
    ]
    fractional = [
        (b, abs(normalised_entropy(b * k_centred) - S_0)) for b in (0.01, 0.05, 0.2)
    ]

    # Кубическая ошибка: центрированный номер бина против натурального 0…M-1.
    case = _stage_c_case("points_and_clutter", "cubic", 3.0)
    truth, estimate = case["_phi_true"], case["_phi_est"]
    k_natural = np.arange(M, dtype=np.float64)
    basis = np.vstack([np.ones(M), k_natural]).T
    coeffs, *_ = np.linalg.lstsq(basis, estimate - truth, rcond=None)
    naive = float(np.sqrt(np.mean((estimate - truth - basis @ coeffs) ** 2)))

    return {
        "constant_phase_entropy_change": abs(normalised_entropy(np.full(M, 1.234)) - S_0),
        "integer_shift_entropy_change_max": max(integer_shifts),
        "fractional_shift_entropy_change": fractional,
        "cubic_rms_natural_k": naive,
        "cubic_rms_centred_k": case["residual_rms_rad"],
    }


def experiment_entropy_versus_truth() -> list[dict]:
    """Проверка правильности, не зависящая от меры расхождения фаз: энтропия
    в найденной точке против энтропии в ИСТИННОЙ точке.

    Алгоритм минимизирует (5-5), а не расстояние до истины. Если найденная
    точка даёт энтропию не хуже истинной — алгоритм сделал ровно то, что
    обещал, и остаточное СКО говорит уже о сцене, а не о нём. Если хуже —
    он застрял, и это надо назвать.
    """
    from backend import get_backend as _gb

    M, N = 256, 64
    rows = []
    for scene_kind in SY.SCENES:
        for error_kind in SY.ERROR_KINDS:
            backend = _gb("numpy")
            rng = np.random.default_rng(20250915)
            sc = SY.scene(scene_kind, M, N, rng)
            phi_err = SY.phase_error(error_kind, M, 3.0, rng)
            h = SY.range_doppler_from_scene(backend, sc.image, phi_err)

            def normalised_entropy(phi: np.ndarray) -> float:
                _, g = C.image_from_phase(backend, h, backend.asarray(phi))
                P, _, _ = C.image_power(backend, g, _floor_for(backend, h))
                return C.entropy(backend, P, float(backend.sum_real(P)))[1]

            result = C.iterate_block(backend, h, np.zeros(M), mu=MU_MEASURED)
            S_truth = normalised_entropy(phi_err)
            S_found = normalised_entropy(result.phi)
            S_start = normalised_entropy(np.zeros(M))
            available = S_start - S_truth
            rows.append({
                "scene": scene_kind, "error": error_kind,
                "S_start": S_start, "S_truth": S_truth, "S_found": S_found,
                # Доля доступного падения энтропии, которую алгоритм НЕ взял:
                # 0 и меньше — сделал не хуже истины; 1 — не сделал ничего.
                "missed_fraction": (S_found - S_truth) / available if available > 0 else float("nan"),
                "found_not_worse_than_truth": S_found <= S_truth + 1e-3 * max(abs(available), 1e-12),
                "residual_rms_rad": phase_residual_rms(phi_err, result.phi),
                "converged": result.converged,
            })
    return rows

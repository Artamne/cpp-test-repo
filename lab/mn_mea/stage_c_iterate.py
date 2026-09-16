"""Этап C. Итерации MN-MEA внутри блока. Формулы (5-3),(5-5),(5-14),(5-13),
(5-19),(5-8),(5-9); порядок операций — §5.8 документа.

Порядок функций — порядок выполнения, один к одному с девятью шагами §5.8:

    total_energy          (5-4) — полная энергия, через Парсеваля из h
    power_floor           решение реализации №1 — порог мощности под логарифмом
    image_from_phase      шаги 1-2  (5-3)   БПФ №1
    image_power           шаг 3     P, ln P
    entropy               (5-4),(5-5),(5-6) — для отчёта, БПФ не стоит
    auxiliary_array       шаг 4     (5-14)  БПФ №2
    w_product             шаг 5     W = G* h e^{j phi} — ЕДИНСТВЕННОЕ место
    first_derivative      шаг 6     (5-13)  Im[W]
    second_derivative     шаг 7     (5-19)  Re[W]
    newton_update         шаг 8     (5-8)   + решение реализации №2
    stop_criterion        шаг 9     (5-9)
    iterate_block         цикл целиком

Этап C НЕ ИСПОЛЬЗУЕТ ни одного параметра движения (§2.1 документа): в
(5-3),(5-13),(5-14),(5-19),(5-8),(5-9) входят только h(k,n) и текущий phi_k.

Ключевая экономия алгоритма: и градиент, и гессиан выражаются через одну
величину W = G* h e^{j phi} — мнимая часть даёт (5-13), вещественная входит
в (5-19). Отсюда два азимутальных БПФ на итерацию и ничего больше. Если в
коде окажется второе место, где считается W, — это ошибка реализации, а не
оптимизация.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backend import Backend

#: Порог останова (5-9). Число книжное: §5.7 документа называет mu = 0,1 и
#: поясняет, что при малых разностях фаз это примерно 0,1 радиана.
MU_DEFAULT = 0.1

#: Предел числа итераций. Книга (§2.3) обещает сходимость за 7-8 итераций на
#: манёвренном носителе и прямо предупреждает, что на ровном полёте с малым
#: phi^(0) сходимость МЕДЛЕННЕЕ. Предел взят оттуда с пятикратным запасом:
#: 5 x 8 = 40. Упереться в него — НЕ сходимость, и отчёт обязан так и сказать.
MAX_ITERATIONS = 40

#: Решение реализации №1: порог мощности под логарифмом, в долях СРЕДНЕЙ
#: мощности изображения S_g/(M N). Порог относительный, а не абсолютный, —
#: иначе он ломал бы независимость шага от нормировки ПФ (решение №3).
#: 1e-12 — это -120 дБ от средней мощности, заведомо ниже динамического
#: диапазона любой реальной записи: на измеренных данных порог не срабатывает
#: никогда. Срабатывает он только на синтетике с ТОЧНЫМИ алгебраическими
#: нулями. Доля сработавших пикселей всегда попадает в отчёт.
POWER_FLOOR_RELATIVE = 1e-12

#: Решение реализации №2, часть вторая: предел шага (5-8) по модулю.
#: Число выбрано ЗАМЕРОМ (validate.experiment_step_limit), и прежнее значение
#: pi было неверным. Довод за pi казался убедительным — §5.7 говорит, что фаза
#: определена с точностью до 2 pi, значит шаг больше pi новых сведений не
#: несёт. Но у предела есть второе, важнейшее свойство: критерий останова
#: (5-9) для шага d равен 2|sin(d/2)|, и при d = pi это РОВНО 2, то есть
#: максимум, какой (5-9) вообще принимает. Бин, упирающийся в предел pi,
#: держит критерий на 2 навсегда, и (5-9) не может выполниться никогда — это
#: предельный цикл, а не сходимость. Замер: при pi критерий встаёт на 2,00
#: семьдесят один раз на 12 блоков и сходятся 2 блока; при 1 рад таких
#: случаев нет вовсе и сходятся 9. Предел обязан быть строго меньше pi;
#: 1 рад — лучшее по замеру.
STEP_MAX_RAD = 1.0


class CurvatureRefusal(RuntimeError):
    """Отказ по решению реализации №2 при curvature_policy='raise':
    вторая производная (5-19) неположительна, Ньютон делить на неё не может."""


@dataclass
class BlockIterations:
    """Всё, что итерации выдали по блоку. Каждое поле — число в отчёт (§8)."""

    phi: np.ndarray
    phi_initial: np.ndarray
    n_iterations: int
    converged: bool
    stop_reason: str
    S_g: float
    entropy_history: list[float] = field(default_factory=list)
    normalised_entropy_history: list[float] = field(default_factory=list)
    criterion_history: list[float] = field(default_factory=list)
    frozen_bins_history: list[int] = field(default_factory=list)
    clipped_bins_history: list[int] = field(default_factory=list)
    fft_per_iteration: list[int] = field(default_factory=list)
    w_per_iteration: list[int] = field(default_factory=list)
    floored_fraction: float = 0.0
    entropy_final: float = 0.0
    normalised_entropy_final: float = 0.0


def total_energy(backend: Backend, h_abs2, M: int) -> float:
    """(5-4): S_g = sum_m sum_n |g(m,n)|^2 — полная энергия изображения блока.

    Принимает бэкенд, готовый массив |h(k,n)|^2 и число азимутальных отсчётов M;
    возвращает одно вещественное число.

    Считается ПРЯМО ИЗ h, без построения изображения. Основание — §5.2
    документа: величина S_g от phi не зависит, потому что ПФ и умножение на
    экспоненту единичного модуля энергию сохраняют. По равенству Парсеваля для
    ненормированного ядра (5-3):

        S_g = alpha^2 * M * sum_{k,n} |h(k,n)|^2

    Множитель M — от ненормированного ядра, alpha^2 — от выбранной нормировки
    (решение реализации №3).

    На вход берётся уже посчитанный |h|^2, а не сам h: он всё равно нужен
    (5-19) на каждой итерации, и второй проход по массиву ни к чему.
    """
    return backend.alpha**2 * M * float(backend.sum_real(h_abs2))


def power_floor(S_g: float, M: int, N: int, floor_relative: float = POWER_FLOOR_RELATIVE) -> float:
    """Решение реализации №1: порог для |g|^2 под логарифмом.

    Принимает полную энергию S_g (5-4), размеры блока M, N и долю;
    возвращает порог мощности — одно вещественное число, ниже которого P
    не опускается.

        порог = floor_relative * S_g / (M N)

    S_g/(M N) — средняя мощность пикселя, поэтому порог ОТНОСИТЕЛЬНЫЙ: он
    следует за яркостью самих данных. Абсолютная эпсилон сломала бы
    независимость шага от нормировки ПФ (решение реализации №3).

    Порог постоянен на все итерации блока: S_g от phi не зависит.
    """
    return floor_relative * S_g / (M * N)


def image_from_phase(backend: Backend, h, phi):
    """(5-3), шаги 1-2 §5.8: g(m,n) = sum_k h(k,n) e^{j phi_k} e^{-j 2 pi k m / M}.

    Принимает бэкенд, данные блока h(k,n) и текущую поправку phi_k;
    возвращает (h_phi, g) — домноженные по строкам данные и изображение блока.
    Стоит ОДНО азимутальное БПФ (ядро «-»).
    """
    h_phi = h * backend.xp.exp(1j * phi)[:, None]  # шаг 1
    g = backend.fft_kernel_minus(h_phi)  # шаг 2, БПФ №1
    return h_phi, g


def image_power(backend: Backend, g, floor: float):
    """Шаг 3 §5.8: P = |g|^2 и ln P, с порогом по решению реализации №1.

    Принимает бэкенд, изображение g и порог мощности; возвращает
    (P, ln P, доля пикселей, попавших на порог).

    Без порога |g| = 0 даёт ln P = -inf, а P ln P в (5-5) — nan; то же в
    L = 1 + ln P (5-14) и в W_n = sum_m (2 + ln P) (5-19).
    """
    P = backend.xp.abs(g) ** 2
    floored = P < floor
    fraction = float(backend.sum_real(floored.astype(backend.real_dtype))) / P.size
    P = backend.xp.maximum(P, backend.xp.asarray(floor, dtype=P.dtype))
    return P, backend.xp.log(P), fraction


def entropy(backend: Backend, P, S_g: float) -> tuple[float, float]:
    """(5-5) ненормированная энтропия и (5-6) нормированная.

    Принимает бэкенд, мощность P (5-4 уже посчитана как S_g) и S_g;
    возвращает (E_g, S).

        E_g = - sum_m sum_n |g|^2 ln |g|^2                (5-5)
        S   = E_g / S_g + ln S_g                          (5-6)

    Отдельного БПФ не стоит: P уже посчитано на шаге 3. Величина S от
    нормировки ПФ не зависит и потому сравнима между блоками, E_g — нет.
    Накопление по M N ячейкам идёт в двойной точности (§3 задания).
    """
    E_g = -float(backend.sum_real(P * backend.xp.log(P)))
    return E_g, E_g / S_g + math.log(S_g)


def auxiliary_array(backend: Backend, ln_P, g):
    """(5-14), шаг 4 §5.8: G(k,n) = sum_m [1 + ln |g|^2] g(m,n) e^{+j 2 pi k m / M}.

    Принимает бэкенд, ln P и изображение g; возвращает массив G(k,n).
    Ядро с ПРОТИВОПОЛОЖНЫМ знаком относительно (5-3).
    Стоит ОДНО азимутальное БПФ (ядро «+»). Это второе и последнее БПФ итерации.
    """
    L = 1.0 + ln_P  # вещественный множитель
    return backend.fft_kernel_plus(L.astype(g.dtype) * g)  # БПФ №2


def w_product(backend: Backend, G, h_phi):
    """Шаг 5 §5.8: W(k,n) = G*(k,n) h(k,n) e^{j phi_k} = G* h_phi.

    Принимает бэкенд, G (5-14) и h_phi (шаг 1); возвращает W(k,n).

    ЕДИНСТВЕННОЕ место в проекте, где считается W. Из него берутся ОБЕ
    производные: Im даёт (5-13), Re входит в (5-19). Счётчик
    backend.counters.w_product существует ровно для того, чтобы замером
    показать однократность (§10.7 задания).
    """
    backend.counters.w_product += 1
    return backend.xp.conj(G) * h_phi


def first_derivative(backend: Backend, W):
    """(5-13), шаг 6 §5.8: E'_k = 2 sum_n Im[G*(k,n) h(k,n) e^{j phi_k}].

    Принимает бэкенд и уже посчитанное W; возвращает вектор длины M
    вещественных чисел — по одному на доплеровский бин.

    В книге напечатано -2 Im[...] при e^{+j phi_k} — см. §8 документа: знак
    перед двойкой и знак phi_k в экспоненте меняются только парой, а в (5-3)
    стоит e^{+j phi_k}, поэтому рабочая форма здесь +2.
    """
    # РАСХОЖДЕНИЕ (5-13): в книге напечатано иначе, см. §8 — там -2 Im[...]
    # при e^{+j phi_k}; знак перед двойкой и знак phi_k меняются только парой.
    return 2.0 * backend.sum_real(W.imag, axis=1)


def second_derivative(backend: Backend, W, h_abs2, ln_P):
    """(5-19), шаг 7 §5.8: вторая производная энтропии по phi_k.

        E''_k = -2 sum_m sum_n (2 + ln|g|^2) |h(k,n)|^2
                + 2 sum_n Re[G*(k,n) h(k,n) e^{j phi_k}]

    Принимает бэкенд, то же самое W, |h|^2 и ln P; возвращает вектор длины M.

    Первый член считается через вспомогательный вектор длины N, один раз за
    итерацию (§5.5 документа):

        W_n(n) = sum_m (2 + ln|g|^2)   =>   первый член = -2 sum_n |h|^2 W_n(n)

    Второй член использует ТОТ ЖЕ W, что и (5-13), — меняется только Im на Re.

    Коэффициент (2 + ln|g|^2) в книге дан без вывода — см. §8 документа: он
    получается из (5-15) при замене Im^2[g* u] на среднее (1/2)|g|^2 |h|^2,
    то есть (5-19) — это усреднённый, а не точный гессиан.
    """
    W_n = backend.sum_real(2.0 + ln_P, axis=0)  # длины N
    # РАСХОЖДЕНИЕ (5-19): в книге напечатано иначе, см. §8 — коэффициент
    # (2 + ln|g|^2) дан без вывода; он получается из (5-15) при замене
    # Im^2[g* u] на среднее (1/2)|g|^2|h|^2, то есть гессиан усреднённый.
    first = -2.0 * (h_abs2 @ W_n)
    second = 2.0 * backend.sum_real(W.real, axis=1)
    return first + second


def newton_update(
    phi,
    E_1,
    E_2,
    backend: Backend,
    curvature_policy: str = "freeze",
    step_max: float = STEP_MAX_RAD,
):
    """(5-8), шаг 8 §5.8: phi^(l+1) = phi^(l) - E'_k / E''_k.

    Принимает текущий phi, производные (5-13) и (5-19), бэкенд и политику
    решения реализации №2; возвращает (phi_new, число замороженных бинов,
    число бинов, упёршихся в предел шага).

    Все M компонент обновляются ОДНОВРЕМЕННО, из одного и того же состояния
    phi^(l) (§5.6 документа) — это обеспечивается векторной записью.

    РЕШЕНИЕ РЕАЛИЗАЦИИ №2. Книга сторожа не даёт, а Ньютон делит на E''.
    Здесь два названных ограничения, и оба попадают в отчёт:

      * E''_k <= 0 — квадратичная модель в этом бине не имеет минимума, и
        -E'/E'' указывает ВВЕРХ по энтропии. Бин не обновляется на этой
        итерации ('freeze'); при curvature_policy='raise' — отказ с причиной.
        Молчаливого шага не делается ни при какой политике.
      * |шаг| > step_max — ограничивается. Это ловит и случай E'' -> +0,
        где Ньютон выдал бы огромный шаг. Предел строго меньше pi, иначе
        упёршийся бин держит критерий (5-9) на его максимуме 2 навсегда;
        см. STEP_MAX_RAD и validate.experiment_step_limit.
    """
    xp = backend.xp
    bad = E_2 <= 0.0
    n_frozen = int(backend.sum_real(bad.astype(backend.accum_real)))

    if n_frozen and curvature_policy == "raise":
        which = np.flatnonzero(backend.to_numpy(bad))
        raise CurvatureRefusal(
            f"(5-19) дала E'' <= 0 в {n_frozen} бинах из {E_2.size}: {which[:16].tolist()}"
            f"{' …' if n_frozen > 16 else ''}. Ньютон (5-8) делит на E''; "
            "шаг не определён, продолжать нельзя."
        )
    if n_frozen == E_2.size:
        return phi, n_frozen, 0

    safe = xp.where(bad, xp.ones_like(E_2), E_2)
    step = xp.where(bad, xp.zeros_like(E_1), -E_1 / safe)

    clipped = xp.abs(step) > step_max
    n_clipped = int(backend.sum_real(clipped.astype(backend.accum_real)))
    step = xp.clip(step, -step_max, step_max)

    return phi + step, n_frozen, n_clipped


def stop_criterion(backend: Backend, phi_new, phi_old) -> float:
    """(5-9), шаг 9 §5.8: max_k |exp(j phi^(l+1)_k) - exp(j phi^(l)_k)|.

    Принимает бэкенд, новую и старую фазу; возвращает одно вещественное число,
    которое сравнивается с порогом mu.

    Максимум по всем бинам: останов наступает, когда успокоился самый
    беспокойный бин. Форма с экспонентами выбрана потому, что фаза определена
    с точностью до 2 pi и разворачивать её не требуется (§5.7 документа):
    |e^{j phi_1} - e^{j phi_2}| = 2 |sin(delta phi / 2)| ~ |delta phi|.

    В книге (5-9) напечатано без модуля — см. §8 документа: комплексная
    величина сравнивается с вещественным порогом.
    """
    xp = backend.xp
    diff = xp.exp(1j * phi_new) - xp.exp(1j * phi_old)
    # РАСХОЖДЕНИЕ (5-9): в книге напечатано иначе, см. §8 — без модуля |.|
    # комплексная величина сравнивалась бы с вещественным порогом.
    return float(xp.max(xp.abs(diff)))


def iterate_block(
    backend: Backend,
    h,
    phi_initial,
    mu: float = MU_DEFAULT,
    max_iterations: int = MAX_ITERATIONS,
    floor_relative: float = POWER_FLOOR_RELATIVE,
    curvature_policy: str = "freeze",
    step_max: float = STEP_MAX_RAD,
) -> BlockIterations:
    """Цикл §5.8 целиком: девять шагов, два азимутальных БПФ размера M на
    итерацию и ничего больше.

    Принимает бэкенд, данные блока h(k,n), начальную фазу phi^(0) из этапа B,
    порог останова mu (5-9) и предел числа итераций; возвращает
    BlockIterations с найденным phi_k и всеми числами для отчёта.

    Счётчики БПФ снимаются вокруг тела каждой итерации: список
    fft_per_iteration обязан состоять из одних двоек, а w_per_iteration —
    из одних единиц (§10.6, §10.7 задания).
    """
    h = backend.asarray(h)
    phi = backend.asarray(np.asarray(phi_initial, dtype=np.float64)).astype(
        backend.real_dtype, copy=False
    )
    phi_initial_out = backend.to_numpy(phi).copy()

    M, N = h.shape
    h_abs2 = backend.xp.abs(h) ** 2  # нужен (5-19) на каждой итерации, считаем один раз
    S_g = total_energy(backend, h_abs2, M)  # (5-4)
    floor = power_floor(S_g, M, N, floor_relative)  # решение реализации №1

    out = BlockIterations(
        phi=phi_initial_out,
        phi_initial=phi_initial_out,
        n_iterations=0,
        converged=False,
        stop_reason="",
        S_g=S_g,
    )

    floored_fraction = 0.0
    for _ in range(max_iterations):
        before = backend.counters.snapshot()

        h_phi, g = image_from_phase(backend, h, phi)  # шаги 1-2, (5-3)
        P, ln_P, floored_fraction = image_power(backend, g, floor)  # шаг 3
        E_g, S = entropy(backend, P, S_g)  # (5-5),(5-6) — для отчёта
        G = auxiliary_array(backend, ln_P, g)  # шаг 4, (5-14)
        W = w_product(backend, G, h_phi)  # шаг 5 — один раз
        E_1 = first_derivative(backend, W)  # шаг 6, (5-13)
        E_2 = second_derivative(backend, W, h_abs2, ln_P)  # шаг 7, (5-19)

        try:
            phi_new, n_frozen, n_clipped = newton_update(  # шаг 8, (5-8)
                phi, E_1, E_2, backend, curvature_policy, step_max
            )
        except CurvatureRefusal as exc:
            out.stop_reason = f"отказ по решению №2: {exc}"
            out.entropy_history.append(E_g)
            out.normalised_entropy_history.append(S)
            break

        criterion = stop_criterion(backend, phi_new, phi)  # шаг 9, (5-9)

        after = backend.counters.snapshot()
        out.fft_per_iteration.append((after[0] - before[0]) + (after[1] - before[1]))
        out.w_per_iteration.append(after[2] - before[2])
        out.entropy_history.append(E_g)
        out.normalised_entropy_history.append(S)
        out.criterion_history.append(criterion)
        out.frozen_bins_history.append(n_frozen)
        out.clipped_bins_history.append(n_clipped)

        phi = phi_new
        out.n_iterations += 1

        if n_frozen == M:
            out.stop_reason = (
                f"(5-19) дала E'' <= 0 во всех {M} бинах: квадратичная модель "
                "не имеет минимума, двигаться некуда — это НЕ сходимость"
            )
            break
        if criterion <= mu:
            out.converged = True
            out.stop_reason = f"(5-9) выполнено: {criterion:.3e} <= mu = {mu:g}"
            break
    else:
        out.stop_reason = (
            f"упёрлись в предел {max_iterations} итераций, (5-9) не выполнено "
            f"(последнее значение {out.criterion_history[-1]:.3e} > mu = {mu:g}) — "
            "это НЕ сходимость"
        )

    out.phi = backend.to_numpy(phi).astype(np.float64)
    out.floored_fraction = floored_fraction

    # Итоговая энтропия — ещё одно (5-3), уже ВНЕ цикла: это то же вычисление,
    # которое делает этап D, и в стоимость итерации оно не входит.
    _, g_final = image_from_phase(backend, h, phi)
    P_final, _, _ = image_power(backend, g_final, floor)
    out.entropy_final, out.normalised_entropy_final = entropy(backend, P_final, S_g)
    return out

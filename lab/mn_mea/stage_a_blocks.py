"""Этап A. Разбиение сцены на блоки. Формулы (5-20)…(5-26), §3 документа.

Порядок функций — порядок выполнения:

    motion_constants                (4-5),(4-7)   A_1, A_2, mu_3
    linearisation_coefficients      (4-83),(4-84) шесть Delta k
    straight_flight_coefficients    §2.2          проверочная ветка
    slant_range_change              (4-23),(4-82) Delta R
    block_half_sizes                (5-20)        x_p, y_p
    block_sample_sizes              (5-22),(5-23) m_p, n_p
    block_counts                    (5-24),(5-25),(5-21)  M_k, N_k, q
    block_number                    (5-26)        q_k
    block_grid                                    сетка блоков целиком

Что требуется извне: объёмные параметры движения v, R_B0, геометрия (§2.1).
ИНС не нужна ни на одном этапе. Книга берёт скорость ОДНИМ ЧИСЛОМ на всю
апертуру — это допущение книги, см. README, раздел «Допущение книги о скорости».
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Объёмные параметры движения и сигнала — вход этапов A и B (§2.1).

    Все скорости и ускорения — постоянные на всю апертуру, по одному числу.
    """

    v_x0: float
    v_y0: float
    v_z0: float
    a_x: float
    a_y: float
    a_z: float
    x_m: float
    y_m: float
    z_m: float
    R_B0: float
    lambda_: float
    f_0: float
    r_a: float
    r_b: float
    T_a: float
    c: float = 299_792_458.0


@dataclass(frozen=True)
class Block:
    """Один блок: сквозной номер q_k по (5-26), его место НА СЦЕНЕ и
    координаты его центра относительно центра сцены, в метрах.

    Границы даны в пикселях ИЗОБРАЖЕНИЯ: m_start:m_stop по азимуту,
    n_start:n_stop по дальности. Именно так требует §3.3 документа —
    «блок занимает [-x_p, +x_p] по азимуту и [-y_p, +y_p] по дальности»,
    где x_p, y_p это метры на местности, а (5-22)/(5-23) переводят их в
    отсчёты изображения.
    """

    q_k: int
    m_k: int
    n_k: int
    m_start: int
    m_stop: int
    n_start: int
    n_stop: int
    x_centre: float
    y_centre: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.m_stop - self.m_start, self.n_stop - self.n_start)


def motion_constants(geom: Geometry) -> tuple[float, float, float]:
    """(4-5),(4-7) через §2.1: A_1, A_2, mu_3 — три объёмные постоянные.

    Принимает Geometry; возвращает (A_1, A_2, mu_3).

        A_1 = x_m v_x0 + y_m v_y0 + z_m v_z0
        A_2 = x_m a_x + y_m a_y + z_m a_z + v_x0^2 + v_y0^2 + v_z0^2
        mu_3 = v_x0 a_x + v_y0 a_y + v_z0 a_z
    """
    A_1 = geom.x_m * geom.v_x0 + geom.y_m * geom.v_y0 + geom.z_m * geom.v_z0
    A_2 = (
        geom.x_m * geom.a_x
        + geom.y_m * geom.a_y
        + geom.z_m * geom.a_z
        + geom.v_x0**2
        + geom.v_y0**2
        + geom.v_z0**2
    )
    mu_3 = geom.v_x0 * geom.a_x + geom.v_y0 * geom.a_y + geom.v_z0 * geom.a_z
    return A_1, A_2, mu_3


def linearisation_coefficients(geom: Geometry) -> dict[str, float]:
    """(4-83) по азимуту и (4-84) по дальности: шесть коэффициентов
    пространственной изменчивости хода дальности.

    Принимает Geometry; возвращает словарь с ключами
    'k1x','k2x','k3x','k1y','k2y','k3y'.

    Эти же коэффициенты используются этапом B (§4.1 документа).
    """
    A_1, A_2, mu_3 = motion_constants(geom)
    R = geom.R_B0
    R3 = R**3

    # (4-83), по азимуту
    k1x = -geom.v_x0 / R + A_1 * geom.x_m / R3
    k2x = -geom.a_x / (2 * R) + A_2 * geom.x_m / (2 * R3) + A_1 * geom.v_x0 / R3
    k3x = A_2 * geom.v_x0 / (2 * R3) + A_1 * geom.a_x / (2 * R3) + mu_3 * geom.x_m / (2 * R3)

    # (4-84), по дальности
    k1y = -geom.v_y0 / R + A_1 * geom.y_m / R3
    k2y = -geom.a_y / (2 * R) + A_2 * geom.y_m / (2 * R3) + A_1 * geom.v_y0 / R3
    k3y = A_2 * geom.v_y0 / (2 * R3) + A_1 * geom.a_y / (2 * R3) + mu_3 * geom.y_m / (2 * R3)

    return {"k1x": k1x, "k2x": k2x, "k3x": k3x, "k1y": k1y, "k2y": k2y, "k3y": k3y}


def straight_flight_coefficients(v: float, R_B0: float, y_m: float) -> dict[str, float]:
    """§2.2 документа: во что превращаются шесть коэффициентов при
    прямолинейном полёте (v_x0 = v, v_y0 = v_z0 = 0, ускорения нули, x_m = 0).

    Принимает v, R_B0, y_m; возвращает те же шесть ключей, что и
    linearisation_coefficients. Служит ПРОВЕРКОЙ общего случая: подставили
    прямолинейный полёт в общие формулы — обязаны получить ровно эти
    три числа и три нуля.

        Delta k_1x = -v / R_B0
        Delta k_3x = v^3 / (2 R_B0^3)
        Delta k_2y = v^2 y_m / (2 R_B0^3)
        Delta k_2x = Delta k_1y = Delta k_3y = 0
    """
    return {
        "k1x": -v / R_B0,
        "k2x": 0.0,
        "k3x": v**3 / (2 * R_B0**3),
        "k1y": 0.0,
        "k2y": v**2 * y_m / (2 * R_B0**3),
        "k3y": 0.0,
    }


def slant_range_change(coef: dict[str, float], t_a, x_p, y_p):
    """(4-23) с подстановкой (4-82): пространственно-изменчивое изменение
    хода дальности в точке (x_p, y_p) на медленном времени t_a.

    Принимает коэффициенты (4-83)/(4-84), t_a, x_p, y_p;
    возвращает Delta R в метрах.

        Delta k_j = Delta k_jx x_p + Delta k_jy y_p            (4-82)
        Delta R   = Delta k_1 t_a + Delta k_2 t_a^2 + Delta k_3 t_a^3   (4-23)
    """
    dk1 = coef["k1x"] * x_p + coef["k1y"] * y_p
    dk2 = coef["k2x"] * x_p + coef["k2y"] * y_p
    dk3 = coef["k3x"] * x_p + coef["k3y"] * y_p
    return dk1 * t_a + dk2 * t_a**2 + dk3 * t_a**3


def block_half_sizes(coef: dict[str, float], geom: Geometry) -> tuple[float, float]:
    """(5-20): |Delta R(t_a; x_p, y_p)| <= r_a / 2 на краю апертуры t_a = T_a/2.

    Принимает коэффициенты (4-83)/(4-84) и Geometry; возвращает (x_p, y_p) —
    полуразмеры блока в метрах: блок занимает [-x_p, +x_p] по азимуту и
    [-y_p, +y_p] по дальности.

    По §3.3 критерий распадается на два независимых неравенства: смещение по
    азимуту ограничивает линейный член через Delta k_1x, смещение по дальности —
    квадратичный через Delta k_2y. Кубический член на порядок меньше и в
    размер блока не входит.

        |Delta k_1x x_p| (T_a/2)   <= r_a/2  =>  x_p <= r_a / (|Delta k_1x| T_a)
        |Delta k_2y y_p| (T_a/2)^2 <= r_a/2  =>  y_p <= 2 r_a / (|Delta k_2y| T_a^2)

    В книге (5-20) напечатано без модуля и без указания t_a — см. §8 документа:
    при отрицательном Delta R условие без модуля пропускает любую по модулю
    ошибку, а изменчивость максимальна именно на краю апертуры.
    """
    # РАСХОЖДЕНИЕ (5-20)/t_a: в книге напечатано иначе, см. §8 — t_a не указано,
    # а изменчивость максимальна на краю апертуры.
    t_edge = geom.T_a / 2.0

    # РАСХОЖДЕНИЕ (5-20)/модуль: в книге напечатано иначе, см. §8 — без |Delta R|
    # условие при отрицательном Delta R пропускает любую по модулю ошибку.
    dk1x, dk2y = abs(coef["k1x"]), abs(coef["k2y"])

    x_p = math.inf if dk1x == 0.0 else (geom.r_a / 2.0) / (dk1x * t_edge)
    y_p = math.inf if dk2y == 0.0 else (geom.r_a / 2.0) / (dk2y * t_edge**2)
    return x_p, y_p


def block_sample_sizes(x_p: float, y_p: float, geom: Geometry) -> tuple[float, float]:
    """(5-22) m_p = 2 x_p / r_a и (5-23) n_p = 2 y_p / r_b — перевод
    полуразмеров блока в отсчёты.

    Принимает (x_p, y_p) в метрах и Geometry; возвращает (m_p, n_p) в отсчётах.

    В книге (5-23) напечатано как n_p = 2 Delta R / r_b — см. §8 документа:
    по симметрии с (5-22) там должно стоять y_p, иначе получается n_p <= 1,
    то есть блок в один строб дальности.
    """
    m_p = 2.0 * x_p / geom.r_a
    # РАСХОЖДЕНИЕ (5-23): в книге напечатано иначе, см. §8 — там 2 Delta R / r_b,
    # с чем получилось бы n_p <= 1, то есть блок в один строб дальности.
    n_p = 2.0 * y_p / geom.r_b
    return m_p, n_p


def block_counts(M: int, N: int, m_p: float, n_p: float) -> tuple[int, int, int]:
    """(5-24) M_k = ceil(M/m_p), (5-25) N_k = ceil(N/n_p), (5-21) q = M_k N_k.

    Принимает размеры массива M, N и размеры блока в отсчётах m_p, n_p;
    возвращает (M_k, N_k, q).

    Края (§7.5 задания): m_p = inf (нет азимутальной изменчивости) даёт
    M_k = 1; m_p < 1 дало бы блок короче отсчёта — число блоков ограничено
    сверху числом отсчётов, больше M блоков по азимуту не бывает.
    """
    M_k = 1 if not math.isfinite(m_p) else min(M, max(1, math.ceil(M / m_p)))
    N_k = 1 if not math.isfinite(n_p) else min(N, max(1, math.ceil(N / n_p)))
    return M_k, N_k, M_k * N_k


def block_number(m_k: int, M_k: int, n_k: int) -> int:
    """(5-26) q_k = m_k + M_k (n_k - 1) — сквозной номер блока.

    Принимает m_k = 1…M_k, M_k и n_k = 1…N_k; возвращает q_k = 1…q.
    Обычная развёртка двумерной сетки: индекс азимута меняется быстрее.

    §3.7 документа: при обработке поячеечно по дальности N_k = 1, тогда
    n_k = 1, второе слагаемое обнуляется и q_k = m_k — разбиение чисто
    азимутальное.
    """
    return m_k + M_k * (n_k - 1)


def block_grid(M: int, N: int, M_k: int, N_k: int, geom: Geometry) -> list[Block]:
    """Сетка блоков целиком: нарезка СЦЕНЫ на M_k x N_k участков и нумерация
    каждого по (5-26).

    Принимает размеры изображения, число блоков по осям и Geometry;
    возвращает список Block, упорядоченный по q_k.

    Режется ИЗОБРАЖЕНИЕ по азимутальным пикселям m, а не массив данных по
    доплеровским бинам k. Основание — §3.3 документа: блок занимает
    [-x_p, +x_p] по азимуту, где x_p это расстояние на местности, и (5-22)
    переводит его в пиксели изображения. Цель, стоящая на местности,
    принадлежит ровно одному блоку — тому, на чьём участке она находится.

    Раскладка следует из (5-24)/(5-25) буквально: там стоит ПОТОЛОК, а потолок
    нужен ровно тогда, когда блоки берутся номинального размера, а последний
    получает остаток. Поэтому номинальный размер блока равен ceil(M/M_k), а
    последняя плитка по каждой оси обрезается (§7.5 задания, «блок, попавший
    на границу и обрезанный»). Все M_k x N_k блоков существуют и покрывают
    массив без нахлёста и без пропусков.

    Центр блока отсчитывается от центра сцены и переводится в метры через
    r_a и r_b.
    """
    k_size = math.ceil(M / M_k)
    n_size = math.ceil(N / N_k)
    k_edges = [min(i * k_size, M) for i in range(M_k + 1)]
    n_edges = [min(j * n_size, N) for j in range(N_k + 1)]

    blocks: list[Block] = []
    for n_k in range(1, N_k + 1):
        for m_k in range(1, M_k + 1):
            k0, k1 = k_edges[m_k - 1], k_edges[m_k]
            n0, n1 = n_edges[n_k - 1], n_edges[n_k]
            blocks.append(
                Block(
                    q_k=block_number(m_k, M_k, n_k),
                    m_k=m_k,
                    n_k=n_k,
                    m_start=k0,
                    m_stop=k1,
                    n_start=n0,
                    n_stop=n1,
                    x_centre=((k0 + k1) / 2.0 - M / 2.0) * geom.r_a,
                    y_centre=((n0 + n1) / 2.0 - N / 2.0) * geom.r_b,
                )
            )
    blocks.sort(key=lambda b: b.q_k)
    return blocks


def scene_image(backend, h):
    """Полное изображение сцены: (5-3) при phi = 0, по ВСЕМ M доплеровским бинам.

    Принимает бэкенд и данные сцены h(k,n); возвращает изображение g(m,n).

    Это то, что дальше режется на блоки. Считается ОДИН РАЗ на всю сцену,
    до цикла по блокам: полоса доплеровских частот у всех блоков одна и та
    же — вся, — различаются они участком местности.
    """
    return backend.fft_kernel_minus(backend.asarray(h))


def block_data(backend, g_scene, block: Block):
    """Данные блока h_qk(k,n) — то, что этап C получает на вход.

    Принимает бэкенд, полное изображение сцены и Block;
    возвращает массив (M, n_p): ПОЛНАЯ доплеровская ось и стробы дальности
    этого блока.

    Два шага. Из изображения берётся участок местности этого блока, всё
    остальное обнуляется, — так энтропия (5-5) считается только по своему
    куску сцены. Потом окно возвращается в область дальность-доплер по всем
    M бинам.

    Ось остаётся ПОЛНОЙ намеренно. Поправка phi_k — свойство носителя на
    всей апертуре, и вектор (5-3) обязан иметь длину M, сколько бы пикселей
    ни занимал блок. Блок выбирает, ПО ЧЕМУ считается энтропия, а не по
    скольким бинам ищется фаза.

    Проверяется это замером, а не рассуждением: при таком способе энтропия в
    ИСТИННОЙ фазе оказывается наименьшей, а при укорочении оси до m_p —
    хуже, чем при phi = 0. См. README, «Как читается нарезка на блоки».
    """
    окно = backend.xp.zeros_like(g_scene)
    окно[block.m_start : block.m_stop, block.n_start : block.n_stop] = g_scene[
        block.m_start : block.m_stop, block.n_start : block.n_stop
    ]
    данные = backend.xp.fft.ifft(окно, axis=0) / backend.alpha
    return данные[:, block.n_start : block.n_stop]


def centre_block_index(M_k: int, N_k: int) -> int:
    """§4.4 документа: номер блока, содержащего центр сцены. Книга берёт его
    как M_k N_k / 2 либо (M_k N_k - 1) / 2 — то из двух, что целое.

    Принимает M_k, N_k; возвращает q_k центрального блока.
    Нужен этапу B для знаменателя (5-29).
    """
    q = M_k * N_k
    return q // 2 if q % 2 == 0 else (q - 1) // 2

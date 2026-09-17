"""Линейка размаза: во сколько раз отклик цели шире, чем должен быть.

Нужна потому, что «размаз как был, так и остался» — суждение глазом, а
спорить об изображении глазами нельзя. Здесь то же самое числом, и числа
подобраны так, чтобы отвечать именно тому, что глаз и видит.

Что меряется у каждой яркой цели:

    ширина по -3 дБ     главный лепесток. Для сфокусированной цели равна
                        0,886 элемента разрешения — это свойство синка, а
                        не подгонка
    ширина по -10 дБ    и по -20 дБ: сюда попадают хвосты, и ИМЕННО их
                        видно глазом как размаз. Главный лепесток шириной
                        в 7 отсчётов на графике занимает 0,7 пикселя и
                        невидим, а хвост в 300 отсчётов виден отлично
    ISLR                доля энергии ВНЕ главного лепестка, в децибелах.
                        Одно число вместо трёх ширин: у сфокусированной
                        цели с прямоугольным окном около -10 дБ, у
                        расфокусированной уходит к нулю и выше

Порядок функций — порядок выполнения:

    find_targets        найти яркие одиночные цели
    azimuth_cut         вырезать КОМПЛЕКСНЫЙ срез по азимуту вокруг цели
    upsample            уплотнить срез (validate.interpolated_cut) и взять модуль
    width_at            ширина на заданном уровне, в элементах разрешения
    islr                доля энергии вне главного лепестка
    measure_target      все числа по одной цели
    measure             таблица по картинке
    compare             та же таблица для двух картинок, до и после
"""

from __future__ import annotations

import numpy as np

from validate import interpolated_cut

#: Во сколько раз уплотняется срез перед замером. Отклик занимает около
#: семи отсчётов, и мерить по ним ширину на уровне -3 дБ значит округлять
#: до 14 %. Шестнадцатикратное уплотнение доводит шаг до 0,06 отсчёта.
#:
#: Уплотнение делает validate.interpolated_cut — точное восстановление
#: через (5-3), с ЦЕНТРИРОВАННЫМ номером бина. Своё писать не надо, и
#: первая попытка это подтвердила: я дописал нули в конец спектра и брал
#: модуль ДО уплотнения. Оба шага неверны — модуль не ограничен по полосе,
#: а нули в конце объявляют частоты бинов натуральными. Линейка показывала
#: ширину 0,076 элемента вместо теоретических 0,886 и не реагировала на
#: расфокусировку вовсе.
OVERSAMPLE = 16

#: Полуширина окна замера, в элементах разрешения. Хвост, который видно
#: глазом, тянется на десятки элементов; за 24 он уже теряется в фоне.
WINDOW_CELLS = 24.0

#: Полуширина главного лепестка при счёте ISLR, в элементах разрешения.
#: Первый ноль синка стоит ровно на одном элементе разрешения от пика.
MAINLOBE_CELLS = 1.0

#: Насколько цели должны быть разнесены, чтобы считаться одиночными:
#: по азимуту в элементах разрешения, по дальности в стробах.
GUARD_CELLS = 8.0
GUARD_GATES = 4


def find_targets(image: np.ndarray, cells: float, count: int,
                 window: int) -> list[tuple[int, int]]:
    """Найти яркие одиночные цели.

    Принимает картинку (M, N), ширину элемента разрешения в отсчётах, число
    целей и полуширину окна замера; возвращает список (m, n).

    Жадно: берётся самый яркий пиксель, вокруг него всё гасится, берётся
    следующий. Гашение и есть условие одиночности — две цели ближе GUARD
    считаются одной, и мерить у них ширину бессмысленно.

    Цели у самого края по азимуту пропускаются: окно замера должно целиком
    лежать в картинке, иначе ширина занизится обрезанием хвоста.
    """
    power = np.abs(np.asarray(image)) ** 2
    M, N = power.shape
    guard_m = max(int(round(GUARD_CELLS * cells)), 1)
    found: list[tuple[int, int]] = []

    work = power.copy()
    work[:window, :] = 0.0
    work[M - window:, :] = 0.0
    for _ in range(count):
        if not work.any():
            break
        m, n = np.unravel_index(int(np.argmax(work)), work.shape)
        found.append((int(m), int(n)))
        lo_m, hi_m = max(m - guard_m, 0), min(m + guard_m + 1, M)
        lo_n, hi_n = max(n - GUARD_GATES, 0), min(n + GUARD_GATES + 1, N)
        work[lo_m:hi_m, lo_n:hi_n] = 0.0
    return found


def azimuth_cut(image: np.ndarray, m: int, n: int, window: int) -> np.ndarray:
    """КОМПЛЕКСНЫЙ срез по азимуту вокруг цели, длиной 2*window.

    Комплексный, а не модуль: уплотнять надо сигнал, а не его модуль. У
    модуля спектр не ограничен, и дополнение нулями для него — уже не
    точное восстановление, а выдумка.
    """
    return np.asarray(image)[m - window : m + window, n]


def upsample(cut: np.ndarray, factor: int = OVERSAMPLE) -> np.ndarray:
    """Уплотнить срез и вернуть модуль.

    Принимает комплексный срез и кратность; возвращает вещественный массив
    в factor раз длиннее.

    Вся работа — в validate.interpolated_cut: восстановление точное, через
    (5-3), с центрированным номером бина, и проверено против прямой суммы
    по определению (см. её докстроку).
    """
    return np.abs(interpolated_cut(np.asarray(cut), factor))


def width_at(profile: np.ndarray, level_db: float, cells: float,
             factor: int = OVERSAMPLE) -> float:
    """Ширина отклика на заданном уровне, в ЭЛЕМЕНТАХ РАЗРЕШЕНИЯ.

    Принимает уплотнённый профиль, уровень в дБ (отрицательный), ширину
    элемента разрешения в отсчётах и кратность уплотнения; возвращает
    ширину.

    Считается по пересечениям уровня слева и справа от пика, с линейной
    доводкой между соседними точками уплотнённого профиля. В элементах
    разрешения, а не в отсчётах и не в метрах, потому что осмысленно
    сравнивать надо с единицей: у сфокусированной цели ширина по -3 дБ
    равна 0,886 элемента, и всякое «в полтора раза шире» видно сразу.
    """
    peak = int(np.argmax(profile))
    threshold = profile[peak] * 10.0 ** (level_db / 20.0)

    def crossing(direction: int) -> float:
        i = peak
        while 0 < i < profile.size - 1 and profile[i] > threshold:
            i += direction
        if profile[i] >= threshold:            # уровень не пересечён
            return float(abs(i - peak))
        a, b = profile[i - direction], profile[i]
        part = (a - threshold) / max(a - b, 1e-300)
        return float(abs(i - direction - peak) + part)

    return (crossing(-1) + crossing(+1)) / (cells * factor)


def islr(profile: np.ndarray, cells: float, factor: int = OVERSAMPLE) -> float:
    """Доля энергии ВНЕ главного лепестка, в децибелах.

    Принимает уплотнённый профиль, ширину элемента разрешения в отсчётах и
    кратность; возвращает 10 lg (энергия снаружи / энергия внутри).

    Главным лепестком считается +-MAINLOBE_CELLS элемента разрешения от
    пика: первый ноль синка стоит ровно там. Число тем выше, чем больше
    энергии ушло в хвосты, — это и есть размаз одним числом.
    """
    peak = int(np.argmax(profile))
    half = int(round(MAINLOBE_CELLS * cells * factor))
    power = profile.astype(np.float64) ** 2
    inside = power[max(peak - half, 0) : peak + half + 1].sum()
    return 10.0 * np.log10(max(power.sum() - inside, 1e-300) / max(inside, 1e-300))


def measure_target(image: np.ndarray, m: int, n: int, cells: float,
                   window: int) -> dict:
    """Все числа по одной цели."""
    cut = azimuth_cut(image, m, n, window)
    profile = upsample(cut)
    peak = float(profile.max())
    return {
        "m": m, "n": n,
        "peak_db": 20.0 * np.log10(max(peak, 1e-300)),
        "width_3db": width_at(profile, -3.0, cells),
        "width_10db": width_at(profile, -10.0, cells),
        "width_20db": width_at(profile, -20.0, cells),
        "islr_db": islr(profile, cells),
        "profile": profile,
    }


def measure(image: np.ndarray, cells: float, count: int = 8) -> list[dict]:
    """Таблица по картинке: count самых ярких одиночных целей."""
    window = int(round(WINDOW_CELLS * cells))
    return [measure_target(image, m, n, cells, window)
            for m, n in find_targets(image, cells, count, window)]


def compare(before: np.ndarray, after: np.ndarray, cells: float,
            count: int = 8) -> list[dict]:
    """Та же таблица для двух картинок, по ОДНИМ И ТЕМ ЖЕ целям.

    Принимает две картинки, ширину элемента разрешения в отсчётах и число
    целей; возвращает список словарей с ключами 'before' и 'after'.

    Цели ищутся на картинке ДО и меряются в обеих на том же месте. Искать
    отдельно в каждой нельзя: тогда в сравнение попали бы разные цели, и
    таблица сказала бы что угодно.
    """
    window = int(round(WINDOW_CELLS * cells))
    rows = []
    for m, n in find_targets(before, cells, count, window):
        rows.append({
            "m": m, "n": n,
            "before": measure_target(before, m, n, cells, window),
            "after": measure_target(after, m, n, cells, window),
        })
    return rows

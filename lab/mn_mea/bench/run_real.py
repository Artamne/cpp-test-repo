"""Прогон MN-MEA на срезе РЕАЛЬНОЙ записи, подготовленном prepare_slice.py.

    python3 bench/run_real.py путь/к/mn_mea_srez [M_k x N_k] [mu] [проба]
    python3 bench/run_real.py путь/к/mn_mea_srez 25x1 iter=200 faza=luchshaya

Читает h_srez.npy и meta.json, проверяет вход, считает, кладёт рядом
картинки до и после и числа. Сетку блоков можно задать вторым аргументом,
например 8x1; без него берётся (5-20) по геометрии из паспорта. Третьим —
порог останова (5-9), по умолчанию MU_REAL.

ИМЕНОВАННЫЕ АРГУМЕНТЫ, их можно мешать с позиционными в любом порядке:

    setka=25x1      сетка блоков, то же что первым позиционным
    mu=0.03         порог останова (5-9)
    proba=0.8       размах пробы в радианах, см. ниже
    iter=200        предел числа итераций вместо книжных 40
    faza=luchshaya  ОДНА фаза на всю сцену, взятая у блока, где энтропия
                    упала сильнее всех. По умолчанию faza=svoya — каждый
                    блок со своей, как в книге
    celi=0.05       решение №9: критерий (5-5) считается только по
                    окрестностям целей, а не по всему блоку. На местности
                    без этого блок возвращает ноль: фон давит цели. По
                    умолчанию выключено, поведение книжное
    start=polinom   ЭТАП B БЕЗ ИНС (по умолчанию): начальная фаза phi^(0)
                    ищется как полином по энтропии всей сцены
                    (stage_b_search.py), и этап C идёт из неё. start=nol —
                    этап C из нуля, как было; на настоящей записи так не
                    работает, см. stage_b_search

ЧЕТВЁРТЫЙ АРГУМЕНТ — ПРОБА, и он отвечает на вопрос, который иначе не
решается. Если прогон почти ничего не меняет, причин ровно две: остаточной
фазы в данных и правда нет, или она есть, но алгоритм её на этой сцене не
видит. Различить их можно, ВНЕСЯ известную ошибку в сами данные и посмотрев,
найдёт ли он её. Проба задаётся размахом в радианах на краю занятой полосы,
например 3; тогда печатается, какую долю внесённого удалось снять.

Проба — диагностика, а не обработка: её результат в картинку не идёт.

ЧЕМ ЭТОТ ПРОГОН ОТЛИЧАЕТСЯ ОТ СТЕНДА, и это главное.

На синтетике есть ИСТИНА: внесённую фазу знаем мы сами, и «главное число»
отчёта — расхождение с ней. На записи истины нет ни у кого. Значит сказать
«фаза найдена верно» здесь НЕЛЬЗЯ, и никакое число этого не скажет.

Что можно сказать: картинка стала резче или нет. Мерится двумя числами —
нормированной энтропией (5-6) и контрастом. Энтропия при этом не является
независимой проверкой: алгоритм её и минимизирует, она обязана упасть.
Независимо смотрит контраст, и он же на графиках — глазами.

Объективно сверить можно НАРЕЗКУ: если prepare_slice.py положил в паспорт
раздел ozhidaemaya_narezka, размер блока и число блоков сличаются с ним
(grid_agreement). Это единственная проверка, у которой тут есть ответ.

Порядок функций — порядок выполнения:

    read_slice          прочитать куски h_srez*.npy и meta.json, склеить
    geometry_from_meta  Geometry из паспорта, разрешение отдельно от шага
    focus               блоки, этап B без ИНС, этап C, этап D
    grid_agreement      сверка нарезки с ожиданием паспорта
    contrast            число, которое алгоритм НЕ минимизирует
    main                всё вместе, с печатью и картинкой
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import sys

_sys_path = pathlib.Path(__file__).resolve().parent.parent / "algorithm"
sys.path.insert(0, str(_sys_path))

import numpy as np

import inputs
import stage_a_blocks as A
import stage_b_search as BS
import stage_c_iterate as C
import stage_d_assemble as D
from backend import get_backend
from stage_a_blocks import Geometry
import smear
from validate import image_sharpness, phase_residual_rms, truth_on_block_grid


#: Процентили для границ показа. Не нормировка по максимуму: см.
#: display_window, там замер, почему максимум даёт чёрную картинку.
DISPLAY_LOW, DISPLAY_HIGH = 5.0, 97.0

#: Сколько самых ярких одиночных целей меряет линейка размаза. Восемь —
#: чтобы медиана была устойчива к тому, что одна-две «цели» окажутся
#: случайными выбросами спекла, и при этом таблица помещалась на экран.
SMEAR_TARGETS = 8

#: Сколько целей из них рисуется срезами. Четыре помещаются в строку и
#: читаются; больше — уже мелко.
SMEAR_CUTS = 4

#: Полуокно крупного плана вокруг цели, в элементах разрешения. На общем
#: плане кадра 11000 строк ужаты в 600 пикселей — 18 строк на пиксель, и
#: сужение пятна с 50 до 30 отсчётов там не видно в принципе. Крупный план
#: 1:1: +-12 элементов = +-84 строки накрывают хвост даже у самой широкой
#: цели (-20 дБ до 13 элементов до автофокуса). По дальности берётся
#: столько же МЕТРОВ, чтобы квадрат на экране был квадратом на земле.
KRUPNO_CELLS = 12.0


#: Контраст однородного полностью развитого спекла. Не подгонка, а точное
#: значение: у комплексно-гауссова поля интенсивность распределена
#: экспоненциально, <I^2> = 2<I>^2, значит sqrt(<I^2>)/<I> = sqrt(2). Ниже
#: этого числа контраст на радиолокационной картинке не опускается, и мерить
#: резкость имеет смысл в долях от него, а не в абсолюте.
SPECKLE_CONTRAST = math.sqrt(2.0)

#: Ниже какого контраста ДО автофокуса говорить, что фокусировать нечего.
#: Вдвое над спеклом — граница мягкая и намеренно низкая: она ловит не
#: «плохую» сцену, а срез, встающий на пустой участок записи. Замер на второй
#: записи владельца: контраст 2,88, то есть 2,0 спекла, и весь прогон дал
#: 2,88 -> 2,89 при девяти блоках, сошедшихся за 1-2 итерации. Для сравнения,
#: на первой его записи было 22,40 — это 15,8 спекла.
SPECKLE_FLOOR_WARN = 2 * SPECKLE_CONTRAST


#: Какую долю внесённой пробы надо снять, чтобы считать, что алгоритм на
#: этих данных работает. Половина — намеренно мягко: проба отвечает на
#: вопрос «видит или не видит», а не меряет точность.
PROBE_FOUND_SHARE = 0.5


#: Выше какого расширения весового окна говорить, что что-то не так. 2,0 —
#: заведомо выше всех употребимых окон: у Блэкмана 1,68, тяжелее в РСА не
#: берут, потому что платят разрешением.
BAND_GAMMA_MAX = 2.0

#: На сколько замеренная gamma может отличаться от той, с которой считан
#: срез, прежде чем советовать менять сетку. 15 % — меньше, чем шаг между
#: соседними целыми числами блоков на любой разумной сетке.
BAND_GAMMA_TOLERANCE = 0.15


#: Порог останова (5-9) для прогона на записи. НЕ равен MU_MEASURED стенда.
#:
#: Стенд берёт 1e-3, и для него это верно: там меряют точность, а не время.
#: Его же таблица (отчёт, решение №4) показывает, где на самом деле колено:
#:
#:   mu     итераций   остаток, рад
#:   0,1       10       1,118e-01
#:   0,03      13       5,781e-02     <- точность уже полная
#:   0,01      15       5,803e-02
#:   0,001     18       5,893e-02
#:
#: На записи разница куда заметнее, потому что блок больше. Замер на срезе
#: 457x200 с занятой полосой 12,5 %:
#:
#:   mu      итераций  сошлось   S итог
#:   0,1        9        да      6,9974
#:   0,03      12        да      6,9811
#:   0,001     40        НЕТ     6,9787
#:
#: То есть 1e-3 стоит вчетверо больше итераций и приносит 0,003 энтропии —
#: три сотых процента. И, что хуже, все блоки докладывают «не сошлось», а
#: этот признак нужен как диагностика, а не как всегдашняя надпись.
MU_REAL = 0.03


def parse_options(words: list[str]) -> tuple:
    """Разобрать аргументы после пути.

    Принимает слова; возвращает (сетка, mu, проба, предел итераций, общая ли
    фаза, доля целей, откуда стартовать).

    Позиционные и именованные мешаются свободно, и правило простое: слово с
    '=' — именованное; слово вида ЧИСЛОxЧИСЛО — сетка; голое число — сперва
    mu, потом проба. Позиционный вид оставлен потому, что им уже пользуются;
    именованный добавлен потому, что настроек стало пять и на память их
    порядок уже не ложится.
    """
    grid, mu, probe = None, MU_REAL, 0.0
    iterations, shared, targets = C.MAX_ITERATIONS, False, None
    start = "polinom"
    bare = 0
    for word in words:
        if "=" in word:
            name, _, value = word.partition("=")
            if name == "setka":
                a, b = value.lower().split("x")
                grid = (int(a), int(b))
            elif name == "mu":
                mu = float(value)
            elif name == "proba":
                probe = float(value)
            elif name == "iter":
                iterations = int(value)
            elif name == "celi":
                targets = float(value)
            elif name == "faza":
                if value not in ("svoya", "luchshaya"):
                    sys.exit(f"faza={value}: есть svoya и luchshaya")
                shared = value == "luchshaya"
            elif name == "start":
                if value not in ("polinom", "nol"):
                    sys.exit(f"start={value}: есть polinom и nol")
                start = value
            else:
                sys.exit(f"неизвестный аргумент {name!r}; есть setka, mu, "
                         "proba, iter, faza, celi, start")
        elif re.fullmatch(r"\d+[xX]\d+", word):
            a, b = word.lower().split("x")
            grid = (int(a), int(b))
        else:
            value = float(word)
            if bare == 0:
                mu = value
            elif bare == 1:
                probe = value
            else:
                sys.exit(f"лишнее число {word!r}: голыми идут только mu и проба")
            bare += 1
    return grid, mu, probe, iterations, shared, targets, start


def read_slice(directory: pathlib.Path) -> tuple[np.ndarray, dict]:
    """Прочитать срез и паспорт, склеив куски.

    Принимает каталог, сделанный prepare_slice_rda.py; возвращает (h, meta).

    Большой срез не влезает в один файл — ни в 30 МБ вложения, ни в 100 МБ
    гита, — и производитель режет его ПО ДАЛЬНОСТИ на h_srez_00.npy,
    h_srez_01.npy, … Имена, границы и sha256 лежат в meta.json; здесь куски
    склеиваются обратно по axis=1.

    Сверяется три вещи, и каждая ловит свою беду:

      sha256 куска    файл дошёл целым, а не обрезанным на полпути;
      границы стробов куски идут в том порядке и без пропуска;
      forma из меты   склеенное совпало с тем, что записал производитель.

    Молчаливая ошибка тут дороже любой другой: неверно склеенный срез
    считается без единой жалобы и даёт неверные числа во всех этапах.

    Старый паспорт, где `fail` — одна строка, читается как раньше: там
    ни кусков, ни sha256 нет, и сверять нечего.
    """
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    data = meta.get("dannye", {})
    parts = data.get("chasti")
    if not parts:
        names = data.get("fail") or "h_srez.npy"
        names = [names] if isinstance(names, str) else list(names)
        parts = [{"fail": n} for n in names]

    pieces, expect = [], 0
    for part in parts:
        full = directory / part["fail"]
        if not full.is_file():
            sys.exit(f"нет куска {part['fail']}; в meta.json их "
                     f"{len(parts)}, передавать надо все")
        if part.get("sha256"):
            digest = hashlib.sha256()
            with open(full, "rb") as fp:
                for chunk in iter(lambda: fp.read(2**20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != part["sha256"]:
                sys.exit(f"{part['fail']}: sha256 не сошлась — файл дошёл "
                         "повреждённым или обрезанным, считать по нему нельзя")
        if part.get("stolbcy") and part["stolbcy"][0] != expect:
            sys.exit(f"{part['fail']}: начинается со строба "
                     f"{part['stolbcy'][0]}, а предыдущий кончился на "
                     f"{expect} — куски не в том порядке или один пропущен")
        piece = np.load(full)
        pieces.append(piece)
        expect += piece.shape[1]

    h = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
    forma = (meta.get("srez") or {}).get("forma")
    if forma and list(h.shape) != list(forma):
        sys.exit(f"склеилось {list(h.shape)}, а в паспорте {list(forma)}")
    if len(pieces) > 1:
        print(f"срез склеен из {len(pieces)} кусков, sha256 сошлась у всех")
    return h, meta


def geometry_from_meta(meta: dict) -> Geometry:
    """Geometry из паспорта среза.

    Принимает meta.json как словарь; возвращает Geometry.

    Два поля берутся НЕ из блока geometry, а из отдельных чисел паспорта:

      r_a       разрешение по азимуту, из r_a_razreshenie — оно идёт в
                критерий (5-20);
      r_a_step  шаг изображения по азимуту, из r_a_shag = v/PRF — он идёт в
                перевод (5-22).

    В паспорте они разные, и путать их нельзя: (5-20) про ширину отклика,
    (5-22) про число отсчётов. См. Geometry.azimuth_step.
    """
    g = dict(meta["geometry"])
    g.pop("lambda_", None)  # алгоритм её не читает, длина волны входит как c/f_0
    g["r_a"] = float(meta.get("r_a_razreshenie", g["r_a"]))
    g["r_a_step"] = float(meta.get("r_a_shag", g["r_a"]))
    return Geometry(**{k: v for k, v in g.items() if k in Geometry.__dataclass_fields__},
                    lambda_=float(meta["geometry"]["lambda_"]))


def best_block(rows: list[dict]) -> dict | None:
    """Блок, у которого энтропия упала сильнее всех.

    Принимает список строк по блокам; возвращает строку-победителя или None.

    Падение энтропии — прямая мера того, СКОЛЬКО сведений о фазе было у
    блока: критерий (5-6) ровно её и минимизирует, и насколько он смог её
    опустить, настолько ему и было за что зацепиться.

    Брать блок с наименьшей ИТОГОВОЙ энтропией нельзя, хотя соблазн есть.
    Низкая энтропия бывает и у блока, наполовину занятого водой: энергия
    распределена неравномерно, а сведений о фазе там ноль. Замер на записи
    владельца: у блока с постройками падение 0,1262, у остальных
    двадцати четырёх не больше 0,0069 — разница в двадцать раз, и спутать
    её не с чем.
    """
    if not rows:
        return None
    return max(rows, key=lambda r: r["entropy_before"] - r["entropy_after"])


def focus(h: np.ndarray, geom: Geometry, grid: tuple[int, int] | None,
          mu: float = MU_REAL, max_iterations: int = C.MAX_ITERATIONS,
          shared_phase: bool = False,
          target_fraction: float | None = None,
          start: str = "polinom") -> dict:
    """Собственно прогон: блоки, этап B без ИНС, этап C, этап D.

    Принимает h, Geometry, сетку блоков (или None — тогда по (5-20)), порог
    останова, предел итераций, признак общей фазы, долю целей и откуда
    стартовать ('polinom' или 'nol'); возвращает словарь с картинками и
    числами по блокам.

    Книжный этап B (из параметров движения) не делается: ИНС нет. Замер на
    стенде показал, что книжная (5-30) без них не помогает, а мешает: член
    -(q-q_k)/2 из (5-29) уводит eta в диапазон около [-6, +0,6]. См.
    README, «Если ИНС нет».

    ВМЕСТО НЕГО — ЭТАП B БЕЗ ИНС (start='polinom', по умолчанию):
    stage_b_search ищет phi^(0) как полином по энтропии всей сцены, и
    каждый блок получает его на свою доплеровскую сетку через
    truth_on_block_grid. Почему из нуля нельзя — замер в шапке
    stage_b_search: на настоящей записи шаг Ньютона из нуля составляет
    0,2…1 % от истины, и даже заведомо внесённые 8 рад этап C из нуля не
    снимает (снято 6 %). Полином по всей сцене ту же ошибку находит.

    ОБЩАЯ ФАЗА (shared_phase) — не из книги, и вот зачем она. Сведения о
    фазе несут только яркие точечные отражатели: замер показал, что крупные
    однородные области (вода, поле, лес) опускают энтропию, но градиента по
    фазе почти не дают — при 1,5 рад прирост +0,0059 против +0,0204 у той
    же сцены с целями. На записи владельца цели собрались в одном блоке из
    двадцати пяти, и остальные двадцать четыре возвращали ноль — правильно
    возвращали, у них этих сведений нет.

    Но носитель-то один и полёт один: ошибка движения у блоков общая, если
    сцена не слишком велика. Поэтому здесь фаза лучшего блока переносится
    на все остальные. Это ПРОТИВОРЕЧИТ духу (5-20), где блоки заведены как
    раз ради пространственной изменчивости, и потому включается явно, а не
    по умолчанию: проверять, стало ли лучше, надо линейкой (bench/smear.py),
    а не верой.

    Перенос идёт через truth_on_block_grid: у блоков разной длины
    доплеровская сетка разная, и резать чужую фазу отрезком нельзя.
    """
    backend = get_backend("auto")
    M, N = h.shape
    coefficients = A.linearisation_coefficients(geom)
    x_p, y_p = A.block_half_sizes(coefficients, geom)
    m_p, n_p = A.block_sample_sizes(x_p, y_p, geom)
    if grid is None:
        M_k, N_k, _ = A.block_counts(M, N, m_p, n_p)
    else:
        M_k, N_k = grid
    blocks = A.block_grid(M, N, M_k, N_k, geom)

    h_device = backend.asarray(h)
    scene = A.scene_image(backend, h_device)

    # этап B без ИНС: одна начальная фаза на сцену, с зависимостью от
    # дальности (stage_b_search.RANGE_POWER_OFFSET). Она ПРИМЕНЯЕТСЯ К
    # ДАННЫМ до нарезки, и этап C идёт по книге из нуля на исправленной
    # сцене. Это то же самое, что стартовать этап C из phi^(0), пока phi^(0)
    # зависит только от бина; с дальностью иначе нельзя — у этапа C фаза
    # одна на все стробы блока. Итог для блока: phi^(0)(k, n) + phi_C(k).
    initial, segments = None, None
    scene_corrected = scene
    if start == "polinom":
        range_scale = BS.range_scale_axis(geom.R_B0, geom.r_b, N)
        if M >= 2 * BS.SEGMENT_ROWS:
            # длинный срез: ошибка меняется вдоль азимута (край кадра
            # владельца: 120 -> 930 рад за 4000 строк), одна фаза на сцену
            # не годится — поправка по сегментам, SEGMENT_PASSES проходов
            scene_corrected, segments = BS.correct_segments(
                backend, h_device, range_scale)
        else:
            initial = BS.search(backend, h_device, range_scale=range_scale)
            h_corrected = h_device * backend.xp.exp(1j * backend.asarray(initial.phi))
            scene_corrected = A.scene_image(backend, h_corrected)

    # первый проход: каждый блок ищет свою фазу
    rows = []
    for block in blocks:
        data = A.block_data(backend, scene_corrected, block)
        L = data.h.shape[0]
        result = C.iterate_block(backend, data.h, np.zeros(L), mu=mu,
                                 max_iterations=max_iterations,
                                 core=(data.core_start, data.core_stop),
                                 target_fraction=target_fraction)
        rows.append({
            "q_k": block.q_k,
            "shape": block.shape,
            "iterations": result.n_iterations,
            "converged": result.converged,
            "entropy_before": result.normalised_entropy_history[0],
            "entropy_after": result.normalised_entropy_final,
            "best_iteration": result.best_iteration,
            "frozen_bins": int(sum(result.frozen_bins_history)),
            "live_bins": result.n_live_bins,
            "block_bins": L,
            "phi": result.phi,
        })

    donor = best_block(rows) if shared_phase else None

    # второй проход: картинки. Блок режется заново — это срез плюс одно
    # ОПФ, против сорока итераций стоимость незаметна, зато не приходится
    # держать в памяти двадцать пять блоков ради возможного переноса фазы
    after, before = {}, {}
    for block, row in zip(blocks, rows):
        data = A.block_data(backend, scene_corrected, block)
        L = data.h.shape[0]
        phi = (truth_on_block_grid(donor["phi"], L) if donor is not None
               else row["phi"])
        row["phi_used"] = phi
        phi = D.remove_image_shift(backend, phi)
        shown = D.azimuth_window(backend, data.h)
        after[block.q_k] = D.block_image(backend, shown, phi)[
            data.core_start : data.core_stop]
        # «до» — с ИСХОДНОЙ сцены, без поправки этапа B
        raw = A.block_data(backend, scene, block)
        before[block.q_k] = D.block_image(
            backend, D.azimuth_window(backend, raw.h), np.zeros(L))[
            raw.core_start : raw.core_stop]

    return {
        "blocks": blocks, "M_k": M_k, "N_k": N_k, "per_block": rows,
        "m_p": m_p, "n_p": n_p, "donor": donor, "initial": initial,
        "segments": segments,
        "image_before": backend.to_numpy(D.assemble(backend, before, blocks, M, N)),
        "image_after": backend.to_numpy(D.assemble(backend, after, blocks, M, N)),
        "backend": backend,
    }


def band_occupancy(backend, h, geom: Geometry,
                   gamma: float | None = None) -> list[str]:
    """Сколько доплеровской оси занято сигналом — по данным и по геометрии.

    Принимает бэкенд, срез h и Geometry; возвращает строки для печати.

    Зачем это первым делом. Книга молча считает ось занятой целиком. На
    записи занятая доля равна отношению ШАГА картинки к РАЗРЕШЕНИЮ: полоса
    доплера B_a = v/r_a, частота повторения PRF = v/шаг, их отношение и есть
    доля. Если она мала, большая часть бинов — чистый шум, и фаза в них не
    определена (решение реализации №7). Число, посчитанное по данным, и
    число, обещанное геометрией, обязаны сойтись; расхождение значит, что
    паспорт описывает не эти данные.
    """
    h_dev = backend.asarray(h)
    live = C.signal_bins(backend, backend.xp.abs(h_dev) ** 2)
    measured = float(backend.sum_real(live.astype(backend.accum_real))) / h.shape[0]
    promised = geom.azimuth_step / geom.r_a
    out = [
        f"полоса доплера занята: по данным {100 * measured:.1f} %, "
        f"по геометрии {100 * promised:.1f} % (шаг/разрешение)",
        "  пустые бины несут только шум; фаза в них не оценивается "
        "(решение №7)" if measured < 0.5 else
        "  ось занята целиком — решение №7 не вмешивается",
    ]
    # Занятость, разрешение и весовое окно связаны тождеством
    #
    #     занятость = B_a / PRF,   B_a = gamma * v / r_a,   PRF = v / шаг
    #     => занятость = gamma * шаг / r_a
    #
    # Три величины, две известны из паспорта — значит третья ВЫВОДИТСЯ. Самая
    # интересная здесь gamma: её в паспорте продукта обычно нет вовсе, а она
    # входит и в длину апертуры, и в размер блока (5-20) через x_p ~ 1/gamma.
    #
    # Выводить вместо неё разрешение — ОШИБКА, и она здесь была: без
    # множителя gamma выходило «r_a = 0,258 м против паспортных 0,400», то
    # есть мнимое расхождение в 1,55 раза, и совет взять вдвое больше блоков.
    # На деле 1,55 — это и есть gamma, и она в обычном диапазоне окон.
    if 0.0 < measured < 1.0:
        implied = measured * geom.r_a / geom.azimuth_step
        out.append(
            f"  отсюда расширение весового окна gamma = {implied:.2f} "
            "(прямоугольное 1,00 · Хэмминг 1,30 · Хэннинг 1,44 · Блэкман 1,68)")
        if implied < 1.0:
            out.append(
                "  ВНИМАНИЕ: gamma меньше единицы невозможна — окно не может "
                "сузить лепесток уже прямоугольного. Значит паспортное "
                f"r_a = {geom.r_a:.3f} м мельче, чем позволяет эта полоса: "
                f"по ней r_a не меньше {geom.azimuth_step / measured:.3f} м")
        elif implied > BAND_GAMMA_MAX:
            out.append(
                f"  ВНИМАНИЕ: gamma = {implied:.2f} велика даже для тяжёлых "
                "окон. Либо r_a в паспорте занижено, либо занятость меряется "
                "вместе со скатами полосы")
        if gamma:
            factor = implied / gamma
            if abs(factor - 1.0) < BAND_GAMMA_TOLERANCE:
                out.append(f"  срез считан при gamma = {gamma:.2f} — сходится, "
                           "сетку менять незачем")
            else:
                out.append(
                    f"  срез считан при gamma = {gamma:.2f}; размер блока "
                    f"(5-20) идёт как 1/gamma, значит блоков по азимуту стоит "
                    f"взять в {max(factor, 1 / factor):.2f} раза "
                    f"{'больше' if factor > 1 else 'меньше'}")
    return out


def grid_agreement(meta: dict, m_p: float, n_p: float,
                   N_k: int | None) -> list[str]:
    """Сверка нарезки с тем, чего ждёт паспорт.

    Принимает meta.json, размеры блока в отсчётах и число блоков по дальности
    (None, если сетку задали рукой — тогда сверять её не с чем); возвращает
    строки для печати, пустой список — если в паспорте ожидания нет.

    Зачем. prepare_slice.py кладёт в паспорт раздел ozhidaemaya_narezka: каким
    размер блока по азимуту ДОЛЖЕН выйти, и каким он выходил, пока код путал
    разрешение с шагом изображения. Раздел написан ловушкой ровно на эту
    путаницу. На записи истины нет, проверить фазу нечем — а нарезку эта
    сверка проверяет, и она единственная.
    """
    expected = meta.get("ozhidaemaya_narezka")
    if not expected:
        return []
    lines = []
    want_m = expected.get("m_p_pravilnyy")
    if want_m is not None:
        agrees = abs(m_p - want_m) <= 0.01 * max(abs(want_m), 1.0)
        lines.append(f"  отсчётов в блоке по азимуту: код {m_p:.2f}, "
                     f"паспорт ждёт {want_m:.2f} — "
                     f"{'сходится' if agrees else 'НЕ СХОДИТСЯ'}")
        wrong = expected.get("m_p_kak_schitaet_kod")
        if wrong is not None and not agrees:
            lines.append(f"  (если вышло около {wrong:.2f} — разрешение и шаг "
                         f"опять слиплись в одно число, см. Geometry.r_a_step)")
    want_n = expected.get("N_k_ozhidaemoe")
    if want_n is not None and N_k is not None:
        lines.append(f"  блоков по дальности: код {N_k}, паспорт ждёт {want_n} — "
                     f"{'сходится' if N_k == want_n else 'НЕ СХОДИТСЯ'} "
                     f"(при n_p = {n_p:.0f} отсчётов на блок)")
    return lines


def display_window(amplitude: np.ndarray, low: float = DISPLAY_LOW,
                   high: float = DISPLAY_HIGH) -> tuple[float, float]:
    """Границы показа картинки в дБ — ПО ПРОЦЕНТИЛЯМ, а не по максимуму.

    Повторяет prepare_slice_rda.display_window намеренно: тот скрипт
    держится самостоятельным, его уносят на машину с данными одним
    файлом, и импорта отсюда там быть не должно.

    Принимает модуль изображения и две процентили; возвращает (vmin, vmax)
    в дБ для imshow.

    Нормировка по максимуму для радиолокационной сцены не годится, и это не
    вкус, а арифметика. Несколько ярких отражателей сидят на 40-50 дБ выше
    местности, поэтому при делении на максимум медиана картинки оказывается
    около -33 дБ, а окно -40…0 оставляет половину пикселей в самом низу:
    изображение выходит чёрным. Замер на срезе владельца:

        P25 -37,3 дБ   P50 -33,2 дБ   P75 -29,4 дБ   P99 -2,7 дБ

    Окно P5…P97 шириной 32 дБ кладёт медиану на 37 % серой шкалы, оставляя
    5 % чёрных и 3 % белых — так и показывают РСА.
    """
    db = 20.0 * np.log10(np.maximum(amplitude, amplitude.max() * 1e-6))
    return float(np.percentile(db, low)), float(np.percentile(db, high))


def probe_phase(backend, h: np.ndarray, edge_rad: float) -> tuple[np.ndarray, float]:
    """Известная квадратичная ошибка для пробы; живёт ТОЛЬКО в занятой полосе.

    Принимает бэкенд, срез и размах в радианах на краю полосы; возвращает
    (вектор длины M, полуширину полосы в нормированной частоте).

    Размах задаётся по краю ЗАНЯТОЙ полосы, а не по краю доплеровской оси.
    На передискретизованной записи это разные вещи: полоса занимает пятую
    часть оси, и квадратичная ошибка, нормированная на край оси, дала бы
    внутри полосы в двадцать пять раз меньше — вносить было бы нечего.

    За полосой проба РАВНА НУЛЮ, и это не косметика. Первая версия
    продолжала параболу по всей оси: при размахе 3 рад по краю полосы на
    краю оси выходило 185 рад. Сигнала там нет, картинке всё равно, — но
    мера остатка заворачивает такую разность в (-pi, pi] и выдаёт шум
    величиной pi/sqrt(3) = 1,81 рад. Замер тогда показывал «снято 1 %» даже
    там, где алгоритм заведомо работает.

    Квадратичная потому, что это и есть расфокусировка — то, что описывает
    этап B (5-27) и что остаётся от неучтённого движения в первом порядке.
    """
    live = backend.to_numpy(C.signal_bins(backend, backend.xp.abs(
        backend.asarray(h)) ** 2)).astype(bool)
    u = np.fft.fftfreq(h.shape[0]) * 2.0          # нормированная частота, край = +-1
    u_band = max(float(np.abs(u[live]).max()), 1e-6)
    inside = np.abs(u) <= u_band
    return edge_rad * (u / u_band) ** 2 * inside, u_band


def contrast(image: np.ndarray) -> float:
    """Контраст картинки: sqrt(<I^2>)/<I> по интенсивности. Больше — резче.

    В отличие от энтропии, алгоритм его не минимизирует, поэтому смотреть
    надо именно на него. К циклическому сдвигу не чувствителен.
    """
    intensity = np.abs(np.asarray(image)) ** 2
    return float(np.sqrt((intensity**2).mean()) / intensity.mean())


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    directory = pathlib.Path(sys.argv[1])
    grid, mu, probe, iterations, shared, targets, start = parse_options(sys.argv[2:])

    h, meta = read_slice(directory)
    geom = geometry_from_meta(meta)
    print(f"срез {h.shape} {h.dtype}, R_B0 = {geom.R_B0:.1f} м, T_a = {geom.T_a:.4f} с")
    print(f"разрешение по азимуту {geom.r_a:.4f} м, шаг {geom.azimuth_step:.4f} м "
          f"(шире в {geom.r_a / geom.azimuth_step:.2f} раза)")

    notes = inputs.check_inputs(h, geom)
    if notes:
        print("\nзамечания по входу:")
        for note in notes:
            print("  -", note)
        print("\nэто не отказ: считаем дальше, но числа читайте с поправкой")

    print("\nчто даёт геометрия:")
    for name, value in inputs.derived_numbers(geom, *h.shape).items():
        print(f"  {name:24} {value:.3f}")

    print()
    for line in band_occupancy(get_backend("auto"), h, geom,
                               meta.get("gamma_win")):
        print(line)

    run = focus(h, geom, grid, mu=mu, max_iterations=iterations,
                shared_phase=shared, target_fraction=targets, start=start)
    print(f"\nсетка блоков {run['M_k']}x{run['N_k']}, порог (5-9) mu = {mu:g}, "
          f"предел {iterations} итераций"
          + (f", критерий по целям ({100*targets:g} % стробов, решение №9)"
             if targets else ""))
    if run["initial"] is not None:
        ini = run["initial"]
        print(f"этап B без ИНС: полином по энтропии сцены, "
              f"S {ini.entropy_start:.4f} -> {ini.entropy_final:.4f} "
              f"({ini.entropy_final - ini.entropy_start:+.4f}), "
              f"{ini.n_evaluations} оценок; коэффициенты при Лежандре "
              + ", ".join(f"P{d} = {c:+.2f}" for d, c in ini.coefficients.items())
              + f" рад (при R = R_B0); размах phi^(0) в полосе "
              f"{ini.phi.max() - ini.phi.min():.1f} рад")
        # квадратичная на краю полосы -> ошибка скорости продукта. Полная
        # фаза апертуры Phi = pi K_a (T_a/2)^2, K_a = 2 v^2 / (lambda R);
        # остаток a = Phi * dK/K = 2 Phi * dv/v, значит dv = a v / (2 Phi).
        # Край полосы взят за край апертуры — на этой записи 93 против
        # 72 Гц, оценка грубая, в полтора раза
        a_quadratic = 1.5 * ini.coefficients.get(2, 0.0)   # P2 = 1,5 u^2 - 0,5
        phi_aperture = (math.pi * geom.v_x0 ** 2 * geom.T_a ** 2
                        / (2.0 * geom.lambda_ * geom.R_B0))
        dv = a_quadratic * geom.v_x0 / (2.0 * phi_aperture)
        print(f"  квадратичная {a_quadratic:+.1f} рад на краю полосы при полной фазе "
              f"апертуры {phi_aperture:.0f} рад — это ошибка скорости продукта "
              f"около {dv:+.2f} м/с (грубо, в полтора раза)")
    elif run["segments"] is not None:
        phi_ap = (math.pi * geom.v_x0 ** 2 * geom.T_a ** 2
                  / (2.0 * geom.lambda_ * geom.R_B0))
        print(f"этап B без ИНС ПО СЕГМЕНТАМ, проходов {len(run['segments'])}: "
              f"своя поправка на каждый сегмент азимута, полная фаза апертуры "
              f"{phi_ap:.0f} рад. Коэффициенты — при u^2 и u^3 на краю полосы, R = R_B0")
        for k, pass_ in enumerate(run["segments"]):
            print(f"  проход {k + 1}: окна {pass_['rows']} строк, шаг {pass_['hop']}"
                  + (", только квадратичная" if k == 0 else ", квадратичная и кубическая"))
            print(f"{'строки':>13} {'полоса -10дБ':>13} {'квадр., рад':>12} "
                  f"{'куб., рад':>10} {'dS':>8} {'  что это'}")
            widest = max(w for w, _ in pass_["bands"]) if pass_["bands"] else 1.0
            for seg, (width, centre) in zip(pass_["segments"], pass_["bands"]):
                quad = 1.5 * seg.coefficients.get(2, 0.0)
                cube = 2.5 * seg.coefficients.get(3, 0.0)
                dS = seg.entropy_final - seg.entropy_start
                note = ""
                if width < BS.SEGMENT_BAND_MIN_FRACTION * widest:
                    note = "полоса урезана — апертура неполная, край кадра"
                elif abs(quad) > 0.25 * phi_ap:
                    note = (f"ошибка {abs(quad)/phi_ap*100:.0f} % фазы апертуры — "
                            "не движение, опорная функция продукта")
                print(f"{seg.row_start:5d}…{seg.row_stop:<7d} {width*100:12.1f} % "
                      f"{quad:+12.0f} {cube:+10.0f} {dS:+8.4f}  {note}")
    else:
        print("начальная фаза нулевая (start=nol), этап B не делается")
    if run["donor"] is not None:
        d = run["donor"]
        drops = sorted(r["entropy_before"] - r["entropy_after"]
                       for r in run["per_block"])
        print(f"ОБЩАЯ ФАЗА: взята у блока {d['q_k']}, у него энтропия упала на "
              f"{d['entropy_before'] - d['entropy_after']:.4f}; "
              f"у следующего за ним — на {drops[-2] if len(drops) > 1 else 0:.4f}")
        print("Это НЕ по книге: (5-20) заводит блоки как раз ради "
              "пространственной изменчивости. Смотрите линейку — стало ли "
              "лучше там, где своей фазы не было.")
    for line in grid_agreement(meta, run["m_p"], run["n_p"],
                               None if grid else run["N_k"]):
        print(line)
    print(f"\n{'блок':>5}{'размер':>10}{'занято':>14}{'итер':>6}{'сошлось':>9}"
          f"{'S до':>9}{'S после':>9}{'заморожено':>12}")
    for row in run["per_block"]:
        size = f"{row['shape'][0]}x{row['shape'][1]}"
        live = f"{row['live_bins']}/{row['block_bins']}"
        print(f"{row['q_k']:>5}{size:>10}{live:>14}{row['iterations']:>6}"
              f"{str(row['converged']):>9}{row['entropy_before']:>9.4f}"
              f"{row['entropy_after']:>9.4f}{row['frozen_bins']:>12}")

    before, after = run["image_before"], run["image_after"]
    print(f"\n{'':22}{'контраст':>10}{'энтропия (5-6)':>17}{'над спеклом':>14}")
    backend = run["backend"]
    for name, image in (("до автофокуса", before), ("после", after)):
        m = image_sharpness(backend, image)
        print(f"{name:22}{m['contrast']:>10.2f}{m['S']:>17.4f}"
              f"{m['contrast'] / SPECKLE_CONTRAST:>13.1f}x")
    if image_sharpness(backend, before)["contrast"] < SPECKLE_FLOOR_WARN:
        print(f"\nВНИМАНИЕ: контраст ДО автофокуса ниже {SPECKLE_FLOOR_WARN:g} — "
              f"это меньше чем вдвое над уровнем однородного спекла "
              f"({SPECKLE_CONTRAST:.2f}). На таком срезе фокусировать нечего: "
              "выраженных отражателей в нём нет, и энтропийный критерий не за "
              "что зацепить. Смотрите srez.png — скорее всего срез встал на "
              "пустой участок записи.")
    if probe:
        # Замер РАЗНОСТНЫЙ, и иначе нельзя. В данных уже сидит своя фаза, и
        # после прогона с пробой найдено будет «своё плюс проба». Сравнивать
        # найденное с одной пробой значит записывать своё в ошибку: на
        # подделке, где алгоритм заведомо работает, так выходило «снято 1 %».
        # Поэтому прогон делается дважды, и с пробой сверяется РАЗНОСТЬ оценок.
        backend = get_backend("auto")
        phi_probe, u_band = probe_phase(backend, h, probe)
        print(f"\nПРОБА: внесена квадратичная ошибка размахом {probe:g} рад по "
              f"краю занятой полосы, СКО {np.std(phi_probe):.3f} рад. "
              "Второй прогон, сверяется разность оценок.")
        spoiled = h * np.exp(-1j * phi_probe)[:, None].astype(h.dtype)
        run_probe = focus(spoiled, geom, grid, mu=mu,
                          max_iterations=iterations, shared_phase=shared,
                          target_fraction=targets, start=start)

        print(f"\n{'блок':>5}{'внесено, рад':>14}{'осталось, рад':>15}"
              f"{'снято':>8}")
        removed = []
        for clean, dirty in zip(run["per_block"], run_probe["per_block"]):
            L = clean["block_bins"]
            truth = truth_on_block_grid(phi_probe, L)
            # остаток — только по занятым бинам блока: в пустом фазы нет ни у
            # пробы, ни у оценки, и разность там — завёрнутый шум
            occupied = np.abs(np.fft.fftfreq(L) * 2.0) <= u_band
            was = phase_residual_rms(truth, np.zeros(L), occupied)
            now = phase_residual_rms(truth, dirty["phi"] - clean["phi"], occupied)
            removed.append(1.0 - now / max(was, 1e-12))
            print(f"{clean['q_k']:>5}{was:>14.3f}{now:>15.3f}"
                  f"{100 * removed[-1]:>7.0f}%")
        share = float(np.median(removed))
        print(f"\nСнято {100 * share:.0f} % внесённой ошибки (медиана по блокам).")
        if share > PROBE_FOUND_SHARE:
            print("Значит алгоритм на ЭТИХ данных работает, и если без пробы "
                  "он почти ничего не менял — остаточной фазы в записи "
                  "действительно мало. Штатный тракт её уже снял.")
        else:
            print("Значит известную ошибку такого размаха алгоритм на этой "
                  "сцене НЕ находит. Две причины, и различить их помогает "
                  "проба поменьше:")
            print("  * размах великоват — из нуля алгоритм уходит в другой "
                  "минимум. На подделке 0,8 и 1,5 рад снимаются на 85 %, "
                  "а 3 рад уже нет вовсе. Попробуйте 0,5 и 1;")
            print("  * сцене нечего дать критерию — тогда и малая проба не "
                  "снимется. Смотрите, насколько картинка над спеклом.")

    # ------------------------------------------------ линейка размаза
    cells = geom.r_a / geom.azimuth_step
    rows = smear.compare(before, after, cells, count=SMEAR_TARGETS)
    if rows:
        print(f"\nРАЗМАЗ по {len(rows)} самым ярким одиночным целям. Ширина в "
              f"элементах разрешения (элемент = {cells:.1f} отсчёта);")
        print("у сфокусированной цели -3 дБ = 0,886, ISLR около -9,7 дБ.")
        print(f"\n{'цель m,n':>14}{'-3 дБ':>16}{'-10 дБ':>16}{'-20 дБ':>16}"
              f"{'ISLR, дБ':>16}{'дальность':>18}")
        print(f"{'':14}{'до':>8}{'после':>8}{'до':>8}{'после':>8}"
              f"{'до':>8}{'после':>8}{'до':>8}{'после':>8}"
              f"{'-3дБ':>9}{'-20дБ':>9}")
        for row in rows:
            b, a = row["before"], row["after"]
            print(f"{row['m']:>7},{row['n']:<6}"
                  f"{b['width_3db']:>8.2f}{a['width_3db']:>8.2f}"
                  f"{b['width_10db']:>8.2f}{a['width_10db']:>8.2f}"
                  f"{b['width_20db']:>8.2f}{a['width_20db']:>8.2f}"
                  f"{b['islr_db']:>8.2f}{a['islr_db']:>8.2f}"
                  f"{b['range_3db']:>9.2f}{b['range_20db']:>9.2f}")
        med = lambda key, side: float(np.median(
            [r[side][key] for r in rows]))
        print(f"\n{'медиана':>14}"
              f"{med('width_3db','before'):>8.2f}{med('width_3db','after'):>8.2f}"
              f"{med('width_10db','before'):>8.2f}{med('width_10db','after'):>8.2f}"
              f"{med('width_20db','before'):>8.2f}{med('width_20db','after'):>8.2f}"
              f"{med('islr_db','before'):>8.2f}{med('islr_db','after'):>8.2f}")
        print("\nДальность в ОТСЧЁТАХ и только «до»: автофокус правит фазу "
              "лишь по азимуту, по дальности он ничего не трогает. Цель, "
              "широкая и там, — не расфокусировка, а протяжённый объект или "
              "остаточная миграция, и азимутальным автофокусом не лечится.")
        print("\nШирину по -3 дБ при сильной расфокусировке читать нельзя: "
              "главный лепесток разваливается, и уровень пересекается не там. "
              "Смотрите -20 дБ и ISLR — это и есть хвосты, которые видно глазом.")

    print("\nЭнтропия обязана упасть — её алгоритм и минимизирует, это НЕ проверка.")
    print("Независимо смотрит контраст и глаза: картинки рядом со срезом.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        # окно показа считается по картинке ДО и применяется к обеим: разные
        # шкалы сделали бы сравнение ложным
        vmin, vmax = display_window(np.abs(before))
        floor = float(np.abs(before).max()) * 1e-6
        for ax, image, title in ((axes[0], before, "до"), (axes[1], after, "после")):
            db = 20 * np.log10(np.maximum(np.abs(image), floor))
            ax.imshow(db, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}, контраст {contrast(image):.2f}")
            ax.set_xlabel("строб дальности n")
            ax.set_ylabel("азимут m")
        fig.suptitle(f"MN-MEA на реальной записи, сетка {run['M_k']}x{run['N_k']}"
                     f"; показ {vmin:.0f} … {vmax:.0f} дБ, окно "
                     f"P{DISPLAY_LOW:g}-P{DISPLAY_HIGH:g}, общее для обеих")
        fig.tight_layout()
        out = directory / "mn_mea_do_posle.png"
        fig.savefig(out, dpi=120)
        print(f"картинка: {out}")

        # срезы по азимуту через самые яркие цели: то же, что меряет
        # линейка, но глазами. Уровень в дБ от собственного пика каждой
        # цели, ось — в элементах разрешения от пика
        shown = rows[:SMEAR_CUTS]
        if shown:
            fig, axes = plt.subplots(1, len(shown), figsize=(4 * len(shown), 4),
                                     squeeze=False, sharey=True)
            for ax, row in zip(axes[0], shown):
                for side, colour, name in (("before", "0.55", "до"),
                                           ("after", "C0", "после")):
                    prof = row[side]["profile"]
                    peak = int(np.argmax(prof))
                    axis = (np.arange(prof.size) - peak) / (
                        cells * smear.OVERSAMPLE)
                    ax.plot(axis, 20 * np.log10(np.maximum(
                        prof / prof.max(), 1e-4)), colour, label=name, lw=1.1)
                ax.set_xlim(-8, 8)
                ax.set_ylim(-45, 2)
                ax.grid(alpha=0.25)
                ax.set_title(f"цель {row['m']},{row['n']}\n"
                             f"ISLR {row['before']['islr_db']:.1f} -> "
                             f"{row['after']['islr_db']:.1f} дБ", fontsize=10)
                ax.set_xlabel("элементы разрешения от пика")
            axes[0][0].set_ylabel("дБ от пика")
            axes[0][0].legend(loc="upper right", fontsize=9)
            fig.suptitle("Отклик по азимуту: чем ниже хвосты, тем меньше размаз")
            fig.tight_layout()
            cuts = directory / "mn_mea_srezy.png"
            fig.savefig(cuts, dpi=120)
            print(f"срезы:    {cuts}")

        # крупный план 1:1 вокруг тех же целей: общий план кадра сужения
        # пятна показать не может (см. KRUPNO_CELLS), а здесь оно видно
        if shown:
            half_m = int(round(KRUPNO_CELLS * cells))
            half_n = max(int(round(KRUPNO_CELLS * geom.r_a / geom.r_b)), 4)
            # imshow: aspect = высота строки / ширина столбца на экране; строка
            # 0,057 м, столбец 0,30 м -> 0,19, тогда метр по азимуту равен
            # метру по дальности (первая версия ставила обратное, панели
            # выходили лентами в 5 раз выше, чем надо)
            aspect = geom.azimuth_step / geom.r_b
            fig, axes = plt.subplots(2, len(shown), figsize=(3.6 * len(shown), 7.6),
                                     squeeze=False)
            for j, row in enumerate(shown):
                m0, n0 = row["m"], row["n"]
                m_lo, m_hi = max(m0 - half_m, 0), min(m0 + half_m, before.shape[0])
                n_lo, n_hi = max(n0 - half_n, 0), min(n0 + half_n, before.shape[1])
                # уровень в дБ от пика цели ДО, общий для обеих картинок
                peak = float(np.abs(before[m0, n0]))
                for i, (image, name) in enumerate(((before, "до"), (after, "после"))):
                    patch = np.abs(image[m_lo:m_hi, n_lo:n_hi])
                    db = 20 * np.log10(np.maximum(patch / max(peak, 1e-300), 1e-6))
                    ax = axes[i][j]
                    ax.imshow(db, cmap="gray", vmin=-35, vmax=0, aspect=aspect,
                              extent=[n_lo, n_hi, m_hi, m_lo])
                    ax.set_title(f"{name} {m0},{n0}: -20 дБ "
                                 f"{row[name == 'до' and 'before' or 'after']['width_20db']:.1f} эл.",
                                 fontsize=10)
                    ax.set_xlabel("строб"); ax.set_ylabel("азимут")
            fig.suptitle(f"Крупный план 1:1, +-{KRUPNO_CELLS:g} элементов "
                         f"({2*half_m} строк x {2*half_n} стробов), дБ от пика цели до")
            fig.tight_layout()
            krupno = directory / "mn_mea_krupno.png"
            fig.savefig(krupno, dpi=120)
            print(f"крупно:   {krupno}")

        # карта размаза: где по кадру он есть, а где нет. Таблица отвечает
        # «каковы лучшие цели», карта — «где картинка размазана», и это
        # разные вопросы
        maps_before = smear.smear_map(before, cells, geom.r_a, geom.r_b)
        maps_after = smear.smear_map(after, cells, geom.r_a, geom.r_b)
        islr_before, width_before, ratio_before = maps_before
        islr_after, width_after, ratio_after = maps_after
        if np.isfinite(ratio_before).any():
            fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
            hi = float(np.nanpercentile([ratio_before, ratio_after], 98))
            for ax, data, name in ((axes[0], ratio_before, "до"),
                                   (axes[1], ratio_after, "после")):
                im = ax.imshow(data, aspect="auto", cmap="inferno",
                               vmin=1.0, vmax=max(hi, 1.5), origin="upper",
                               extent=[0, after.shape[1], after.shape[0], 0])
                ax.set_title(f"{name}, медиана отношения "
                             f"{np.nanmedian(data):.2f}")
                ax.set_xlabel("строб дальности n")
            axes[0].set_ylabel("азимут m")
            fig.colorbar(im, ax=axes,
                         label="ширина по азимуту / по дальности, в метрах")
            fig.suptitle(
                "Где картинка РАЗМАЗАНА: отношение ширины отклика по азимуту "
                "к ширине по дальности.\nОколо 1 — объект просто такого "
                "размера. Много больше 1 — растянут только по азимуту, это "
                f"размаз. Пусто — цели нет ({np.isnan(ratio_before).sum()} "
                f"плиток из {ratio_before.size})")
            out_map = directory / "mn_mea_karta.png"
            fig.savefig(out_map, dpi=120)
            print(f"карта:    {out_map}")
            print(f"\nПО КАРТЕ, {np.isfinite(ratio_before).sum()} плиток с целями:")
            print(f"  отношение азимут/дальность: медиана "
                  f"{np.nanmedian(ratio_before):.2f} -> "
                  f"{np.nanmedian(ratio_after):.2f}, "
                  f"худшая плитка {np.nanmax(ratio_before):.2f} -> "
                  f"{np.nanmax(ratio_after):.2f}")
            print(f"  плиток с отношением выше 1,5: "
                  f"{int(np.nansum(ratio_before > 1.5))} -> "
                  f"{int(np.nansum(ratio_after > 1.5))}")
            print(f"  ISLR медиана {np.nanmedian(islr_before):.2f} -> "
                  f"{np.nanmedian(islr_after):.2f} дБ, ширина -20 дБ "
                  f"{np.nanmedian(width_before):.2f} -> "
                  f"{np.nanmedian(width_after):.2f} элемента")
            print("  Отношение около 1 значит, что объект широк в обе стороны "
                  "одинаково — он просто такого размера, и автофокус тут ни "
                  "при чём. Размаз — это только то, что вытянуто по азимуту.")
    except ImportError:
        print("matplotlib нет, картинку пропустили")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

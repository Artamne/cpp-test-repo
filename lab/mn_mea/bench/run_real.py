"""Прогон MN-MEA на срезе РЕАЛЬНОЙ записи, подготовленном prepare_slice.py.

    python3 bench/run_real.py путь/к/mn_mea_srez [M_k x N_k] [mu]

Читает h_srez.npy и meta.json, проверяет вход, считает, кладёт рядом
картинки до и после и числа. Сетку блоков можно задать вторым аргументом,
например 8x1; без него берётся (5-20) по геометрии из паспорта. Третьим —
порог останова (5-9), по умолчанию MU_REAL.

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

    read_slice          прочитать h_srez.npy и meta.json
    geometry_from_meta  Geometry из паспорта, разрешение отдельно от шага
    focus               блоки, этап C из нуля, этап D
    grid_agreement      сверка нарезки с ожиданием паспорта
    contrast            число, которое алгоритм НЕ минимизирует
    main                всё вместе, с печатью и картинкой
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

_sys_path = pathlib.Path(__file__).resolve().parent.parent / "algorithm"
sys.path.insert(0, str(_sys_path))

import numpy as np

import inputs
import stage_a_blocks as A
import stage_c_iterate as C
import stage_d_assemble as D
from backend import get_backend
from stage_a_blocks import Geometry
from validate import image_sharpness


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


def read_slice(directory: pathlib.Path) -> tuple[np.ndarray, dict]:
    """Прочитать срез и паспорт.

    Принимает каталог, сделанный prepare_slice.py; возвращает (h, meta).
    """
    h = np.load(directory / "h_srez.npy")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
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


def focus(h: np.ndarray, geom: Geometry, grid: tuple[int, int] | None,
          mu: float = MU_REAL) -> dict:
    """Собственно прогон: блоки, этап C из нуля, этап D.

    Принимает h, Geometry, сетку блоков (или None — тогда по (5-20)) и порог
    останова; возвращает словарь с картинками и числами по блокам.

    Начальная фаза НУЛЕВАЯ, этап B не делается. Замер на стенде показал, что
    книжная (5-30) не помогает, а мешает: член -(q-q_k)/2 из (5-29) уводит
    eta в диапазон около [-6, +0,6]. См. README, «Если ИНС нет».
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
    after, before, rows = {}, {}, []
    for block in blocks:
        data = A.block_data(backend, scene, block)
        L = data.h.shape[0]
        result = C.iterate_block(backend, data.h, np.zeros(L), mu=mu,
                                 core=(data.core_start, data.core_stop))
        phi = D.remove_image_shift(backend, result.phi)
        shown = D.azimuth_window(backend, data.h)
        after[block.q_k] = D.block_image(backend, shown, phi)[
            data.core_start : data.core_stop]
        before[block.q_k] = D.block_image(backend, shown, np.zeros(L))[
            data.core_start : data.core_stop]
        rows.append({
            "q_k": block.q_k,
            "shape": block.shape,
            "iterations": result.n_iterations,
            "converged": result.converged,
            "entropy_before": result.normalised_entropy_history[0],
            "entropy_after": result.normalised_entropy_final,
            "frozen_bins": int(sum(result.frozen_bins_history)),
            "live_bins": result.n_live_bins,
            "block_bins": L,
        })
    return {
        "blocks": blocks, "M_k": M_k, "N_k": N_k, "per_block": rows,
        "m_p": m_p, "n_p": n_p,
        "image_before": backend.to_numpy(D.assemble(backend, before, blocks, M, N)),
        "image_after": backend.to_numpy(D.assemble(backend, after, blocks, M, N)),
        "backend": backend,
    }


def band_occupancy(backend, h, geom: Geometry) -> list[str]:
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
    return [
        f"полоса доплера занята: по данным {100 * measured:.1f} %, "
        f"по геометрии {100 * promised:.1f} % (шаг/разрешение)",
        "  пустые бины несут только шум; фаза в них не оценивается "
        "(решение №7)" if measured < 0.5 else
        "  ось занята целиком — решение №7 не вмешивается",
    ]


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
    grid = None
    if len(sys.argv) > 2:
        a, b = sys.argv[2].lower().split("x")
        grid = (int(a), int(b))
    mu = float(sys.argv[3]) if len(sys.argv) > 3 else MU_REAL

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
    for line in band_occupancy(get_backend("auto"), h, geom):
        print(line)

    run = focus(h, geom, grid, mu=mu)
    print(f"\nсетка блоков {run['M_k']}x{run['N_k']}, порог (5-9) mu = {mu:g}, "
          f"начальная фаза нулевая (этап B не делается)")
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
    print("\nЭнтропия обязана упасть — её алгоритм и минимизирует, это НЕ проверка.")
    print("Независимо смотрит контраст и глаза: картинки рядом с h_srez.npy.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        top = float(np.abs(after).max())
        for ax, image, title in ((axes[0], before, "до"), (axes[1], after, "после")):
            db = 20 * np.log10(np.maximum(np.abs(image), top * 1e-3) / top)
            ax.imshow(db, aspect="auto", cmap="gray", vmin=-45, vmax=0)
            ax.set_title(f"{title}, контраст {contrast(image):.2f}")
            ax.set_xlabel("строб дальности n")
            ax.set_ylabel("азимут m")
        fig.suptitle(f"MN-MEA на реальной записи, сетка {run['M_k']}x{run['N_k']}")
        fig.tight_layout()
        out = directory / "mn_mea_do_posle.png"
        fig.savefig(out, dpi=120)
        print(f"картинка: {out}")
    except ImportError:
        print("matplotlib нет, картинку пропустили")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

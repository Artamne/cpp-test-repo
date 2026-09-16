"""Прогон MN-MEA на срезе РЕАЛЬНОЙ записи, подготовленном prepare_slice.py.

    python3 bench/run_real.py путь/к/mn_mea_srez [M_k x N_k]

Читает h_srez.npy и meta.json, проверяет вход, считает, кладёт рядом
картинки до и после и числа. Сетку блоков можно задать вторым аргументом,
например 8x1; без него берётся (5-20) по геометрии из паспорта.

ЧЕМ ЭТОТ ПРОГОН ОТЛИЧАЕТСЯ ОТ СТЕНДА, и это главное.

На синтетике есть ИСТИНА: внесённую фазу знаем мы сами, и «главное число»
отчёта — расхождение с ней. На записи истины нет ни у кого. Значит сказать
«фаза найдена верно» здесь НЕЛЬЗЯ, и никакое число этого не скажет.

Что можно сказать: картинка стала резче или нет. Мерится двумя числами —
нормированной энтропией (5-6) и контрастом. Энтропия при этом не является
независимой проверкой: алгоритм её и минимизирует, она обязана упасть.
Независимо смотрит контраст, и он же на графиках — глазами.
"""

from __future__ import annotations

import json
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
          mu: float = 1e-3) -> dict:
    """Собственно прогон: блоки, этап C из нуля, этап D.

    Принимает h, Geometry, сетку блоков (или None — тогда по (5-20)) и порог
    останова; возвращает словарь с картинками и числами по блокам.

    Начальная фаза НУЛЕВАЯ, этап B не делается. Замер на стенде показал, что
    книжная (5-30) не помогает, а мешает: член -(q-q_k)/2 из (5-29) уводит
    eta в диапазон около [-6, +0,6]. См. README, «Если ИНС нет».
    """
    backend = get_backend("auto")
    M, N = h.shape
    if grid is None:
        coefficients = A.linearisation_coefficients(geom)
        x_p, y_p = A.block_half_sizes(coefficients, geom)
        m_p, n_p = A.block_sample_sizes(x_p, y_p, geom)
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
        })
    return {
        "blocks": blocks, "M_k": M_k, "N_k": N_k, "per_block": rows,
        "image_before": backend.to_numpy(D.assemble(backend, before, blocks, M, N)),
        "image_after": backend.to_numpy(D.assemble(backend, after, blocks, M, N)),
        "backend": backend,
    }


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

    run = focus(h, geom, grid)
    print(f"\nсетка блоков {run['M_k']}x{run['N_k']}, "
          f"начальная фаза нулевая (этап B не делается)")
    print(f"\n{'блок':>5}{'размер':>10}{'итер':>6}{'сошлось':>9}"
          f"{'S до':>9}{'S после':>9}{'заморожено':>12}")
    for row in run["per_block"]:
        size = f"{row['shape'][0]}x{row['shape'][1]}"
        print(f"{row['q_k']:>5}{size:>10}{row['iterations']:>6}"
              f"{str(row['converged']):>9}{row['entropy_before']:>9.4f}"
              f"{row['entropy_after']:>9.4f}{row['frozen_bins']:>12}")

    before, after = run["image_before"], run["image_after"]
    print(f"\n{'':22}{'контраст':>10}{'энтропия (5-6)':>17}")
    backend = run["backend"]
    for name, image in (("до автофокуса", before), ("после", after)):
        m = image_sharpness(backend, image)
        print(f"{name:22}{m['contrast']:>10.2f}{m['S']:>17.4f}")
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

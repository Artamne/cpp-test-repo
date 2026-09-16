"""Графики. §8 задания, семь обязательных плюс график к решению реализации №1.

Все графики — ФАЙЛАМИ НА ДИСК, рядом с числами. Не в окно.

Порядок функций — порядок, в котором run.py их вызывает; имя каждой говорит,
какой это номер из §8 задания.

    plot_entropy_per_iteration      1. энтропия по итерациям, кривая на блок
    plot_stop_criterion             2. критерий останова с линией mu
    plot_phase_true_found           3. внесённая и найденная фаза и разность
    plot_image_before_after         4. кусок изображения до и после
    plot_azimuth_cut                5. срез через точечную цель, в децибелах
    plot_block_map                  6. карта блоков с номерами по (5-26)
    plot_residual_versus_magnitude  7. остаточная против размаха внесённой
    plot_power_floor_effect         решение №1: влияние порога на энтропию
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # на диск, не в окно

import matplotlib.pyplot as plt
import numpy as np

import validate as V

#: Динамический диапазон картинок в децибелах — общая шкала яркости «до» и
#: «после» (§8, график 4: шкала обязана быть одинаковой, иначе сравнение
#: показывает работу автоматической яркости, а не работу алгоритма).
IMAGE_DYNAMIC_RANGE_DB = 45.0


def _decibels(image: np.ndarray) -> np.ndarray:
    """Мощность в децибелах относительно максимума. Принимает комплексное
    изображение; возвращает вещественный массив в дБ."""
    power = np.abs(image) ** 2
    peak = power.max()
    return 10.0 * np.log10(np.maximum(power, peak * 1e-12) / max(peak, 1e-300))


def plot_entropy_per_iteration(full_run: dict, path: pathlib.Path) -> pathlib.Path:
    """График 1 §8: энтропия по итерациям, кривая на блок. Должна падать
    монотонно; не падает — это результат, и его надо показать."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    non_monotonic = 0
    for block in full_run["per_block"]:
        history = np.asarray(block["_history"])
        if np.any(np.diff(history) > 1e-12):
            non_monotonic += 1
        ax.plot(np.arange(1, history.size + 1), history, marker=".", linewidth=1,
                label=f"q_k={block['q_k']}")
    ax.set_xlabel("номер итерации l")
    ax.set_ylabel("нормированная энтропия S, (5-6)")
    ax.set_title(
        f"1. Энтропия по итерациям, кривая на блок\n"
        f"блоков {full_run['q']}, немонотонных кривых: {non_monotonic}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_stop_criterion(full_run: dict, mu: float, path: pathlib.Path) -> pathlib.Path:
    """График 2 §8: критерий останова (5-9) по итерациям с горизонтальной
    линией mu."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for block in full_run["per_block"]:
        criterion = np.asarray(block["_criterion"])
        ax.semilogy(np.arange(1, criterion.size + 1), np.maximum(criterion, 1e-18),
                    marker=".", linewidth=1, label=f"q_k={block['q_k']}")
    ax.axhline(mu, color="k", linestyle="--", linewidth=1.6, label=f"порог mu = {mu:g}")
    ax.set_xlabel("номер итерации l")
    ax.set_ylabel(r"$\max_k |e^{j\varphi^{(l+1)}} - e^{j\varphi^{(l)}}|$, (5-9)")
    ax.set_title("2. Критерий останова (5-9) по итерациям")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_phase_true_found(case: dict, path: pathlib.Path) -> pathlib.Path:
    """График 3 §8: внесённая и найденная фаза на одной оси, и отдельно их
    разность после снятия линейной части. Это и есть доказательство
    правильности."""
    truth, estimate = case["_phi_true"], case["_phi_est"]
    M = truth.size
    k = V.centred_bin_index(M)
    order = np.argsort(k)
    a, b, residual = V.ambiguity_fit(truth, estimate)
    aligned = estimate - a - b * k  # оценка, приведённая к истине по (5-9)-мере

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(k[order], truth[order], linewidth=2, label="внесённая (истина)")
    ax1.plot(k[order], aligned[order], linewidth=1.2, linestyle="--",
             label="найденная, за вычетом a + b·k")
    ax1.set_ylabel("фаза, рад")
    ax1.set_title(
        f"3. Внесённая и найденная фаза\n"
        f"сцена «{case['scene']}», ошибка «{case['error']}», "
        f"размах {case['edge_rad']:g} рад на краю апертуры",
        fontsize=11,
    )
    ax1.grid(alpha=0.3)
    ax1.legend()

    rms = float(np.sqrt(np.mean(residual**2)))
    ax2.plot(k[order], residual[order], linewidth=1, color="crimson")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("номер доплеровского бина k (центрированный)")
    ax2.set_ylabel("разность, рад")
    ax2.set_title(f"разность после снятия постоянной и линейной: СКО = {rms:.3e} рад")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_image_before_after(
    before: np.ndarray, after: np.ndarray, title: str, path: pathlib.Path
) -> pathlib.Path:
    """График 4 §8: кусок изображения до и после, ОДИНАКОВАЯ шкала яркости."""
    db_before, db_after = _decibels(before), _decibels(after)
    vmax = max(db_before.max(), db_after.max())
    vmin = vmax - IMAGE_DYNAMIC_RANGE_DB

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, label in ((axes[0], db_before, "до"), (axes[1], db_after, "после")):
        image = ax.imshow(data, aspect="auto", origin="lower", cmap="gray",
                          vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(label)
        ax.set_xlabel("строб дальности n")
        ax.set_ylabel("азимут m")
        fig.colorbar(image, ax=ax, label="дБ")
    fig.suptitle(f"4. {title} (шкала яркости общая, {IMAGE_DYNAMIC_RANGE_DB:g} дБ)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_azimuth_cut(case: dict, path: pathlib.Path) -> pathlib.Path:
    """График 5 §8: срез через точечную цель по азимуту, до и после, в дБ."""
    column, row = case["_cut_col"], case["_cut_row"]
    before = np.abs(case["_image_before"][:, column]) ** 2
    after = np.abs(case["_image_after"][:, column]) ** 2
    reference = after.max()
    to_db = lambda x: 10 * np.log10(np.maximum(x, reference * 1e-9) / reference)

    M = before.size
    shifted = np.arange(M) - row
    order = np.argsort(((shifted + M // 2) % M) - M // 2)
    axis = ((shifted + M // 2) % M) - M // 2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(axis[order], to_db(before)[order], linewidth=1.4, label="до")
    ax.plot(axis[order], to_db(after)[order], linewidth=1.4, label="после")
    ax.axhline(-3, color="k", linestyle=":", linewidth=1, label="уровень -3 дБ")
    ax.set_xlim(-24, 24)
    ax.set_ylim(-60, 3)
    ax.set_xlabel("азимут относительно цели, отсчёты")
    ax.set_ylabel("мощность, дБ относительно пика «после»")
    ax.set_title(
        f"5. Срез через точечную цель по азимуту (строб n={column})\n"
        f"выигрыш по пику {case['peak_gain_db']:+.2f} дБ, "
        f"ширина по -3 дБ {case['width_before']:.2f} -> {case['width_after']:.2f} отсчёта"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_block_map(full_run: dict, path: pathlib.Path) -> pathlib.Path:
    """График 6 §8: карта блоков на сцене с номерами по (5-26) — чтобы видеть,
    что разбиение легло так, как задумано.

    Ось азимута здесь — ось пикселей изображения m: этап A режет СЦЕНУ на
    участки местности (§3.3 документа).
    """
    image_db = _decibels(full_run["_image_after"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_db, aspect="auto", origin="lower", cmap="gray",
              vmin=image_db.max() - IMAGE_DYNAMIC_RANGE_DB, vmax=image_db.max(),
              interpolation="nearest")
    for block in full_run["blocks"]:
        ax.add_patch(plt.Rectangle(
            (block.n_start - 0.5, block.m_start - 0.5),
            block.n_stop - block.n_start, block.m_stop - block.m_start,
            fill=False, edgecolor="deepskyblue", linewidth=1.4))
        ax.text((block.n_start + block.n_stop) / 2, (block.m_start + block.m_stop) / 2,
                f"$q_k$={block.q_k}\n$m_k$={block.m_k}, $n_k$={block.n_k}",
                color="yellow", ha="center", va="center", fontsize=8)
    ax.set_xlabel("строб дальности n")
    ax.set_ylabel("азимутальный пиксель m")
    ax.set_title(
        f"6. Карта блоков, нумерация по (5-26): $q_k = m_k + M_k(n_k-1)$\n"
        f"$M_k$={full_run['M_k']}, $N_k$={full_run['N_k']}, $q$={full_run['q']}; "
        f"$m_p$={full_run['m_p']:.1f}, $n_p$={full_run['n_p']:.1f} отсчёта"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_residual_versus_magnitude(sweep: list[dict], path: pathlib.Path) -> pathlib.Path:
    """График 7 §8: остаточная ошибка против размаха внесённой — где алгоритм
    ломается."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for kind in sorted({row["error"] for row in sweep}):
        rows = sorted((r for r in sweep if r["error"] == kind), key=lambda r: r["edge_rad"])
        edges = [r["edge_rad"] for r in rows]
        ax.loglog(edges, [r["residual_rms_rad"] for r in rows], marker="o", label=f"{kind}, остаток")
        ax.loglog(edges, [r["initial_rms_rad"] for r in rows], marker=".", linestyle=":",
                  alpha=0.55, label=f"{kind}, было внесено")
        for r in rows:
            if not r["converged"]:
                ax.plot(r["edge_rad"], r["residual_rms_rad"], marker="x", color="k",
                        markersize=11, linestyle="none")
    ax.plot([], [], marker="x", color="k", linestyle="none", label="(5-9) НЕ выполнено")
    ax.set_xlabel("размах внесённой ошибки на краю апертуры, рад")
    ax.set_ylabel("СКО, рад")
    ax.set_title("7. Остаточная ошибка против размаха внесённой")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_power_floor_effect(rows: list[dict], path: pathlib.Path) -> pathlib.Path:
    """Решение реализации №1, §6.1 задания: «что бы ни выбрали — показать на
    графике, как выбор влияет на энтропию»."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for scene_kind in sorted({r["scene"] for r in rows}):
        selected = sorted((r for r in rows if r["scene"] == scene_kind),
                          key=lambda r: r["floor_relative"])
        floors = [r["floor_relative"] for r in selected]
        ax1.semilogx(floors, [r["entropy_after"] for r in selected], marker="o", label=scene_kind)
        ax2.loglog(floors, [r["residual_rms_rad"] for r in selected], marker="o", label=scene_kind)
    for ax in (ax1, ax2):
        ax.axvline(1e-12, color="k", linestyle="--", linewidth=1.2)
        ax.set_xlabel("порог мощности, в долях средней (решение №1)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    ax1.set_ylabel("нормированная энтропия после, (5-6)")
    ax1.set_title("энтропия против выбора порога")
    ax2.set_ylabel("остаточное СКО, рад")
    ax2.set_title("остаточная ошибка против выбора порога")
    fig.suptitle("Решение реализации №1: порог под логарифмом (пунктир — принятое значение 1e-12)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

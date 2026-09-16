"""Песочница для разглядывания данных. НЕ часть алгоритма.

Отдельный файл, ничего из него никто не импортирует: он существует только
чтобы можно было пройти по шагам §5.8 руками, посмотреть на массивы и
нарисовать срезы.

Как пользоваться:

    python3 playground.py              прогнать целиком, картинки лягут в look/

    в VS Code: F5 -> «MN-MEA: песочница playground.py»
    точки останова ставятся мышью слева от номера строки

Внутри стоит строка breakpoint() — на ней запуск ОСТАНОВИТСЯ и пустит вас
в отладчик. Хотите пройти без остановки — закомментируйте её.
"""

from __future__ import annotations

# Стенд лежит отдельно от алгоритма: algorithm/ рядом, и путь к нему
# добавляется здесь явно. Алгоритм про стенд не знает ничего и знать не
# должен — это и есть граница, по которой его снимать на реальные данные.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "algorithm"))


import pathlib

import matplotlib

matplotlib.use("Agg")  # рисуем в файл, окна в отладчике ненадёжны

import matplotlib.pyplot as plt
import numpy as np

import stage_c_iterate as C
import synthetic as SY
from backend import get_backend

#: Куда складывать картинки песочницы.
LOOK = pathlib.Path(__file__).resolve().parent.parent / "look"


def в_numpy(массив):
    """Снять массив с устройства, если он там лежит.

    Рисовать и печатать умеет только numpy на процессоре, а на видеокарте
    np.asarray(массив) запрещён — cupy отвечает отказом и требует явного
    .get(). Снятие делает backend.to_numpy, и делается оно ЗДЕСЬ, в одном
    месте, а не по всем помощникам вразнобой.
    """
    return backend.to_numpy(массив)


def показать(массив, подпись: str, столбец: int = 0) -> pathlib.Path:
    """Нарисовать срез массива по азимуту и сохранить в файл.

    Принимает массив (M, N) или вектор (M,), подпись и номер столбца;
    возвращает путь к сохранённому файлу и печатает его.
    """
    LOOK.mkdir(exist_ok=True)
    данные = в_numpy(массив)
    срез = данные if данные.ndim == 1 else данные[:, столбец]

    fig, ax = plt.subplots(figsize=(9, 4))
    if np.iscomplexobj(срез):
        мощность = np.abs(срез) ** 2
        ax.plot(10 * np.log10(np.maximum(мощность, мощность.max() * 1e-9) / мощность.max()))
        ax.set_ylabel("мощность, дБ от пика")
    else:
        ax.plot(срез)
        ax.set_ylabel("значение")
    ax.set_xlabel("номер отсчёта")
    ax.set_title(подпись)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    путь = LOOK / (подпись.replace(" ", "_").replace("/", "_") + ".png")
    fig.savefig(путь, dpi=120)
    plt.close(fig)
    print(f"   картинка: {путь}")
    return путь


def карта(массив, подпись: str) -> pathlib.Path:
    """Нарисовать весь массив (M, N) как картинку в децибелах.
    Принимает массив и подпись; возвращает путь к файлу."""
    LOOK.mkdir(exist_ok=True)
    мощность = np.abs(в_numpy(массив)) ** 2
    дб = 10 * np.log10(np.maximum(мощность, мощность.max() * 1e-6) / мощность.max())

    fig, ax = plt.subplots(figsize=(7, 6))
    картинка = ax.imshow(дб, aspect="auto", origin="lower", cmap="gray", vmin=-45, vmax=0)
    fig.colorbar(картинка, ax=ax, label="дБ")
    ax.set_xlabel("строб дальности n")
    ax.set_ylabel("азимут m")
    ax.set_title(подпись)
    fig.tight_layout()

    путь = LOOK / (подпись.replace(" ", "_").replace("/", "_") + ".png")
    fig.savefig(путь, dpi=120)
    plt.close(fig)
    print(f"   картинка: {путь}")
    return путь


def глянуть(имя: str, массив) -> None:
    """Напечатать про массив всё существенное: форму, тип, размах.
    Принимает имя и массив; ничего не возвращает."""
    a = в_numpy(массив)
    if np.iscomplexobj(a):
        print(f"   {имя:8s} {str(a.shape):10s} {str(a.dtype):12s} "
              f"|min|={np.abs(a).min():.4e}  |max|={np.abs(a).max():.4e}")
    else:
        print(f"   {имя:8s} {str(a.shape):10s} {str(a.dtype):12s} "
              f"min={a.min():.4e}  max={a.max():.4e}")


# ---------------------------------------------------------------- готовим блок
backend = get_backend("auto")
M, N = 128, 32
rng = np.random.default_rng(20250915)

sc = SY.scene("points_and_clutter", M, N, rng)
phi_err = SY.phase_error("quadratic", M, 3.0, rng)      # ИСТИНА этого опыта
h = backend.asarray(SY.range_doppler_from_scene(backend, sc.image, phi_err))
phi = backend.asarray(np.zeros(M))                      # стартуем с нуля

print(f"блок {M}x{N}, точечных целей {len(sc.points)}, первая в {sc.points[0]}")
print(f"внесена ошибка: квадратичная, {3.0} рад на краю апертуры")
print()

# ------------------------------------------------- девять шагов §5.8, по одному
print("ШАГИ 1-2, (5-3): строим изображение при текущей phi")
h_phi, g = C.image_from_phase(backend, h, phi)
глянуть("h_phi", h_phi)
глянуть("g", g)
карта(backend.to_numpy(g), "01 изображение при phi=0 (расфокусировано)")
показать(backend.to_numpy(g), "02 срез по азимуту при phi=0", столбец=sc.points[0][1])

print("\nШАГ 3: мощность и логарифм")
h_abs2 = backend.xp.abs(h) ** 2
S_g = C.total_energy(backend, h_abs2, M)
floor = C.power_floor(S_g, M, N)
P, ln_P, доля = C.image_power(backend, g, floor)
глянуть("P", P)
глянуть("ln_P", ln_P)
print(f"   S_g = {S_g:.4e},  порог = {floor:.4e},  доля на пороге = {доля}")

print("\n(5-5),(5-6): энтропия")
E_g, S = C.entropy(backend, P, S_g)
print(f"   E_g = {E_g:.4f}   S = {S:.4f}")

print("\nШАГ 4, (5-14): вспомогательный массив G")
G = C.auxiliary_array(backend, ln_P, g)
глянуть("G", G)

print("\nШАГ 5: W = G* h_phi — единственное место")
W = C.w_product(backend, G, h_phi)
глянуть("W", W)

print("\nШАГИ 6-7: производные")
E_1 = C.first_derivative(backend, W)
E_2 = C.second_derivative(backend, W, h_abs2, ln_P)
глянуть("E_1", E_1)
глянуть("E_2", E_2)
показать(backend.to_numpy(E_1), "03 первая производная (5-13) по бинам")
показать(backend.to_numpy(E_2), "04 вторая производная (5-19) по бинам")
print(f"   бинов с E'' <= 0: {int(np.sum(backend.to_numpy(E_2) <= 0))} из {M}")

print("\nШАГ 8, (5-8): шаг Ньютона")
phi_new, заморожено, ограничено = C.newton_update(phi, E_1, E_2, backend)
print(f"   заморожено бинов: {заморожено},  упёрлось в предел: {ограничено}")
показать(backend.to_numpy(phi_new), "05 phi после одного шага")

print("\nШАГ 9, (5-9): критерий останова")
print(f"   criterion = {C.stop_criterion(backend, phi_new, phi):.4e}")

# ---------------------------------------------------- вот тут запуск встанет
print("\n>>> точка останова: смотрите переменные h, g, P, W, E_1, E_2, phi_new")
print(">>> в Debug Console можно писать что угодно, например:")
print(">>>     показать(backend.to_numpy(g), 'моя картинка')")
print(">>>     np.abs(backend.to_numpy(g)).max()   # с карты сначала снять")
breakpoint()

# --------------------------------------------- а теперь то же самое до конца
print("\nПРОГОН ЦЕЛИКОМ")
r = C.iterate_block(backend, h, np.zeros(M), mu=1e-3)
print(f"   итераций {r.n_iterations}, сошлось {r.converged}")
print(f"   S: {r.normalised_entropy_history[0]:.4f} -> {r.normalised_entropy_final:.4f}")

g_после = C.image_from_phase(backend, h, backend.asarray(r.phi))[1]
карта(backend.to_numpy(g_после), "06 изображение после сходимости")
показать(backend.to_numpy(g_после), "07 срез по азимуту после", столбец=sc.points[0][1])
показать(r.normalised_entropy_history, "08 энтропия по итерациям")
показать(r.criterion_history, "09 критерий останова по итерациям")
показать(r.phi - phi_err, "10 найденная минус внесённая")

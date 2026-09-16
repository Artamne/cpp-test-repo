"""Срез для MN-MEA из результата продукта: файл этапа `az` (`*_rda@*.h5`).

Запуск — один аргумент, путь к rda-файлу:

    python prepare_slice_rda.py "D:\\test1\\results\\src-0454fb28e3_rda@4ec41d.h5"

Больше ничего вводить не надо. Всё остальное скрипт находит сам:

    паспорт съёмки   — внутри того же h5, группа `params` и `params/__extra__`
    скорость         — `..\\cache\\src-<тот же id>_vel@*.npz`, свежий по времени
    проверка записи  — `..\\cache\\check_src-<тот же id>.json`

Кладёт рядом с rda-файлом папку `mn_mea_srez/`:

    h_srez.npy   комплексный массив (M, N) — вход этапа A MN-MEA
    meta.json    Geometry и всё, что к нему полагается
    srez.png     картинка среза, посмотреть глазами

Порядок функций — порядок выполнения:

    find_companions     найти vel-npz и check-json рядом с rda
    read_h5_passport    паспорт из самого h5: params + __extra__ + stage
    read_velocity       матрица скорости: выбрать согласную зону
    range_axis_metres   ось дальности в метрах
    choose_slice        границы среза по дальности и по азимуту
    read_slice          прочитать окно и перевернуть в (азимут, дальность)
    to_range_doppler    h = ifft(g), чтобы fft(h) = g тождественно
    build_meta          собрать Geometry и проверки
    main

ПОЧЕМУ ВХОД ИМЕННО `rda`, А НЕ `rcmc`. В `rcmc@*.h5` азимут ещё во временной
области и НЕ сжат: остаточная фаза там не единицы радиан, а вся азимутальная
ЛЧМ целиком. MN-MEA такое не вытянет — цель в картинке при phi = 0 размазана
шире своего блока. `rda@*.h5` — этап `az`, изображение уже построено, в нём
осталась только та фаза, которую прямолинейная модель снять не смогла. Её и
ищет MN-MEA.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------- настройки

#: Половина окна по дальности, в долях R_B0. Geometry в MN-MEA ОДНА на сцену,
#: а T_a пропорционально дальности: +-10 % по дальности держат T_a в тех же
#: +-10 % по всему окну.
RANGE_HALF_FRACTION = 0.10

#: Порог отбора ячеек скорости: берём те, чей вес не ниже этой доли от
#: максимального веса в зоне. Ячейки с малым весом на этой записи расходятся
#: в разы (28…58 м/с), и усреднять их со всеми подряд нельзя.
WEIGHT_KEEP_FRACTION = 0.5

#: Расширение главного лепестка весовым окном при азимутальном сжатии. Входит
#: в длину апертуры: L = gamma * lambda * R / (2 * rho). В паспорте продукта
#: этого числа нет — если в `params/__extra__` не найдётся, берётся отсюда и
#: попадает в meta.json как ДОПУЩЕНИЕ.
GAMMA_WIN_DEFAULT = 1.3

#: Разрешение по азимуту, если в паспорте кадра его нет. Совпадает с
#: `sar_core.resolution.DEFAULT_AZIMUTH_RESOLUTION_M`: там прямо написано, что
#: это умолчание продукта, паспортом не подтверждённое.
RESOLUTION_DEFAULT_M = 0.4

C_LIGHT = 299_792_458.0
OUT_NAME = "mn_mea_srez"


def find_companions(rda_path: str) -> dict:
    """Найти vel-npz и check-json, относящиеся к этому же источнику.

    Принимает путь к rda-файлу; возвращает словарь с путями и id источника.

    Имена в кэше устроены как `src-<id>_<этап>@<хеш>`, и id один на всю
    цепочку. Берётся самый свежий vel по времени правки: он отвечает
    последнему прогону, а rda лежит в той же серии.
    """
    name = os.path.basename(rda_path)
    m = re.match(r"(src-[0-9a-f]+)_", name)
    if not m:
        sys.exit(f"не разобрать id источника в имени {name!r}; "
                 "ожидалось src-<id>_rda@<хеш>.h5")
    src = m.group(1)
    cache = os.path.join(os.path.dirname(os.path.dirname(rda_path)), "cache")

    vels = sorted(glob.glob(os.path.join(cache, f"{src}_vel@*.npz")),
                  key=os.path.getmtime)
    if not vels:
        sys.exit(f"рядом нет файла скорости {src}_vel@*.npz в {cache}")
    check = os.path.join(cache, f"check_{src}.json")
    return {"src": src, "vel": vels[-1],
            "check": check if os.path.isfile(check) else ""}


def read_h5_passport(rda_path: str) -> dict:
    """Паспорт съёмки из самого rda-файла.

    Принимает путь; возвращает словарь: stage, форма массива, скалярные
    параметры (`fc`, `prf`, `fs_range`, …) и векторы из `params/__extra__`
    (`R_vec`, `v_per_col`, …). Массив НЕ читается.
    """
    import h5py

    out: dict = {"extra": {}}
    with h5py.File(rda_path, "r") as f:
        out["stage"] = _plain(f.attrs.get("stage", ""))
        ds = f["array"]
        out["shape"] = (int(ds.shape[0]), int(ds.shape[1]))
        out["dtype"] = str(ds.dtype)
        grp = f.get("params")
        if grp is not None:
            for k in grp.attrs:
                out[k] = _plain(grp.attrs[k])
            ex = grp.get("__extra__")
            if ex is not None:
                for k in ex.attrs:
                    out["extra"][k] = _plain(ex.attrs[k])
                for k in ex.keys():
                    out["extra"][k] = np.asarray(ex[k])
    return out


def _plain(value):
    """h5-атрибут → обычный питон. Строки в h5 лежат байтами, скаляры —
    нульмерными массивами."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return _plain(value.reshape(-1)[0])
    return value


def read_velocity(vel_path: str) -> dict:
    """Скорость среза и место, где её брать, из матрицы скорости.

    Принимает путь к vel-npz; возвращает словарь: v, r_bin, az_bin (центр
    согласной области), список взятых ячеек.

    Скорость здесь МАТРИЦА по дальности и азимуту, и ячейки расходятся: на
    записи владельца от 28 до 58 м/с. Поэтому:

      1. отбрасываются ячейки без числа (`v_star` = NaN) и краевые (`edge`);
      2. выбирается зона дальности с наибольшей СУММОЙ весов — там оценке
         есть на что опереться;
      3. внутри зоны берутся ячейки с весом не ниже WEIGHT_KEEP_FRACTION от
         максимального в ней, и по ним считается взвешенное среднее.

    Центр окна по азимуту — взвешенный центр тех же ячеек. Срез встаёт туда,
    где скорость известна, а не туда, где красивее картинка.
    """
    z = np.load(vel_path, allow_pickle=True)
    rows = json.loads(str(z["bands.rows"]))
    good = [r for r in rows
            if r.get("v_star") == r.get("v_star")     # не NaN
            and not r.get("edge") and float(r.get("weight", 0)) > 0]
    if not good:
        sys.exit(f"в {os.path.basename(vel_path)} нет ни одной годной ячейки "
                 "скорости: все NaN или краевые")

    zones: dict[int, list] = {}
    for r in good:
        zones.setdefault(int(r["r_bin"]), []).append(r)
    r_bin = max(zones, key=lambda rb: sum(float(r["weight"]) for r in zones[rb]))

    cells = zones[r_bin]
    w_max = max(float(r["weight"]) for r in cells)
    kept = [r for r in cells
            if float(r["weight"]) >= WEIGHT_KEEP_FRACTION * w_max]
    w = np.array([float(r["weight"]) for r in kept])
    v = np.array([float(r["v_star"]) for r in kept])
    az = np.array([float(r["az_bin"]) for r in kept])
    return {
        "v": float(np.sum(w * v) / np.sum(w)),
        "r_bin": int(r_bin),
        "az_bin": int(round(float(np.sum(w * az) / np.sum(w)))),
        "kept": [(int(r["az_bin"]), float(r["v_star"]), float(r["weight"]))
                 for r in kept],
        "n_total": len(rows),
        "n_good": len(good),
    }


def range_axis_metres(passport: dict, check: dict, ka_dir: str) -> np.ndarray:
    """Ось дальности в метрах, по одному числу на строб.

    Принимает паспорт из h5, содержимое check-json и каталог кэша;
    возвращает вектор длины Nr.

    Порядок источников — от надёжного к выводимому:

      1. `params/__extra__/R_vec` — если он в файле есть, это и есть ось;
      2. таблица полос из `*_ka@*.npz`: её `r_lo_m` даёт начало оси, а шаг
         выводится из частоты дискретизации, `r_b = c / (2 fs)`.

    Если не вышло ни то, ни другое — отказ, а не догадка: без оси дальности
    ни R_B0, ни размер блока по дальности посчитать не из чего.
    """
    R_vec = passport["extra"].get("R_vec")
    if isinstance(R_vec, np.ndarray) and R_vec.size == passport["shape"][0]:
        return R_vec.astype(np.float64).ravel()

    fs = float(passport.get("fs_range") or check.get("fs_hz") or 0.0)
    if fs <= 0:
        sys.exit("нет ни R_vec в паспорте, ни частоты дискретизации — "
                 "ось дальности построить не из чего")
    r_b = C_LIGHT / (2.0 * fs)

    kas = sorted(glob.glob(os.path.join(ka_dir, "*_ka@*.npz")),
                 key=os.path.getmtime)
    if not kas:
        sys.exit("нет R_vec в паспорте и нет ни одного *_ka@*.npz — "
                 "начало оси дальности взять неоткуда")
    table = json.loads(str(np.load(kas[-1], allow_pickle=True)
                           ["by_range_table.rows"]))
    r0 = float(table[0]["r_lo_m"])
    return r0 + r_b * np.arange(passport["shape"][0], dtype=np.float64)


def choose_slice(R_axis: np.ndarray, vel: dict, n_az: int,
                 prf: float, lambda_: float, rho_a: float,
                 gamma: float) -> dict:
    """Границы среза по дальности и по азимуту.

    Принимает ось дальности, выбор скорости, число импульсов кадра, PRF,
    длину волны, разрешение и расширение окна; возвращает словарь границ.

        R_B0 = R_axis[r_bin]                          центр зоны скорости
        T_a  = gamma * lambda * R_B0 / (2 * rho_a * v)   время апертуры
        M    = T_a * PRF                              импульсов в апертуре

    Окно по дальности — +-RANGE_HALF_FRACTION от R_B0 (см. настройку).
    Окно по азимуту — M импульсов вокруг центра согласных ячеек скорости,
    подвинутое внутрь кадра, если упёрлось в край.
    """
    n_r = R_axis.size
    r_b = float(np.mean(np.diff(R_axis)))
    r_bin = min(max(vel["r_bin"], 0), n_r - 1)
    R_B0 = float(R_axis[r_bin])

    half = int(RANGE_HALF_FRACTION * R_B0 / r_b)
    r0, r1 = max(0, r_bin - half), min(n_r, r_bin + half)

    T_a = gamma * lambda_ * R_B0 / (2.0 * rho_a * vel["v"])
    M = int(round(T_a * prf))
    M = min(M, n_az)
    a0 = int(vel["az_bin"] - M // 2)
    a0 = min(max(a0, 0), n_az - M)
    return {"r0": r0, "r1": r1, "a0": a0, "a1": a0 + M,
            "R_B0": R_B0, "r_b": r_b, "T_a": T_a, "M_aperture": int(round(T_a * prf))}


def read_slice(rda_path: str, cut: dict) -> np.ndarray:
    """Прочитать окно и перевернуть в раскладку MN-MEA.

    Принимает путь и границы; возвращает массив (азимут, дальность) complex64.

    В продукте массив лежит как (Nr, Naz) — дальность по оси 0. MN-MEA
    суммирует (5-3) по азимуту и ждёт азимут по оси 0, поэтому здесь
    транспонирование, и это единственное место, где оно делается.
    """
    import h5py

    with h5py.File(rda_path, "r") as f:
        block = f["array"][cut["r0"]:cut["r1"], cut["a0"]:cut["a1"]]
    return np.ascontiguousarray(np.asarray(block).T.astype(np.complex64))


def to_range_doppler(g: np.ndarray) -> np.ndarray:
    """h = ifft(g) по азимуту — вход этапа A.

    Принимает изображение среза (M, N); возвращает h той же формы.

    Соглашение (5-3) требует, чтобы ПРЯМОЕ ПФ от h давало картинку, а numpy
    даёт fft(ifft(g)) = g тождественно. Никакой расфокусировки здесь нет,
    меняется только представление.

    ПОБОЧНОЕ СЛЕДСТВИЕ. Раз h = ifft(g), а настоящий спектр есть fft(g), то
    бин k в h отвечает ОТРИЦАТЕЛЬНОЙ доплеровской частоте: h[k] = S[-k]/M.
    На (5-27) не влияет — eps ~ f_a^2 чётна. На чём-либо линейном по f_a
    влияло бы.
    """
    return np.fft.ifft(g, axis=0).astype(np.complex64)


def image_entropy(image: np.ndarray) -> float:
    """Нормированная энтропия (5-6): мера сфокусированности, меньше — резче.
    Считается так же, как `stage_c_iterate.entropy`, чтобы числа были
    сравнимы с отчётом MN-MEA."""
    P = np.abs(image).astype(np.float64) ** 2
    S_g = float(P.sum())
    P = np.maximum(P, S_g / P.size * 1e-12)
    return -float(np.sum(P * np.log(P))) / S_g + math.log(S_g)


def build_meta(rda_path: str, passport: dict, check: dict, vel: dict,
               cut: dict, lambda_: float, rho_a: float, rho_src: str,
               gamma: float, gamma_src: str, prf: float,
               h: np.ndarray, resid: float, entropy: float,
               y_m_over_R: float = 0.9) -> dict:
    """Собрать Geometry, допущения и проверки в один словарь для meta.json.

    Ни одно число не назначено рукой: всё выведено из паспорта h5, матрицы
    скорости и границ среза. Что вывести не из чего — стоит в «допущениях» с
    прямым указанием, что это допущение.
    """
    v = vel["v"]
    R_B0 = cut["R_B0"]
    y_m = y_m_over_R * R_B0
    z_m = -math.sqrt(max(R_B0**2 - y_m**2, 0.0))
    f_0 = C_LIGHT / lambda_

    assumptions = [
        "ускорение носителя нулевое: в паспорте кадра его нет. Следствие: "
        "A_2 = v^2, mu_3 = 0, этап B вырождается в прямолинейный полёт и "
        "даёт phi^(0) около нуля — сходимость будет медленнее",
        f"y_m / R_B0 = {y_m_over_R}: высоты носителя в паспорте нет. Входит "
        "только в знаменатель (5-29) через полуразмер блока по дальности; "
        "на нарезку при N_k = 1 не влияет",
        f"скорость взята одним числом {v:.3f} м/с — взвешенное среднее "
        f"{len(vel['kept'])} согласных ячеек матрицы в зоне r_bin="
        f"{vel['r_bin']}; книга требует одно число на апертуру",
    ]
    if rho_src != "паспорт":
        assumptions.append(
            f"разрешение по азимуту {rho_a} м — {rho_src}, паспортом не "
            "подтверждено; от него один к одному зависит длина апертуры")
    if gamma_src != "паспорт":
        assumptions.append(
            f"расширение окна gamma = {gamma} — {gamma_src}; входит в длину "
            "апертуры тем же множителем")

    return {
        "istochnik": {
            "fail": os.path.basename(rda_path),
            "proverka": os.path.basename(check.get("__path__", "")),
            "stage": passport["stage"],
            "forma_kadra": list(passport["shape"]),
            "zapis": (check.get("files") or [{}])[0].get("name", ""),
            "stadiya": "дальность сжата, RCMC выполнена, азимут сжат штатным "
                       "трактом продукта (этап az); остаточная фаза — то, что "
                       "прямолинейная модель снять не смогла",
            "bez_INS": True,
        },
        "srez": {
            "stroby": [cut["r0"], cut["r1"]],
            "impulsy": [cut["a0"], cut["a1"]],
            "forma": list(h.shape),
            "zamechanie": "M — длина массива по азимуту; M_aperture и T_a "
                          "описывают апертуру радара, это разные числа",
        },
        "dannye": {
            "fail": "h_srez.npy",
            "dtype": "complex64",
            "oblast": "дальность-доплер",
            "pravilo": "g = numpy.fft.fft(h, axis=0)",
            "raskladka_k": "numpy.fft, ноль в бине 0, НЕ fftshift",
            "znak_dopplera": "бин k отвечает частоте -fftfreq(k); на (5-27) "
                             "не влияет, eps ~ f_a^2 чётна",
        },
        "geometry": {
            "v_x0": v, "v_y0": 0.0, "v_z0": 0.0,
            "a_x": 0.0, "a_y": 0.0, "a_z": 0.0,
            "x_m": 0.0, "y_m": y_m, "z_m": z_m,
            "R_B0": R_B0,
            "lambda_": lambda_,
            "f_0": f_0,
            "r_a": v / prf,
            "r_b": cut["r_b"],
            "T_a": cut["T_a"],
            "c": C_LIGHT,
        },
        "prf": prf,
        "m_aperture": cut["M_aperture"],
        # два разных числа, сведённые в коде MN-MEA в одно r_a: шаг идёт в
        # (5-22), разрешение — в (5-20). На этой записи они расходятся в разы
        "r_a_shag": v / prf,
        "r_a_razreshenie": rho_a,
        "skorost": {
            "vzyato": v,
            "zona_r_bin": vel["r_bin"],
            "centr_az_bin": vel["az_bin"],
            "yacheek_vsego": vel["n_total"],
            "yacheek_godnyh": vel["n_good"],
            "yacheek_vzyato": [{"az_bin": a, "v": vv, "ves": w}
                               for a, vv, w in vel["kept"]],
        },
        "dopushcheniya": assumptions,
        "proverki": {
            # вектор на центр сцены обязан иметь длину R_B0 — это одна и та же
            # точка. Число считается, а не назначается: назначенный ноль
            # означал бы «сошлось» даже когда не сошлось
            "vektor_i_dalnost": float(abs(y_m**2 + z_m**2 - R_B0**2)),
            "lambda_f0_minus_c": float(abs(lambda_ * f_0 - C_LIGHT)),
            "m_aperture_delit_T_a": float(cut["M_aperture"] / cut["T_a"]),
            "prf_pasporta": prf,
            "fft_h_minus_g_otnositelno": resid,
            "entropiya_sreza": entropy,
        },
        "ozhidaemaya_narezka": {
            "m_p_kak_schitaet_kod": 4.0 * rho_a / (gamma * lambda_),
            "m_p_pravilnyy": 4.0 * rho_a**2 / (gamma * lambda_ * (v / prf)),
            "N_k_ozhidaemoe": 1,
        },
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("Порядок функций")[0])
    rda = sys.argv[1]
    if not os.path.isfile(rda):
        sys.exit(f"нет файла {rda}")
    out_dir = os.path.join(os.path.dirname(rda), OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)

    comp = find_companions(rda)
    print(f"источник {comp['src']}")
    print(f"  скорость: {os.path.basename(comp['vel'])}")

    passport = read_h5_passport(rda)
    print(f"  этап '{passport['stage']}', массив {passport['shape']} "
          f"{passport['dtype']}")
    if passport["stage"] and "FOCUS" not in str(passport["stage"]).upper():
        print(f"  ВНИМАНИЕ: этап {passport['stage']!r}, а MN-MEA ждёт "
              "изображение этапа az (FOCUSED). Если это rcmc — азимут не "
              "сжат, подавать нельзя")

    check = {}
    if comp["check"]:
        with open(comp["check"], encoding="utf-8") as fp:
            check = json.load(fp)
        check["__path__"] = comp["check"]

    prf = passport.get("prf") or check.get("prf_hz")
    f_0 = passport.get("fc") or check.get("fc") or check.get("f0_hz")
    if not prf or not f_0:
        sys.exit("в паспорте кадра и в check-json нет "
                 f"{'PRF' if not prf else 'несущей частоты'}; "
                 "без неё ни доплеровская ось, ни длина волны не считаются")
    prf, f_0 = float(prf), float(f_0)
    lambda_ = C_LIGHT / f_0

    rho_a, rho_src = passport["extra"].get("resolution"), "паспорт"
    if not rho_a:
        rho_a, rho_src = RESOLUTION_DEFAULT_M, "умолчание продукта"
    gamma, gamma_src = passport["extra"].get("gamma_win"), "паспорт"
    if not gamma:
        gamma, gamma_src = GAMMA_WIN_DEFAULT, "умолчание скрипта"
    rho_a, gamma = float(rho_a), float(gamma)
    print(f"  PRF {prf:.4f} Гц   f_0 {f_0:.4e} Гц   lambda {lambda_:.7f} м")
    print(f"  разрешение {rho_a} м ({rho_src})   gamma {gamma} ({gamma_src})")

    R_axis = range_axis_metres(passport, check,
                               os.path.dirname(comp["vel"]))
    print(f"  дальность {R_axis[0]:.1f} … {R_axis[-1]:.1f} м, "
          f"шаг {np.mean(np.diff(R_axis)):.6f} м")

    vel = read_velocity(comp["vel"])
    print(f"\nскорость: годных ячеек {vel['n_good']} из {vel['n_total']}, "
          f"зона r_bin={vel['r_bin']}")
    for a, vv, w in vel["kept"]:
        print(f"    az_bin {a:6d}   v {vv:6.2f} м/с   вес {w:5.2f}")
    print(f"  взято {vel['v']:.3f} м/с, центр по азимуту {vel['az_bin']}")

    cut = choose_slice(R_axis, vel, passport["shape"][1], prf, lambda_,
                       rho_a, gamma)
    print(f"\nсрез")
    print(f"  R_B0 = {cut['R_B0']:.1f} м, T_a = {cut['T_a']:.4f} с, "
          f"M_aperture = {cut['M_aperture']}")
    print(f"  стробы   {cut['r0']} … {cut['r1']}  ({cut['r1']-cut['r0']})")
    print(f"  импульсы {cut['a0']} … {cut['a1']}  ({cut['a1']-cut['a0']})")

    g = read_slice(rda, cut)
    h = to_range_doppler(g)
    resid = float(np.max(np.abs(np.fft.fft(h, axis=0) - g))
                  / max(float(np.max(np.abs(g))), 1e-30))
    S = image_entropy(g)
    print(f"  h {h.shape} {h.dtype}, {h.nbytes/2**20:.0f} МБ")
    print(f"  сверка fft(h) == g: {resid:.2e} относительно")
    print(f"  нормированная энтропия среза: {S:.4f}")

    np.save(os.path.join(out_dir, "h_srez.npy"), h)
    meta = build_meta(rda, passport, check, vel, cut, lambda_, rho_a,
                      rho_src, gamma, gamma_src, prf, h, resid, S)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        A = np.abs(g)
        dB = 20 * np.log10(np.maximum(A, A.max() * 1e-5) / A.max())
        plt.figure(figsize=(10, 7))
        plt.imshow(dB, aspect="auto", cmap="gray", vmin=-40, vmax=0)
        plt.colorbar(label="дБ")
        plt.xlabel("дальность, отсчёты")
        plt.ylabel("азимут, отсчёты")
        plt.title(f"срез, R_B0 = {cut['R_B0']:.0f} м, v = {vel['v']:.1f} м/с, "
                  f"энтропия {S:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "srez.png"), dpi=110)
    except ImportError:
        print("  matplotlib нет, картинку пропустили")

    print(f"\nготово: {out_dir}")
    print("  h_srez.npy   вход этапа A")
    print("  meta.json    Geometry, допущения, проверки")
    print("  srez.png     на что смотреть глазами")


if __name__ == "__main__":
    main()

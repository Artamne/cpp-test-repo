"""Подготовка кусочка реальных данных для MN-MEA: из SAR_RCMC_Result.mat
делает массив h(k,n) на нужной стадии плюс паспорт к нему.

Запускать на машине, где лежат данные:

    python bench/prepare_slice.py                 # путь из DATA_DIR ниже
    python bench/prepare_slice.py E:\путь\к\данным  # или аргументом

Ничего не спрашивает, ничего не перезаписывает во входных файлах. Кладёт рядом
папку mn_mea_srez/ с тремя файлами:

    h_srez.npy        комплексный массив (M, N) — вход этапа A
    meta.json         паспорт: все 16 чисел Geometry плюс PRF и M_aperture
    srez.png          картинка окна, чтобы глазами убедиться, что это не мусор

Порядок функций — порядок выполнения:

    read_params                 паспорт радара из SAR_RDA_Params.mat
    open_rcmc                   открыть большой .mat, найти сам массив
    survey_energy               прореженный обзор: где на записи есть сцена
    choose_window               выбрать окно по дальности и по азимуту
    read_gates                  прочитать только выбранные стробы, все импульсы
    detect_azimuth_domain       данные уже в доплере или ещё во времени
    doppler_centroid            оценка f_dc по самим данным
    azimuth_matched_filter      опорная функция азимутального сжатия
    focus_line                  применить фильтр, получить строку изображения
    cut_and_backtransform       вырезать окно и вернуть его в дальность-доплер
    build_passport              собрать Geometry и всё, что к нему полагается
    main                        всё вместе, с печатью доказательств

ЧТО ЭТОТ СКРИПТ ДЕЛАЕТ И ЧЕГО НЕ ДЕЛАЕТ.

Делает: азимутальное согласованное сжатие по ПРЯМОЛИНЕЙНОЙ модели с одной
скоростью — той, что записана в паспорте. Это ровно тот тракт, который работает
без ИНС: он знает PRF, несущую, номинальную скорость и дальность, и больше
ничего.

Не делает: не компенсирует отклонения носителя от прямой. Их никто и не может
компенсировать без ИНС — именно они и остаются в данных остаточной фазой
phi_k, и именно их MN-MEA обязан найти. Если бы скрипт их снял, тестировать
было бы нечего.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------- настройки

#: Где лежат входные файлы. Значение по умолчанию; первым аргументом
#: командной строки перебивается, менять файл ради другого каталога не нужно.
DATA_DIR = r"E:\Work\__Galogram__\cluade\RSA_MAX"
RCMC_NAME = "SAR_RCMC_Result.mat"
PARAMS_NAME = "SAR_RDA_Params.mat"

#: Куда кладётся срез: подкаталог рядом с данными. Имя одно и то же всегда —
#: run_real.py ждёт именно его.
OUT_NAME = "mn_mea_srez"

#: Половина ширины окна по дальности, в долях R_B0.
#: Geometry в MN-MEA ОДНА на всю сцену, а время апертуры T_a пропорционально
#: дальности. Взяв +-10 %, держим T_a в тех же +-10 % по всему окну. Шире —
#: и одна геометрия перестаёт описывать край окна.
RANGE_HALF_FRACTION = 0.10

#: Прореживание обзорного чтения: каждый SURVEY_STEP-й импульс. Нужно только
#: чтобы найти, где на записи есть сцена, а не шум. 3,5 ГБ читать незачем.
SURVEY_STEP = 128


def read_params(path: str) -> dict:
    """Паспорт радара из SAR_RDA_Params.mat.

    Принимает путь; возвращает словарь с PRF, fc, v, resolution, gamma_win,
    R_vec. Все дальнейшие числа выводятся ИЗ ЭТИХ и больше ниоткуда.
    """
    import scipy.io as sio

    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    v_per_col = np.asarray(d["v_per_col"], dtype=np.float64)
    v_spread = float(v_per_col.max() - v_per_col.min())
    if v_spread > 1e-3:
        print(f"  ВНИМАНИЕ: v_per_col не постоянна, разброс {v_spread:.3e} м/с; "
              "книга требует одну скорость на апертуру, берём среднюю")

    R_vec = np.asarray(d["R_vec"], dtype=np.float64)
    return {
        "PRF": float(np.asarray(d["PRF"])),
        "f_0": float(np.asarray(d["fc"])),
        "v": float(v_per_col.mean()),
        "resolution": float(np.asarray(d["resolution"])),
        "gamma_win": float(np.asarray(d["gamma_win"])),
        "R_vec": R_vec,
    }


def open_rcmc(path: str):
    """Открыть большой .mat и найти в нём сам массив данных.

    Принимает путь; возвращает (файл, датасет, ось_дальности, ось_азимута).

    MATLAB версии 7.3 — это HDF5, и h5py отдаёт массив ТРАНСПОНИРОВАННЫМ
    относительно того, как он выглядит в MATLAB. Поэтому оси определяются не по
    порядку, а по размеру: та, что совпала с длиной R_vec, и есть дальность.
    """
    try:
        import h5py
    except ImportError:
        sys.exit("нужен h5py:  pip install h5py")

    f = h5py.File(path, "r")
    best = None
    for name, obj in f.items():
        if hasattr(obj, "shape") and obj.ndim == 2 and obj.size > 10**6:
            if best is None or obj.size > best[1].size:
                best = (name, obj)
    if best is None:
        f.close()
        sys.exit(f"в {path} не нашлось двумерного массива данных")
    return f, best[1], best[0]


def _as_complex(raw) -> np.ndarray:
    """Комплексный массив MATLAB в HDF5 лежит составным типом с полями
    'real' и 'imag'. Здесь он превращается в обычный complex64."""
    if raw.dtype.names and "real" in raw.dtype.names:
        return (raw["real"].astype(np.float32)
                + 1j * raw["imag"].astype(np.float32)).astype(np.complex64)
    return np.asarray(raw).astype(np.complex64)


def survey_energy(dset, n_range: int, range_axis: int) -> np.ndarray:
    """Прореженный обзор: средняя мощность по каждому стробу дальности.

    Принимает датасет, число стробов и номер оси дальности; возвращает вектор
    длины n_range.

    Читается каждый SURVEY_STEP-й импульс — этого хватает, чтобы увидеть, где
    сцена, а где шум приёмника, и не хватает, чтобы съесть память.
    """
    n_az = dset.shape[1 - range_axis]
    idx = np.arange(0, n_az, SURVEY_STEP)
    if range_axis == 0:
        block = _as_complex(dset[:, idx])
    else:
        block = _as_complex(dset[idx, :])
        block = block.T
    return np.mean(np.abs(block) ** 2, axis=1)


def choose_window(power_by_gate: np.ndarray, R_vec: np.ndarray,
                  T_a_of_R, PRF: float) -> dict:
    """Выбрать окно по дальности и его длину по азимуту.

    Принимает профиль мощности по стробам, ось дальности, функцию T_a(R) и PRF;
    возвращает словарь с границами окна и производными числами.

    Дальность выбирается ПО ДАННЫМ: берётся положение, вокруг которого средняя
    мощность максимальна при ширине +-RANGE_HALF_FRACTION. Так окно попадает на
    сцену, а не на пустой строб, и ни одно число не назначается рукой.

    Длина по азимуту равна апертуре на выбранной дальности: M = T_a * PRF.
    Это же число идёт в паспорт как M_aperture, и тогда PRF = M_aperture / T_a
    сходится с настоящим PRF тождественно.
    """
    n_range = R_vec.size
    r_b = float(np.mean(np.diff(R_vec)))
    # накопленная сумма: средняя мощность любого окна берётся за две операции,
    # а не пересчётом по отсчётам — иначе это 23 000 проходов по массиву
    cumulative = np.concatenate([[0.0], np.cumsum(power_by_gate)])

    best = None
    for centre in range(n_range):
        # ширина окна в стробах зависит от дальности: она задана В ДОЛЯХ R_B0
        half = int(RANGE_HALF_FRACTION * R_vec[centre] / r_b)
        lo, hi = centre - half, centre + half
        if lo < 0 or hi > n_range or hi <= lo:
            continue
        score = (cumulative[hi] - cumulative[lo]) / (hi - lo)
        if best is None or score > best[0]:
            best = (score, centre, lo, hi)
    if best is None:
        sys.exit("окно по дальности не помещается: запись слишком узкая")

    _, centre, lo, hi = best
    R_B0 = float(R_vec[centre])
    T_a = float(T_a_of_R(R_B0))
    return {
        "gate_lo": int(lo),
        "gate_hi": int(hi),
        "gate_centre": int(centre),
        "R_B0": R_B0,
        "T_a": T_a,
        "M_aperture": int(round(T_a * PRF)),
    }


def read_gates(dset, lo: int, hi: int, range_axis: int) -> np.ndarray:
    """Прочитать выбранные стробы целиком по азимуту.

    Принимает датасет, границы по дальности и номер оси дальности;
    возвращает массив (азимут, дальность) в complex64.

    Читается ТОЛЬКО окно по дальности, но ВСЯ длина по азимуту: азимутальный
    согласованный фильтр живёт в доплеровской области всей строки, и обрезать
    её до вырезания окна нельзя.
    """
    if range_axis == 0:
        block = _as_complex(dset[lo:hi, :]).T
    else:
        block = _as_complex(dset[:, lo:hi])
    return np.ascontiguousarray(block)


def detect_azimuth_domain(block: np.ndarray) -> str:
    """Данные по азимуту уже в доплере или ещё во времени.

    Принимает массив (азимут, дальность); возвращает 'doppler' или 'time'.

    Признак — сосредоточенность энергии. Доплеровский спектр занимает полосу
    уже, чем PRF (её задаёт диаграмма антенны), а во временной области сигнал
    размазан по всей строке. Считается доля энергии в самой сильной трети оси,
    для массива как есть и для его ПФ; где сосредоточеннее, та область и есть
    доплеровская. Число печатается, чтобы решение было видно, а не угадано.
    """
    def concentration(a: np.ndarray) -> float:
        p = np.mean(np.abs(a) ** 2, axis=1)
        p = p / p.sum()
        third = max(1, p.size // 3)
        # круговое скользящее окно: спектр может лежать через край
        double = np.concatenate([p, p])
        window = np.convolve(double, np.ones(third), mode="valid")[:p.size]
        return float(window.max())

    as_is = concentration(block)
    after_fft = concentration(np.fft.fft(block, axis=0))
    print(f"  сосредоточенность: как есть {as_is:.3f}, после ПФ {after_fft:.3f}")
    return "doppler" if as_is >= after_fft else "time"


def doppler_centroid(spectrum: np.ndarray, PRF: float) -> float:
    """Оценка доплеровского центроида f_dc по самим данным.

    Принимает азимутальный спектр (доплер, дальность) и PRF; возвращает f_dc
    в герцах.

    Считается круговым первым моментом: ось доплера замкнута, поэтому обычное
    среднее по номеру бина дало бы бессмыслицу на спектре, лежащем через край.
    Берётся аргумент суммы p_k * exp(j 2 pi k / K).
    """
    p = np.mean(np.abs(spectrum) ** 2, axis=1)
    k = np.arange(p.size)
    moment = np.sum(p * np.exp(2j * np.pi * k / p.size))
    frac = np.angle(moment) / (2 * np.pi)
    return float(frac * PRF)


def azimuth_matched_filter(f_a: np.ndarray, R_vec_window: np.ndarray,
                           lambda_: float, v: float, sign: float) -> np.ndarray:
    """Опорная функция азимутального сжатия, прямолинейная модель.

    Принимает ось доплеровских частот (длина K), дальности окна (длина N),
    длину волны, скорость и знак; возвращает массив (K, N).

        D(f_a) = sqrt(1 - (lambda f_a / (2 v))^2)
        H(f_a, R) = exp( sign * j * 4 pi R D(f_a) / lambda )

    Это ТОЧНАЯ, а не квадратичная форма: на широкой полосе доплера квадратичное
    приближение уже врёт. Своя дальность на каждый строб — фильтр меняется
    вдоль N.

    Знак берётся не на веру: main считает картинку с обоими и оставляет ту,
    что даёт меньшую энтропию. Соглашение о знаке в записи неизвестно, и
    угадывать его нельзя.
    """
    ratio = lambda_ * f_a / (2.0 * v)
    ratio = np.clip(ratio, -1.0, 1.0)          # за полосой корень мнимый
    D = np.sqrt(1.0 - ratio**2)
    phase = sign * 4.0 * np.pi * np.outer(D, R_vec_window) / lambda_
    return np.exp(1j * phase).astype(np.complex64)


def focus_line(spectrum: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Применить фильтр и получить строку изображения.

    Принимает спектр (K, N) и опорную функцию (K, N); возвращает изображение
    (K, N) — обратное ПФ по азимуту, как в классическом RDA.
    """
    return np.fft.ifft(spectrum * H, axis=0)


def image_entropy(image: np.ndarray) -> float:
    """Нормированная энтропия (5-6) картинки — мера сфокусированности.

    Принимает комплексное изображение; возвращает одно число. Меньше — резче.
    Считается ровно так же, как в stage_c_iterate.entropy, чтобы числа были
    сравнимы с отчётом MN-MEA.
    """
    P = np.abs(image).astype(np.float64) ** 2
    S_g = float(P.sum())
    P = np.maximum(P, S_g / P.size * 1e-12)
    E_g = -float(np.sum(P * np.log(P)))
    return E_g / S_g + math.log(S_g)


def cut_and_backtransform(image: np.ndarray, M: int) -> tuple[np.ndarray, int]:
    """Вырезать окно по азимуту и вернуть его в область дальность-доплер.

    Принимает строку изображения (K, N) и нужную длину M; возвращает
    (h, начало окна).

        h = ifft(g)   так что   fft(h) = g   тождественно

    Это ровно то соглашение, которого требует (5-3): прямое ПФ от h обязано
    давать картинку. Обратное преобразование здесь — не "расфокусировка", а
    смена представления: ни один отсчёт не теряется, round-trip точен до
    машинного нуля.

    ПОБОЧНОЕ СЛЕДСТВИЕ, которое надо знать. Раз h = ifft(g), а настоящий
    спектр есть fft(g), то номер бина k в h отвечает ОТРИЦАТЕЛЬНОЙ
    доплеровской частоте: h[k] = S[-k]/M. На (5-27) это не влияет, потому что
    eps ~ f_a^2 — функция чётная. На чём-либо линейном по f_a — влияло бы.

    Окно берётся там, где энергия максимальна: смотреть на пустой участок
    записи бессмысленно.
    """
    K = image.shape[0]
    if M >= K:
        M = K
        start = 0
    else:
        power = np.mean(np.abs(image) ** 2, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(power)])
        sums = cumulative[M:] - cumulative[:-M]
        start = int(np.argmax(sums))
    window = image[start:start + M]
    return np.fft.ifft(window, axis=0).astype(np.complex64), start


def build_passport(params: dict, win: dict, r_b: float, y_m_over_R: float) -> dict:
    """Собрать паспорт: Geometry MN-MEA плюс всё, что к нему полагается.

    Принимает паспорт радара, выбранное окно, шаг по дальности и допущение об
    угле места; возвращает словарь, который пойдёт в meta.json.

    Ни одно число здесь не назначено — каждое выведено из семи чисел
    SAR_RDA_Params.mat и выбранного окна. Что вывести не из чего, стоит в
    разделе "допущения" с прямым указанием, что это допущение.
    """
    v, PRF = params["v"], params["PRF"]
    lambda_ = 299_792_458.0 / params["f_0"]
    R_B0 = win["R_B0"]

    # угол места неизвестен без ИНС; вектор на центр сцены строится по
    # допущению y_m / R_B0, см. раздел "допущения"
    y_m = y_m_over_R * R_B0
    z_m = -math.sqrt(max(R_B0**2 - y_m**2, 0.0))

    return {
        "geometry": {
            "v_x0": v, "v_y0": 0.0, "v_z0": 0.0,
            "a_x": 0.0, "a_y": 0.0, "a_z": 0.0,
            "x_m": 0.0, "y_m": y_m, "z_m": z_m,
            "R_B0": R_B0,
            "lambda_": lambda_,
            "f_0": params["f_0"],
            "r_a": v / PRF,
            "r_b": r_b,
            "T_a": win["T_a"],
            "c": 299_792_458.0,
        },
        "prf": PRF,
        "m_aperture": win["M_aperture"],
        # два разных числа, которые в коде MN-MEA сведены в одно r_a, см. §8.1
        # контракта. Шаг идёт в (5-22), разрешение — в (5-20). Разрешение здесь
        # ровно params['resolution']: gamma_win уже учтён в T_a, то есть
        # апертура взята такой, чтобы достичь именно этого разрешения.
        "r_a_shag": v / PRF,
        "r_a_razreshenie": params["resolution"],
    }


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    out_dir = os.path.join(data_dir, OUT_NAME)
    print(f"данные: {data_dir}")
    missing = [n for n in (RCMC_NAME, PARAMS_NAME)
               if not os.path.isfile(os.path.join(data_dir, n))]
    if missing:
        sys.exit(f"в этом каталоге нет: {', '.join(missing)}\n"
                 f"поправьте DATA_DIR в начале файла или дайте путь аргументом")
    os.makedirs(out_dir, exist_ok=True)

    print("паспорт радара")
    params = read_params(os.path.join(data_dir, PARAMS_NAME))
    R_vec = params["R_vec"]
    r_b = float(np.mean(np.diff(R_vec)))
    lambda_ = 299_792_458.0 / params["f_0"]
    v, PRF = params["v"], params["PRF"]
    print(f"  PRF {PRF:.5f} Гц   f_0 {params['f_0']:.4e} Гц   lambda {lambda_:.7f} м")
    print(f"  v {v:.5f} м/с   разрешение {params['resolution']} м   "
          f"gamma_win {params['gamma_win']}")
    print(f"  стробов {R_vec.size}, дальность {R_vec[0]:.1f} … {R_vec[-1]:.1f} м, "
          f"шаг {r_b:.6f} м")

    def T_a_of_R(R: float) -> float:
        # время апертуры, нужное для заданного разрешения: L = gamma*lambda*R/(2*rho)
        return params["gamma_win"] * lambda_ * R / (2.0 * params["resolution"] * v)

    print("\nоткрываем запись")
    f, dset, name = open_rcmc(os.path.join(data_dir, RCMC_NAME))
    range_axis = 0 if dset.shape[0] == R_vec.size else 1
    n_az = dset.shape[1 - range_axis]
    print(f"  массив '{name}' {dset.shape}, ось дальности {range_axis}, "
          f"импульсов {n_az} ({n_az / PRF:.2f} с, {n_az * v / PRF:.0f} м трассы)")

    print("\nобзор: где на записи сцена")
    power_by_gate = survey_energy(dset, R_vec.size, range_axis)
    win = choose_window(power_by_gate, R_vec, T_a_of_R, PRF)
    print(f"  окно по дальности: стробы {win['gate_lo']}…{win['gate_hi']} "
          f"({win['gate_hi'] - win['gate_lo']} шт), R_B0 = {win['R_B0']:.1f} м")
    print(f"  апертура T_a = {win['T_a']:.4f} с = {win['M_aperture']} импульсов")

    print("\nчитаем окно")
    block = read_gates(dset, win["gate_lo"], win["gate_hi"], range_axis)
    f.close()
    print(f"  прочитано {block.shape}, {block.nbytes / 2**20:.0f} МБ")

    domain = detect_azimuth_domain(block)
    print(f"  область по азимуту: {domain}")
    spectrum = block if domain == "doppler" else np.fft.fft(block, axis=0)
    del block

    f_a = np.fft.fftfreq(spectrum.shape[0], d=1.0 / PRF)
    f_dc = doppler_centroid(spectrum, PRF)
    print(f"  доплеровский центроид {f_dc:.2f} Гц ({f_dc / PRF * 100:.1f} % PRF)")

    R_window = R_vec[win["gate_lo"]:win["gate_hi"]]
    print("\nазимутальное сжатие, оба знака опорной функции")
    results = {}
    for sign in (+1.0, -1.0):
        H = azimuth_matched_filter(f_a, R_window, lambda_, v, sign)
        image = focus_line(spectrum, H)
        S = image_entropy(image)
        results[sign] = (S, image)
        print(f"  знак {sign:+.0f}: нормированная энтропия {S:.4f}")
    S_raw = image_entropy(np.fft.ifft(spectrum, axis=0))
    print(f"  без фильтра:      нормированная энтропия {S_raw:.4f}")

    sign = min(results, key=lambda s: results[s][0])
    S_best, image = results[sign]
    print(f"  взят знак {sign:+.0f}; выигрыш против несжатого {S_raw - S_best:+.4f}")
    if S_best >= S_raw:
        print("  ВНИМАНИЕ: сжатие энтропию не улучшило — модель не та, "
              "кусочек подавать в MN-MEA нельзя")

    print("\nвырезаем окно и возвращаем в дальность-доплер")
    h, az_start = cut_and_backtransform(image, win["M_aperture"])
    print(f"  импульсы {az_start}…{az_start + h.shape[0]}, h {h.shape} "
          f"{h.dtype}, {h.nbytes / 2**20:.0f} МБ")
    if h.shape[0] != win["M_aperture"]:
        print(f"  длина массива {h.shape[0]} не равна M_aperture "
              f"{win['M_aperture']} — записи не хватило на целую апертуру. "
              "Это НЕ ошибка: M_aperture и T_a описывают апертуру радара, а "
              "длина массива — сколько отсчётов взято; в паспорт идут оба.")
    check = np.max(np.abs(np.fft.fft(h, axis=0)
                          - image[az_start:az_start + h.shape[0]]))
    scale = float(np.max(np.abs(image)))
    print(f"  сверка fft(h) == g: {check:.3e} при масштабе {scale:.3e} "
          f"({check / scale:.2e} относительно)")

    np.save(os.path.join(out_dir, "h_srez.npy"), h)

    passport = build_passport(params, win, r_b, y_m_over_R=0.9)
    meta = {
        "istochnik": {
            "fail": RCMC_NAME,
            "massiv": name,
            "stadiya": "дальность сжата, RCMC выполнена, азимут сжат "
                       "прямолинейной моделью с одной скоростью",
            "bez_INS": True,
        },
        "srez": {
            "impulsy": [az_start, az_start + int(h.shape[0])],
            "stroby": [win["gate_lo"], win["gate_hi"]],
            "forma": list(h.shape),
            "dlina_massiva_M": int(h.shape[0]),
            "zamechanie": "M — длина массива по азимуту; M_aperture и T_a "
                          "описывают апертуру радара. Это разные числа, и в "
                          "Geometry идёт T_a, а не длина массива",
        },
        "dannye": {
            "fail": "h_srez.npy",
            "dtype": "complex64",
            "oblast": "дальность-доплер",
            "pravilo": "g = numpy.fft.fft(h, axis=0)",
            "raskladka_k": "numpy.fft, ноль в бине 0, НЕ fftshift",
            "znak_dopplera": "бин k отвечает ЧАСТОТЕ -fftfreq(k); "
                             "на (5-27) не влияет, eps ~ f_a^2 чётна",
        },
        **passport,
        "dopushcheniya": [
            "ускорение носителя нулевое: в паспорте записи его нет. "
            "Следствие: A_2 = v^2, mu_3 = 0, этап B вырождается в "
            "прямолинейный полёт и даёт phi^(0) около нуля",
            "y_m / R_B0 = 0.9 (угол места ~26 градусов): высоты носителя в "
            "паспорте нет. Входит только в знаменатель (5-29) через "
            "полуразмер блока по дальности; на нарезку при N_k = 1 не влияет",
            "скорость одним числом на всю апертуру — так требует книга, и "
            "v_per_col в паспорте записи действительно постоянна",
        ],
        "proverki": {
            "vektor_i_dalnost": float(abs(
                passport["geometry"]["x_m"]**2 + passport["geometry"]["y_m"]**2
                + passport["geometry"]["z_m"]**2 - win["R_B0"]**2)),
            "lambda_f0_minus_c": float(abs(
                passport["geometry"]["lambda_"] * params["f_0"] - 299_792_458.0)),
            "m_aperture_delit_T_a": float(win["M_aperture"] / win["T_a"]),
            "prf_pasporta": PRF,
            "fft_h_minus_g_otnositelno": float(check / scale),
            "entropiya_bez_filtra": S_raw,
            "entropiya_posle": S_best,
        },
        "ozhidaemaya_narezka": {
            "m_p_kak_schitaet_kod": 4.0 * params["resolution"]
                                    / (params["gamma_win"] * lambda_),
            "m_p_pravilnyy": 2.0 * params["resolution"]**2
                             / (params["gamma_win"] * lambda_) * 2.0 / (v / PRF),
            "N_k_ozhidaemoe": 1,
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        g = np.fft.fft(h, axis=0)
        A = np.abs(g)
        dB = 20 * np.log10(np.maximum(A, A.max() * 1e-5) / A.max())
        plt.figure(figsize=(10, 7))
        plt.imshow(dB, aspect="auto", cmap="gray", vmin=-40, vmax=0)
        plt.colorbar(label="дБ")
        plt.xlabel("дальность, отсчёты")
        plt.ylabel("азимут, отсчёты")
        plt.title(f"окно среза, R_B0 = {win['R_B0']:.0f} м, "
                  f"нормированная энтропия {S_best:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "srez.png"), dpi=110)
        print(f"  картинка записана")
    except ImportError:
        print("  matplotlib нет, картинку пропустили")

    print(f"\nготово: {out_dir}")
    print("  h_srez.npy   вход этапа A")
    print("  meta.json    паспорт")
    print("  srez.png     на что смотреть глазами")


if __name__ == "__main__":
    main()

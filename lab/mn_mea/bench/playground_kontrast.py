# -*- coding: utf-8 -*-
"""Песочница показа: как из комплексной картинки сделать красивую и контрастную.

Глобальный код не трогает. Берёт комплексное изображение (.npy) и рисует
рядом варианты показа, печатает по каждому числа. Запуск:

    python bench/playground_kontrast.py путь/к/posle.npy [шаг_азимута_м] [шаг_дальности_м] [nabor=rezkie|myagkie]

Набор rezkie (по умолчанию) — без пространственного сглаживания: тон,
S-кривая, нерезкое маскирование, медиана 3x3. Набор myagkie — фильтры
спекла (Ли, усиленный Ли), они мылят текстуру.

Варианты (все — над одним и тем же изображением; сперва многовзгляд x7 по
азимуту на элемент разрешения и выравнивание по дальности, дальше — тон):

  nabor=rezkie (по умолчанию):
    а  дБ P5…P97, как сейчас (уменьшено x7 уже по показанному)
    б  x7 по интенсивности + выравнивание по дальности, дБ P5…P97
    в  б + S-кривая (центр — медиана, наклон 4 дБ)
    г  б + S-кривая, наклон 6 дБ
    д  б + нерезкое маскирование 9x9, вес 0,6
    е  в + нерезкое маскирование 9x9, вес 0,4 — ВЫБРАН владельцем и
       перенесён в bench/run_real.py ключом pokaz=rezko
    ж  б + медиана 3x3 (только выбросы) + S-кривая 4 дБ
    з  б + нормировка большим окном 512x128 вполсилы 0,3 + S-кривая 5 дБ
  nabor=myagkie (фильтры спекла; владелец отверг: «жёстко замылили»):
    а, б — те же
    в  б + Ли 7x7, L = 1,09 (замер ENL, см. ENL_AFTER_ML)
    г  б + Ли 11x11
    д  б + усиленный Ли 9x9
    е  б + усиленный Ли 15x15
    ж  б + x2 по дальности + усиленный Ли 9x9
    з  д, но окно P2…P99,5 и гамма 0,85

Только numpy и PIL: scipy в окружении нет, локальные средние — через
интегральные изображения.
"""
import sys, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AZ_STEP_M, RG_STEP_M = 0.0571, 0.2998
LOOKS_AZ = 7                 # отсчётов на элемент разрешения по азимуту
RANGE_SMOOTH_GATES = 201     # окно сглаживания профиля по дальности
LOCAL_WIN = (128, 32)        # окно локальной нормировки (строки, стробы) после многовзгляда: 51 x 10 м
#: Действительное число взглядов после многовзгляда x7 по азимуту, ЗАМЕР по
#: полю на срезе владельца: ENL = mean^2/var = 1,09 (исходный 0,86; 7x2 —
#: 1,33; 7x4 — 2,06). Семь соседних отсчётов лежат в одном элементе
#: разрешения и коррелированы (автокорреляция 0,97/0,87/0,73/0,58/0,43 на
#: лагах 1…5), так что усреднение семи даёт один взгляд, а не семь. Фильтр
#: Ли с L = 7 принимал спекл за текстуру и не сглаживал; с L = ENL — работает.
ENL_AFTER_ML = 1.09


def box_mean(a, win):
    """Среднее по прямоугольному окну через интегральное изображение."""
    wm, wn = win
    pm, pn = wm // 2, wn // 2
    p = np.pad(a.astype(np.float64), ((pm, wm - pm), (pn, wn - pn)), mode="reflect")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[wm:, wn:] - c[:-wm, wn:] - c[wm:, :-wn] + c[:-wm, :-wn]
    return (s / (wm * wn))[: a.shape[0], : a.shape[1]]


def multilook_az(I, looks):
    M = I.shape[0] // looks * looks
    return I[:M].reshape(M // looks, looks, I.shape[1]).mean(axis=1)


def multilook_rg(I, looks):
    N = I.shape[1] // looks * looks
    return I[:, :N].reshape(I.shape[0], N // looks, looks).mean(axis=2)


def flatten_range(I, gates, limit_db=6.0):
    """Выравнивание по дальности: медианный профиль в дБ, сглаженный, поправка не больше limit_db.

    Средний профиль ловит яркие цели и воду и оставляет вертикальные полосы;
    медиана по азимуту от них свободна. Сглаживание в дБ, а не в мощности.
    """
    prof_db = 10 * np.log10(np.maximum(np.median(I, axis=0), I.max() * 1e-9))
    k = np.ones(gates) / gates
    smooth = np.convolve(np.pad(prof_db, gates // 2, mode="edge"), k, mode="valid")[: prof_db.size]
    corr_db = np.clip(smooth.mean() - smooth, -limit_db, limit_db)
    return I * (10 ** (corr_db / 10))[None, :]


def lee(I, win, looks):
    """Фильтр Ли по интенсивности: I_hat = m + k (I - m), k = max(0, 1 - Cu^2/Ci^2), Cu^2 = 1/L."""
    m = box_mean(I, (win, win))
    v = box_mean(I * I, (win, win)) - m * m
    ci2 = np.maximum(v, 0) / np.maximum(m * m, 1e-30)
    cu2 = 1.0 / looks
    k = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-30), 0.0, 1.0)
    return m + k * (I - m)


def lee_enhanced(I, win, looks):
    """Усиленный Ли (Лопес): однородное — среднее, кромки — как есть, между — плавно."""
    m = box_mean(I, (win, win))
    v = box_mean(I * I, (win, win)) - m * m
    ci = np.sqrt(np.maximum(v, 0)) / np.maximum(m, 1e-30)
    cu = 1.0 / np.sqrt(looks); cmax = np.sqrt(1.0 + 2.0 / looks)
    w = np.exp(-4.0 * (ci - cu) / np.maximum(cmax - ci, 1e-6))
    out = np.where(ci <= cu, m, m + w * (I - m))
    return np.where(ci >= cmax, I, out)


def sigmoid_db(db, centre, slope):
    """S-кривая в дБ: y = 1/(1+exp(-(db-centre)/slope)) — контраст в средних тонах без клиппинга."""
    return 1.0 / (1.0 + np.exp(-(db - centre) / slope))


def unsharp(db, win, amount):
    """Нерезкое маскирование в дБ: db + amount*(db - среднее по окну)."""
    return db + amount * (db - box_mean(db, win))


def median3(a):
    """Медиана 3x3 через сортировку девяти сдвигов — против одиночных выбросов, кромки целы."""
    p = np.pad(a, 1, mode="reflect")
    stack = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
    return np.median(stack, axis=0)


def to_db(I):
    return 10.0 * np.log10(np.maximum(I, I.max() * 1e-9))


def window(x, lo, hi, gamma=1.0):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    y = np.clip((x - a) / max(b - a, 1e-12), 0, 1) ** gamma
    return y


def chisla(y, name):
    """Числа по показанной картинке (0…1): контраст, доля чёрного/белого, энтропия гистограммы."""
    hist, _ = np.histogram(y, bins=256, range=(0, 1)); p = hist / hist.sum(); p = p[p > 0]
    print(f"  {name:38} контраст std/mean {y.std()/max(y.mean(),1e-9):5.2f}   "
          f"чёрных {100*(y<0.02).mean():4.1f} %  белых {100*(y>0.98).mean():4.1f} %   "
          f"энтропия гистограммы {-(p*np.log2(p)).sum():4.2f} бит из 8")


def main():
    path = pathlib.Path(sys.argv[1])
    nums = [float(w) for w in sys.argv[2:] if "=" not in w]
    az = nums[0] if len(nums) > 0 else AZ_STEP_M
    rg = nums[1] if len(nums) > 1 else RG_STEP_M
    g = np.load(path)
    I = np.abs(g).astype(np.float64) ** 2
    print(f"{path.name}: {g.shape}, многовзгляд x{LOOKS_AZ} -> {I.shape[0]//LOOKS_AZ} строк, "
          f"пиксель после {LOOKS_AZ*az:.2f} x {rg:.2f} м")

    I_ml = multilook_az(I, LOOKS_AZ)
    I_fl = flatten_range(I_ml, RANGE_SMOOTH_GATES)
    db_fl = to_db(I_fl)
    local = box_mean(db_fl, LOCAL_WIN)
    db_loc = db_fl - 0.5 * (local - local.mean())          # вполсилы: озеро остаётся тёмным
    nabor = "rezkie"
    for w in sys.argv[2:]:
        if w.startswith("nabor="):
            nabor = w.split("=", 1)[1]
    if nabor == "rezkie":
        # РЕЗКИЕ: никакого пространственного сглаживания, только тон и подчёркивание
        med = np.percentile(db_fl, 50)
        variants = [
            ("а  дБ P5-P97, как сейчас (уменьшено x7)", multilook_az(window(to_db(I), 5, 97), LOOKS_AZ), 1),
            ("б  x7 по интенсивности + выравнивание, дБ P5-P97", window(db_fl, 5, 97), 1),
            ("в  б + S-кривая (центр медиана, наклон 4 дБ)", sigmoid_db(db_fl, med, 4.0), 1),
            ("г  б + S-кривая, наклон 6 дБ", sigmoid_db(db_fl, med, 6.0), 1),
            ("д  б + нерезкое маскирование 9x9, 0,6", window(unsharp(db_fl, (9, 9), 0.6), 3, 99), 1),
            ("е  в + нерезкое маскирование 9x9, 0,4", sigmoid_db(unsharp(db_fl, (9, 9), 0.4), med, 4.0), 1),
            ("ж  б + медиана 3x3 (только выбросы) + S 4 дБ", sigmoid_db(median3(db_fl), med, 4.0), 1),
            ("з  б + нормировка большим окном 512x128 x0,3 + S 5 дБ",
             sigmoid_db(db_fl - 0.3 * (box_mean(db_fl, (512, 128)) - med), med, 5.0), 1),
        ]
        krupno_name = "kontrast_rezkie_krupno.png"; sheet_name = "kontrast_rezkie.png"
    else:
        krupno_name = "kontrast_krupno.png"; sheet_name = "kontrast_varianty.png"
    L = ENL_AFTER_ML
    I_lee7 = lee(I_fl, 7, L)
    I_lee11 = lee(I_fl, 11, L)
    I_enh9 = lee_enhanced(I_fl, 9, L)
    I_enh15 = lee_enhanced(I_fl, 15, L)
    I_72 = multilook_rg(I_fl, 2)
    I_72_enh = lee_enhanced(I_72, 9, 1.33)

    if nabor != "rezkie":
      variants = [
        ("а  дБ P5-P97, как сейчас (уменьшено x7)", multilook_az(window(to_db(I), 5, 97), LOOKS_AZ), 1),
        ("б  x7 + выравнивание по дальности, дБ P5-P97", window(db_fl, 5, 97), 1),
        ("в  б + Ли 7x7, L = 1,09 (замер)", window(to_db(I_lee7), 5, 97), 1),
        ("г  б + Ли 11x11, L = 1,09", window(to_db(I_lee11), 5, 97), 1),
        ("д  б + усиленный Ли 9x9", window(to_db(I_enh9), 5, 97), 1),
        ("е  б + усиленный Ли 15x15", window(to_db(I_enh15), 5, 97), 1),
        ("ж  б + x2 по дальности + усиленный Ли 9x9", window(to_db(I_72_enh), 5, 97), 2),
        ("з  д, но окно P2-P99.5 и гамма 0.85", window(to_db(I_enh9), 2, 99.5, 0.85), 1),
      ]
    print("\nчисла по показанному (0…1), весь кадр:")
    for name, y, _ in variants:
        chisla(y, name)

    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(fp, 15); break
        except OSError:
            pass

    def tile(y, rg_looks, scale):
        """Картинка с квадратными метрами: строка 7*az, столбец rg*rg_looks."""
        img = Image.fromarray((y * 255).astype(np.uint8))
        w = int(round(img.width * rg * rg_looks / (LOOKS_AZ * az) * scale)); h = int(round(img.height * scale))
        return img.resize((max(w, 1), max(h, 1)), Image.LANCZOS)

    # лист 1: весь кадр, два столбца, масштаб 0,6
    tiles = [(name, tile(y, rl, 0.6)) for name, y, rl in variants]
    W = max(t.width for _, t in tiles); H = max(t.height for _, t in tiles); cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("L", (cols * (W + 10), rows * (H + 28)), 40); d = ImageDraw.Draw(sheet)
    for i, (name, t) in enumerate(tiles):
        x, y0 = (i % cols) * (W + 10), (i // cols) * (H + 28)
        sheet.paste(t, (x, y0 + 24)); d.text((x + 4, y0 + 4), name, fill=255, font=font)
    out = path.with_name(sheet_name); sheet.save(out)

    # лист 2: кроп 1:1 (после многовзгляда) — центр кадра 300 строк x 400 стробов, масштаб 1,5
    m0, n0 = I_ml.shape[0] // 2 - 150, I_ml.shape[1] // 2 - 200
    tiles = []
    for name, y, rl in variants:
        c = y[m0:m0 + 300, n0 // rl:(n0 + 400) // rl]
        tiles.append((name, tile(c, rl, 1.5)))
    W = max(t.width for _, t in tiles); H = max(t.height for _, t in tiles); cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet2 = Image.new("L", (cols * (W + 10), rows * (H + 28)), 40); d2 = ImageDraw.Draw(sheet2)
    for i, (name, t) in enumerate(tiles):
        x, y0 = (i % cols) * (W + 10), (i // cols) * (H + 28)
        sheet2.paste(t, (x, y0 + 24)); d2.text((x + 4, y0 + 4), name, fill=255, font=font)
    out2 = path.with_name(krupno_name); sheet2.save(out2)
    print(f"\nлист вариантов (весь кадр): {out}\nкроп 1:1 (центр 120 x 120 м): {out2}")


if __name__ == "__main__":
    main()

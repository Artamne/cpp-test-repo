"""Выбор видеокарты или процессора — ОДНО место на весь проект (§3 задания).

Остальные файлы про cupy/numpy не знают: они получают объект `Backend`
и работают его методами.

Здесь же живут два азимутальных преобразования Фурье, потому что нормировка
прямого и обратного ПФ — это одно согласованное решение, а не два независимых
(см. README, «решение реализации №3»):

    fft_kernel_minus  ядро e^{-j2*pi*k*m/M}   — (5-3),  множитель alpha
    fft_kernel_plus   ядро e^{+j2*pi*k*m/M}   — (5-14), множитель beta = 1/alpha

Счётчик вызовов этих двух методов — единственный способ доказать замером,
что итерация стоит ровно два азимутальных БПФ (§10.6 задания).

Накопление сумм по M*N ячейкам — всегда в двойной точности, независимо от
того, в чём лежит массив (§3 задания): в `E_g = -sum |g|^2 ln|g|^2` десятки
тысяч слагаемых, и одинарная точность там теряет младшие разряды.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field

import numpy as np

#: Множитель прямого ПФ (5-3). Обратный (5-14) берёт beta = 1/alpha.
#: Книга нормировку не указывает (§8 документа); почему пара взаимно обратная
#: и почему alpha при этом не влияет ни на E', ни на E'' — README, решение №3.
FFT_SCALE_ALPHA = 1.0


@dataclass
class Counters:
    """Счётчики вызовов. Нужны для §10.6 и §10.7: стоимость итерации и
    однократность W показываются замером, а не утверждением."""

    fft_kernel_minus: int = 0
    fft_kernel_plus: int = 0
    w_product: int = 0

    @property
    def fft_azimuth(self) -> int:
        """Всего азимутальных БПФ — сумма обоих ядер."""
        return self.fft_kernel_minus + self.fft_kernel_plus

    def snapshot(self) -> tuple[int, int, int]:
        return (self.fft_kernel_minus, self.fft_kernel_plus, self.w_product)


@dataclass
class Backend:
    """Модуль массивов плюс дисциплина точности и счётчики.

    name             — 'numpy' / 'numpy32' / 'cupy' / 'cupy32'
    xp               — numpy или cupy
    complex_dtype    — в чём лежат массивы
    accum_complex    — в чём накапливаются суммы (всегда двойная точность)
    alpha            — множитель (5-3); множитель (5-14) равен 1/alpha
    """

    name: str
    xp: object
    complex_dtype: object
    real_dtype: object
    on_gpu: bool
    alpha: float = FFT_SCALE_ALPHA
    counters: Counters = field(default_factory=Counters)

    accum_complex = np.complex128
    accum_real = np.float64

    def asarray(self, a) -> "object":
        """Кладёт массив на устройство бэкенда и приводит к его точности.
        Комплексный вход -> complex_dtype, вещественный -> real_dtype.

        Массив, УЖЕ лежащий на устройстве, через numpy не прогоняется. На
        видеокарте это не оптимизация, а единственный рабочий путь: cupy
        запрещает неявное превращение в numpy-массив и на np.asarray(a)
        отвечает отказом. Меняется только точность, и без копии, если она
        и так нужная.
        """
        if isinstance(a, self.xp.ndarray):
            dtype = self.complex_dtype if a.dtype.kind == "c" else self.real_dtype
            return a.astype(dtype, copy=False)
        a = np.asarray(a)
        dtype = self.complex_dtype if np.iscomplexobj(a) else self.real_dtype
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a) -> np.ndarray:
        """Возвращает массив в numpy на процессоре — для печати и графиков.

        Снятие с устройства делается только для того, что на устройстве и
        лежит: на видеокарте сюда может прийти и обычный numpy-массив, а у
        него метода .get() нет.
        """
        if self.on_gpu and isinstance(a, self.xp.ndarray):
            a = a.get()
        return np.asarray(a)

    def fft_kernel_minus(self, a):
        """(5-3): g(m,n) = sum_k a(k,n) e^{-j2*pi*k*m/M}, множитель alpha.

        Принимает массив (M, N), преобразует по первому индексу (азимут),
        возвращает массив (M, N). Один вызов = одно азимутальное БПФ.
        """
        self.counters.fft_kernel_minus += 1
        out = self.xp.fft.fft(a, axis=0)
        if self.alpha != 1.0:
            out = out * self.alpha
        return out.astype(self.complex_dtype, copy=False)

    def fft_kernel_plus(self, a):
        """(5-14): G(k,n) = sum_m a(m,n) e^{+j2*pi*k*m/M}, множитель beta = 1/alpha.

        Ядро с противоположным знаком относительно (5-3). numpy.fft.ifft несёт
        собственный множитель 1/M — он здесь снимается умножением на M, чтобы
        метод считал ровно ту сумму, что написана в (5-14).
        Принимает (M, N), возвращает (M, N). Один вызов = одно азимутальное БПФ.
        """
        self.counters.fft_kernel_plus += 1
        m_size = a.shape[0]
        out = self.xp.fft.ifft(a, axis=0) * (m_size / self.alpha)
        return out.astype(self.complex_dtype, copy=False)

    def sum_real(self, a, axis=None):
        """Сумма вещественного массива с накоплением в float64 (§3 задания).
        Результат остаётся float64 — он идёт в E_g, E' и E''."""
        return self.xp.sum(a, axis=axis, dtype=self.accum_real)

    def sum_complex(self, a, axis=None):
        """Сумма комплексного массива с накоплением в complex128 (§3 задания)."""
        return self.xp.sum(a, axis=axis, dtype=self.accum_complex)


def gpu_status() -> tuple[bool, str]:
    """Есть ли на машине РАБОЧАЯ видеокарта и, если нет, почему.

    Возвращает (годится, причина словами).

    Мало проверить, что cupy импортируется, и мало посчитать одно БПФ. cupy
    собирает ядра на лету, каждое своё, и отказ NVRTC приходит на ТОМ ядре,
    которое понадобилось, — а не на первом попавшемся. Поэтому здесь
    прогоняется уменьшенная копия настоящей итерации §5.8: те же действия над
    теми же типами, что и в stage_c_iterate, только на массиве 8 x 4.

    Список действий не выдуман, а снят с самого алгоритма: два БПФ (5-3) и
    (5-14), умножение на комплексную экспоненту с broadcasting, abs, exp, log,
    maximum, clip, where, conj, ones_like, zeros_like, astype, сравнение и
    сумма с накоплением в двойной точности. Добавится действие в этап C —
    добавить его и сюда.
    """
    try:
        xp = importlib.import_module("cupy")
    except Exception as exc:
        return False, f"cupy не установлен или не импортируется: {type(exc).__name__}"
    try:
        _gpu_smoke_test(xp)
    except Exception as exc:
        text = str(exc).strip()
        first_line = next((ln for ln in text.splitlines() if "error" in ln.lower()), "")
        if not first_line:
            first_line = text.splitlines()[0] if text else ""
        return False, (f"cupy импортируется, но не считает — {type(exc).__name__}"
                       + (f": {first_line.strip()[:160]}" if first_line else ""))
    return True, "видеокарта считает"


def _gpu_smoke_test(xp) -> None:
    """Уменьшенная копия итерации §5.8 — всё, что делает этап C, на 8 x 4.
    Ничего не возвращает; если видеокарта не годится, отсюда летит отказ."""
    M, N = 8, 4
    h = xp.asarray(np.arange(M * N, dtype=np.complex128).reshape(M, N) + 1.0)
    phi = xp.asarray(np.linspace(0.0, 1.0, M, dtype=np.float64))

    h_phi = h * xp.exp(1j * phi)[:, None]          # шаг 1
    g = xp.fft.fft(h_phi, axis=0)                  # (5-3)
    P = xp.abs(g) ** 2                             # шаг 3
    P = xp.maximum(P, xp.asarray(1e-30, dtype=P.dtype))
    ln_P = xp.log(P)
    E_g = float(xp.sum(P * ln_P, dtype=np.float64))  # (5-5), накопление float64
    G = xp.fft.ifft((1.0 + ln_P) * g, axis=0) * M    # (5-14)
    W = xp.conj(G) * h_phi                           # W = G* h e^{j phi}
    E_1 = 2.0 * xp.sum(W.imag, axis=1, dtype=np.float64)   # (5-13)
    E_2 = 2.0 * xp.sum(W.real, axis=1, dtype=np.float64)   # (5-19), укороченно

    bad = E_2 <= 0.0                                 # сравнение
    n_bad = float(xp.sum(bad.astype(xp.float64), dtype=np.float64))
    safe = xp.where(bad, xp.ones_like(E_2), E_2)
    step = xp.clip(xp.where(bad, xp.zeros_like(E_1), -E_1 / safe), -1.0, 1.0)
    crit = float(xp.max(xp.abs(xp.exp(1j * (phi + step)) - xp.exp(1j * phi))))
    g32 = g.astype(xp.complex64)                     # одинарная точность тоже
    float(xp.sum(xp.abs(g32) ** 2, dtype=np.float64))
    if not (E_g == E_g and n_bad >= 0.0 and crit >= 0.0):
        raise RuntimeError("видеокарта вернула не-число")


def available_backends() -> list[str]:
    """Какие пути счёта есть на этой машине. Первым — предпочтительный.
    Нужен для §3: сверка двух бэкендов обязательна, а какие именно доступны,
    решается здесь и нигде больше.

    Видеокарта попадает в список, только если она ПРОВЕРЕНА счётом, см.
    gpu_status. Иначе 'auto' выбрал бы путь, который развалится на первом же
    ядре, и отказ пришёл бы не там, где его причина.
    """
    names = []
    if gpu_status()[0]:
        names += ["cupy", "cupy32"]
    names += ["numpy", "numpy32"]
    return names


def get_backend(name: str = "auto", alpha: float = FFT_SCALE_ALPHA) -> Backend:
    """Возвращает бэкенд по имени. 'auto' — видеокарта, если она есть и
    РАБОТАЕТ, иначе процессор. Принимает имя и множитель нормировки alpha,
    возвращает Backend.

    Переменная окружения MN_MEA_BACKEND перебивает 'auto'. Это рубильник на
    случай, когда видеокарта на машине есть, но пользоваться ей нельзя, а
    трогать код не хочется:

        MN_MEA_BACKEND=numpy python3 run.py

    Явно названный в вызове бэкенд не перебивается ничем: если сказано
    get_backend('cupy'), значит нужен именно он, и отказ должен быть виден,
    а не подменён тихой заменой.
    """
    if name == "auto":
        forced = os.environ.get("MN_MEA_BACKEND", "").strip()
        name = forced if forced else available_backends()[0]

    if name.startswith("cupy"):
        xp = importlib.import_module("cupy")
        on_gpu = True
    elif name.startswith("numpy"):
        xp = np
        on_gpu = False
    else:
        raise ValueError(f"неизвестный бэкенд: {name!r}; есть {available_backends()}")

    single = name.endswith("32")
    return Backend(
        name=name,
        xp=xp,
        complex_dtype=xp.complex64 if single else xp.complex128,
        real_dtype=xp.float32 if single else xp.float64,
        on_gpu=on_gpu,
        alpha=alpha,
    )

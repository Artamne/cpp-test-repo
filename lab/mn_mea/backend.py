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

    Мало проверить, что cupy импортируется. Импорт может пройти, а первое же
    ядро — не собраться: у сборки cupy и установленного CUDA Toolkit разные
    наборы заголовков, и NVRTC отказывает уже посреди работы. Поэтому здесь
    считается крошечное БПФ со сравнением — ровно то, на чём стоит (5-3) и
    ровно та связка, на которой отказ и вылезает.
    """
    try:
        xp = importlib.import_module("cupy")
    except Exception as exc:
        return False, f"cupy не установлен или не импортируется: {type(exc).__name__}"
    try:
        out = xp.abs(xp.fft.fft(xp.arange(4, dtype=xp.complex128)))
        bool(xp.all(out >= 0))
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else ""
        return False, (f"cupy импортируется, но не считает — {type(exc).__name__}"
                       + (f": {first_line}" if first_line else ""))
    return True, "видеокарта считает"


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
    """Возвращает бэкенд по имени. 'auto' — видеокарта, если она есть, иначе
    процессор. Принимает имя и множитель нормировки alpha, возвращает Backend."""
    if name == "auto":
        name = available_backends()[0]

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

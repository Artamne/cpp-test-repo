"""Этап D. Сборка изображения из блоков. §6 документа.

Порядок функций — порядок выполнения:

    block_image   (5-3) с найденным phi_hat
    assemble      укладка блоков на свои места по (5-26)

Параметров движения этап D не требует (§0 документа, таблица этапов).
"""

from __future__ import annotations

import numpy as np

from backend import Backend
from stage_a_blocks import Block


def block_image(backend: Backend, h_block, phi_hat):
    """(5-3) с найденным phi_hat: финальное изображение блока.

        g_qk(m,n) = sum_k h_qk(k,n) e^{j phi_hat_k} e^{-j 2 pi k m / M}

    Принимает бэкенд, данные блока h_qk(k,n) и сошедшийся вектор phi_hat
    длины M; возвращает комплексное изображение блока (M, N). M, N здесь —
    размеры БЛОКА, как объявлено в §5 документа, а не сцены.

    Это ровно то же (5-3), что и на шаге 2 итерации, но вне цикла: в
    стоимость итерации (§10.6) оно не входит.
    """
    h_block = backend.asarray(h_block)
    phi = backend.asarray(phi_hat)  # хоть numpy, хоть уже на устройстве
    return backend.fft_kernel_minus(h_block * backend.xp.exp(1j * phi)[:, None])


def assemble(backend: Backend, images: dict[int, object], blocks: list[Block], M: int, N: int):
    """Укладка блоков на свои места в сцене согласно нумерации (5-26).

    Принимает бэкенд, словарь {q_k: изображение блока}, список блоков этапа A
    и размеры сцены; возвращает собранное изображение (M, N).

    Блоки нарезаны из ИЗОБРАЖЕНИЯ сцены плитками m_p x n_p (см. block_grid), и
    сборка — это обратная укладка тех же плиток. Каждая плитка встаёт по
    своим (m_start:m_stop, n_start:n_stop), то есть ровно туда, откуда была
    взята, а её сквозной номер q_k = m_k + M_k(n_k - 1) — это (5-26).

    Размер изображения блока совпадает с его местом на сцене отсчёт в отсчёт:
    (5-3) взято длиной в блок, значит и выдаёт ровно m_b строк. Подрезать или
    дополнять нечего.
    """
    out = backend.xp.zeros((M, N), dtype=backend.complex_dtype)
    for block in blocks:
        out[block.m_start : block.m_stop, block.n_start : block.n_stop] = images[block.q_k]
    return out

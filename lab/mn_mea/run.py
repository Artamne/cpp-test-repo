"""Прогон целиком: все опыты, все графики, печать чисел и отчёт на диск.

    python3 run.py [каталог_вывода]

По умолчанию пишет в ./out: восемь графиков файлами и report.md с теми же
числами, что печатаются на экран. Ничего не открывает в окне.

ВНИМАНИЕ: ./out лежит в репозитории — это отчёт того прогона, что отвечает
коду. Пока возитесь, задавайте СВОЙ каталог аргументом, иначе перезапишете
его и git pull откажет. Уже перезаписали — git checkout -- lab/mn_mea/out/

Снизу — проверка десяти пунктов приёмки §10 задания. Каждый пункт либо
подтверждён замером, либо назван невыполненным; выдумывать зелёную галочку
там, где её нет, запрещено (правило 14).
"""

from __future__ import annotations

import pathlib
import sys

import plots
import backend as B_
import stage_a_blocks as A_
import stage_c_iterate as C
import validate as V


def main(out_dir: pathlib.Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("# MN-MEA: отчёт о прогоне")
    say()
    say(f"Порог останова: книжный mu = {C.MU_DEFAULT:g} (§5.7 документа), "
        f"замеренный mu = {V.MU_MEASURED:g} (см. «Порог останова» ниже).")
    say(f"Предел итераций: {C.MAX_ITERATIONS} = 5 x 8, где 8 — обещанные книгой "
        f"7-8 итераций (§2.3 документа).")
    say(f"Порог под логарифмом: {C.POWER_FLOOR_RELATIVE:g} от средней мощности "
        f"(решение реализации №1).")
    say()

    # ---------------------------------------------------------------- §10.1
    say("## 1. Этап C в одиночку (§7.1, §10.1)")
    say()
    say("Геометрия в опыт не входит вовсе: начальная фаза нулевая, "
        "параметров движения этап C не использует.")
    say()
    main_case = V.experiment_stage_c_alone()
    say(f"* сцена: {main_case['scene']}, ошибка: {main_case['error']}, "
        f"размах {main_case['edge_rad']:g} рад, блок {main_case['M']}x{main_case['N']}")
    say(f"* итераций до останова: **{main_case['iterations']}**, сошлось: "
        f"**{main_case['converged']}** — {main_case['stop_reason']}")
    say(f"* энтропия (5-6) до/после: **{main_case['entropy_before']:.4f} -> "
        f"{main_case['entropy_after']:.4f}**")
    say(f"* остаточная ошибка фазы, СКО после снятия линейной части: "
        f"**{main_case['residual_rms_rad']:.4e} рад** (внесено было "
        f"{main_case['initial_rms_rad']:.4f} рад)")
    say(f"* выигрыш по пику точечной цели: **{main_case['peak_gain_db']:+.2f} дБ**")
    say(f"* ширина отклика по азимуту по -3 дБ: "
        f"**{main_case['width_before']:.2f} -> {main_case['width_after']:.2f}** отсчёта")
    say(f"* порог под логарифмом сработал в доле пикселей: {main_case['floored_fraction']:.3g}")
    say()

    # ---------------------------------------------------------------- §10.2
    say("## 2. Четыре вида вносимой ошибки (§7.3, §10.2)")
    say()
    say("| вид | итераций | сошлось | внесено, рад | остаток, рад | пик, дБ | ширина по -3 дБ |")
    say("|---|---:|---|---:|---:|---:|---|")
    error_rows = V.experiment_error_kinds()
    for row in error_rows:
        say(f"| {row['error']} | {row['iterations']} | {row['converged']} | "
            f"{row['initial_rms_rad']:.3f} | **{row['residual_rms_rad']:.3e}** | "
            f"{row['peak_gain_db']:+.2f} | {row['width_before']:.2f} -> {row['width_after']:.2f} |")
    say()

    # ---------------------------------------------------------------- §10.3
    say("## 3. Виды сцены, в том числе без точечных целей (§7.4, §10.3)")
    say()
    say("| сцена | итераций | сошлось | остаток, рад | S до | S после | бинов с E''<=0 |")
    say("|---|---:|---|---:|---:|---:|---:|")
    scene_rows = V.experiment_scenes()
    for row in scene_rows:
        say(f"| {row['scene']} | {row['iterations']} | {row['converged']} | "
            f"{row['residual_rms_rad']:.3e} | {row['entropy_before']:.3f} | "
            f"{row['entropy_after']:.3f} | {row['frozen_bins_total']} |")
    say()
    clutter = next(r for r in scene_rows if r["scene"] == "clutter_only")
    truth_rows = V.experiment_entropy_versus_truth()
    clutter_truth = [r for r in truth_rows if r["scene"] == "clutter_only"]
    say("**Замер по сцене без единой точечной цели.** Энтропийный критерий на "
        "однородном спекле выраженного минимума не имеет: за весь прогон "
        f"энтропия (5-6) прошла путь {clutter['entropy_before']:.4f} -> "
        f"{clutter['entropy_after']:.4f}, то есть изменилась на "
        f"{clutter['entropy_after'] - clutter['entropy_before']:+.4f} — "
        "при том, что в истинной точке она равна "
        f"{clutter_truth[0]['S_truth']:.4f}. Вся разница между «ничего не делать», "
        "«истина» и «найдено» укладывается в "
        f"{max(abs(r['S_found'] - r['S_truth']) for r in clutter_truth):.4f} единицы энтропии.")
    say()
    say(f"Алгоритм на ней **блуждает**: (5-9) не выполнено за {clutter['iterations']} "
        f"итераций, вторая производная (5-19) оказалась неположительной "
        f"{clutter['frozen_bins_total']} раз (суммарно по бинам и итерациям), "
        f"остаточное СКО {clutter['residual_rms_rad']:.3e} рад — то есть фаза "
        "не найдена. Это не отказ реализации, а свойство критерия, и оно названо замером.")
    say()

    # ------------------------------------------------- энтропия против истины
    say("## 4. Энтропия в найденной точке против энтропии в истинной")
    say()
    say("Алгоритм минимизирует (5-5), а не расстояние до истины. Доля "
        "недобранного падения энтропии: 0 и меньше — сделал не хуже истины, "
        "1 — не сделал ничего.")
    say()
    say("| сцена | ошибка | S старт | S истина | S найдено | недобрано | остаток, рад |")
    say("|---|---|---:|---:|---:|---:|---:|")
    for row in truth_rows:
        say(f"| {row['scene']} | {row['error']} | {row['S_start']:.4f} | "
            f"{row['S_truth']:.4f} | {row['S_found']:.4f} | {row['missed_fraction']:+.3f} | "
            f"{row['residual_rms_rad']:.3e} |")
    say()

    # ---------------------------------------------------------------- §10.4
    say("## 5. Полный прогон A->B->C->D (§10.4)")
    say()
    full = V.experiment_full_run()
    geom = full["geometry"]
    say(f"Геометрия: v = ({geom.v_x0:g}, {geom.v_y0:g}, {geom.v_z0:g}) м/с, "
        f"a = ({geom.a_x:g}, {geom.a_y:g}, {geom.a_z:g}) м/с², "
        f"R_B0 = {geom.R_B0:.1f} м, T_a = {geom.T_a:g} с, "
        f"r_a = {geom.r_a:g} м, r_b = {geom.r_b:g} м.")
    say()
    say(f"* этап A: полуразмеры блока x_p = {full['x_p_m']:.2f} м, y_p = {full['y_p_m']:.2f} м; "
        f"в отсчётах (5-22)/(5-23) m_p = {full['m_p']:.1f}, n_p = {full['n_p']:.1f}")
    say(f"* этап A: (5-24)/(5-25)/(5-21) M_k = **{full['M_k']}**, N_k = **{full['N_k']}**, "
        f"q = **{full['q']}**")
    say(f"* этап B: D_x = {full['D_x']:.4e}, D_y = {full['D_y']:.4e}")
    say(f"* сошлось блоков: **{full['converged_blocks']} из {full['q']}**; "
        f"остаточное СКО по блокам: среднее {full['mean_residual_rms_rad']:.4f} рад, "
        f"худшее {full['worst_residual_rms_rad']:.4f} рад")
    say()
    say("| q_k | m_k | n_k | размер | eta (5-29) | итераций | сошлось | S до | S после | остаток, рад |")
    say("|---:|---:|---:|---|---:|---:|---|---:|---:|---:|")
    for block in full["per_block"]:
        say(f"| {block['q_k']} | {block['m_k']} | {block['n_k']} | "
            f"{block['shape'][0]}x{block['shape'][1]} | {block['eta']:.3f} | "
            f"{block['iterations']} | {block['converged']} | {block['entropy_before']:.4f} | "
            f"{block['entropy_after']:.4f} | {block['residual_rms_rad']:.4f} |")
    say()

    # ---------------------------------------------------------------- §10.5
    say("## 6. Сверка двух бэкендов (§3, §10.5)")
    say()
    gpu_ok, gpu_reason = B_.gpu_status()
    if not gpu_ok:
        say(f"Видеокарта не используется: {gpu_reason}.")
        say()
    backends = V.experiment_backends()
    say(f"Доступные пути счёта: {', '.join(backends['available'])}. "
        f"Видеокарта на этой машине: **{'есть' if backends['gpu_present'] else 'НЕТ'}**.")
    if not backends["gpu_present"]:
        say()
        say("Видеокарты нет, поэтому сверены два доступных пути процессора — "
            "двойная и одинарная точность. Это не подмена сверки: одинарная "
            "точность прямо проверяет требование §3 о накоплении сумм по M·N "
            "ячейкам в двойной точности независимо от того, в чём лежит массив. "
            "Путь видеокарты написан и выбирается в backend.py, но на этой "
            "машине не исполнялся.")
    say()
    for comparison in backends["comparisons"]:
        say(f"* **{comparison['reference']} против {comparison['other']}**: "
            f"max|Δφ| = **{comparison['phi_max_abs_diff_rad']:.3e} рад**, "
            f"СКО Δφ = {comparison['phi_rms_diff_rad']:.3e} рад, "
            f"|ΔS| = {comparison['entropy_abs_diff']:.3e}, "
            f"итераций {comparison['iterations'][0]} против {comparison['iterations'][1]}")
    say()
    say("| бэкенд | видеокарта | мс на итерацию | M x N |")
    say("|---|---|---:|---|")
    for row in V.experiment_timing():
        say(f"| {row['backend']} | {'да' if row['on_gpu'] else 'нет'} | "
            f"{row['ms_per_iteration']:.2f} | {row['M']}x{row['N']} |")
    say()

    # ----------------------------------------------------------- §10.6,§10.7
    say("## 7. Стоимость итерации: счётчиком, а не утверждением (§10.6, §10.7)")
    say()
    cost = V.experiment_iteration_cost()
    say(f"* итераций в прогоне: {cost['iterations']}")
    say(f"* азимутальных БПФ за итерацию, множество замеренных значений: "
        f"**{cost['fft_per_iteration_set']}** — ровно два: {cost['exactly_two_ffts']}")
    say(f"* вычислений W за итерацию, множество замеренных значений: "
        f"**{cost['w_per_iteration_set']}** — ровно одно: {cost['exactly_one_w']}")
    say(f"* всего БПФ, включая итоговое изображение вне цикла: "
        f"{cost['fft_total_incl_final_image']} = 2 x {cost['iterations']} + "
        f"{cost['fft_outside_loop']}")
    say()

    # ------------------------------------------------------- решения §6
    say("## 8. Решения реализации (§6 задания — четыре; пятое и шестое найдены в работе)")
    say()
    say("### Решение №1: нули в логарифме")
    floor_rows = V.experiment_power_floor()
    say()
    say("| сцена | порог | S после | остаток, рад | доля пикселей на пороге |")
    say("|---|---:|---:|---:|---:|")
    for row in floor_rows:
        say(f"| {row['scene']} | {row['floor_relative']:g} | {row['entropy_after']:.4f} | "
            f"{row['residual_rms_rad']:.3e} | {row['floored_fraction']:.3g} |")
    say()
    say("### Решение №2: нулевая или отрицательная вторая производная")
    curvature = V.experiment_curvature_policy()
    frozen = curvature["freeze"]
    say()
    say(f"* политика 'freeze' на сцене без точечных целей: "
        f"бин с E'' <= 0 не обновляется, таких случаев за прогон "
        f"**{frozen['frozen_bins_total']}**, шаг упёрся в предел pi "
        f"{frozen['clipped_bins_total']} раз; прогон закончился так: "
        f"{frozen['stop_reason']}")
    say(f"* политика 'raise' на той же сцене даёт отказ с причиной: "
        f"`{(curvature['refusal_message'] or 'отказа не случилось')[:160]}`")
    say()
    say("Предел шага выбран ЗАМЕРОМ, а не рассуждением. Колонка «критерий = 2» "
        "это подпись предельного цикла: (5-9) для шага d равен 2|sin(d/2)|, "
        "значит при d = pi он равен 2 — максимуму меры, — и упёршийся в такой "
        "предел бин не даёт (5-9) выполниться никогда.")
    say()
    say("| предел шага, рад | сошлось блоков | итераций (ср.) | остаток (ср.), рад | критерий = 2, раз |")
    say("|---:|---|---:|---:|---:|")
    for row in V.experiment_step_limit():
        step_label = "pi (прежнее, НЕВЕРНОЕ)" if row["is_pi"] else f"{row['step_max_rad']:g}"
        mark = " ← взято" if row["step_max_rad"] == C.STEP_MAX_RAD else ""
        say(f"| {step_label}{mark} | {row['converged_blocks']}/{row['blocks']} | "
            f"{row['mean_iterations']:.1f} | {row['mean_residual_rms_rad']:.4f} | "
            f"{row['criterion_pinned_at_two']} |")
    say()
    say("### Решение №5: снятие сдвига блока перед укладкой")
    say()
    say("Энтропия слепа к линейной по k фазе — та лишь СДВИГАЕТ картинку. "
        "Значит каждый блок приходит со своим произвольным сдвигом, и мозаика "
        "(5-26) разъезжается. В книге такого шага нет; решение принято здесь. "
        "Сдвиг берётся целый, поэтому энтропия не меняется вовсе. Блоки с "
        "точечными целями и без них считаются отдельно: на спекле сама мера "
        "сдвига почти бессмысленна.")
    say()
    say("| (5-26) | блоков с целями на своём месте | средний сдвиг | худший | S (среднее) |")
    say("|---|---:|---:|---:|---:|")
    for label, row in V.experiment_image_shift().items():
        say(f"| {label} | {row['on_place_percent']:.1f} % | "
            f"{row['mean_shift']:.2f} отсчёта | {row['worst_shift']} | "
            f"{row['mean_entropy']:.4f} |")
    say()

    say("### Решение №6: окно блока с перехлёстом и нулями")
    say()
    say("Книга режет сцену на плитки встык. Но расфокусированная цель "
        "размазана ШИРЕ своей плитки — без перехлёста её хвосты теряются; а "
        "свёртка в (5-3) круговая — без запаса нулей уехавшее за край "
        "содержимое выходит с другой стороны и накладывается само на себя. "
        "В мозаику при этом идёт только СЕРДЦЕВИНА, ровно m_b строк, так что "
        "укладка (5-26) остаётся точной плиткой без нахлёста.")
    say()
    say("Доли даны в единицах m_b. Остаток фазы падает и дальше, но контраст "
        "при перехлёсте 1,0 УХУДШАЕТСЯ: окно захватывает землю, где поправка "
        "уже другая, — ровно то, против чего сделано разбиение (5-20). "
        "Внесённая в стенде ошибка одна на всю сцену, поэтому остаток этой "
        "цены не видит, а картинка видит. Выбрано по картинке.")
    say()
    say("| перехлёст | нули | длина (5-3) | сошлось | остаток, рад | контраст | пик |")
    say("|---:|---:|---:|---|---:|---:|---:|")
    for row in V.experiment_block_window():
        mark = " ← как в книге" if row["is_book"] else (" ← взято" if row["is_chosen"] else "")
        say(f"| {row['overlap_fraction']:.2f}{mark} | {row['zero_pad_fraction']:.2f} | "
            f"{row['transform_length']} | {row['converged_blocks']:.1f}/12 | "
            f"{row['mean_residual_rms_rad']:.4f} | {row['contrast']:.2f} | {row['peak']:.1f} |")
    say()
    say(f"Цена: длина (5-3) выросла с m_b до "
        f"{1 + 2 * A_.BLOCK_OVERLAP_FRACTION + 2 * A_.BLOCK_ZERO_PAD_FRACTION:g} m_b, "
        f"значит и оба азимутальных БПФ итерации во столько же раз длиннее. "
        f"Число БПФ на итерацию не изменилось — см. раздел 7.")
    say()

    say("### Решение №3: нормировка ПФ")
    normalisation = V.experiment_normalisation()
    say()
    say("| alpha (5-3), beta = 1/alpha | отн. разность E'' | отн. разность шага (5-8) |")
    say("|---:|---:|---:|")
    for row in normalisation["consistent"]:
        say(f"| {row['alpha']:g} | {row['E2_max_abs_rel_diff']:.3e} | "
            f"{row['step_max_abs_rel_diff']:.3e} |")
    say(f"| НЕсогласованная пара alpha·beta = 1/M | "
        f"{normalisation['mismatched_E2_max_abs_rel_diff']:.3e} | "
        f"**{normalisation['mismatched_step_max_abs_rel_diff']:.3e}** |")
    say()
    say("Шаг (5-8) от нормировки НЕ зависит, пока пара взаимно обратна "
        "(alpha·beta = 1) — расхождение на уровне машинного нуля. При "
        "несогласованной паре меняется и E'', и шаг. Это уточняет оговорку §8 "
        "документа: «согласован» означает именно взаимно обратную пару.")
    say()
    say("### Решение №4: предел числа итераций")
    say()
    say("| mu | итераций | сошлось | остаток, рад |")
    say("|---:|---:|---|---:|")
    for row in V.experiment_stop_threshold():
        say(f"| {row['mu']:g} | {row['iterations']} | {row['converged']} | "
            f"{row['residual_rms_rad']:.3e} |")
    say()
    book_mu = next(r for r in V.experiment_stop_threshold() if r["mu"] == 0.1)
    say(f"Книжный порог mu = 0,1 останавливает на {book_mu['iterations']} итерации — "
        f"книга обещает 7-8 (§2.3 документа), и это подтверждается. Остаточная "
        f"ошибка при нём {book_mu['residual_rms_rad']:.3e} рад; дальнейшее "
        "ужесточение порога ниже 1e-3 ничего не улучшает, поэтому в опытах "
        f"взято mu = {V.MU_MEASURED:g} — это выбор по ЗАМЕРУ, а не по вкусу.")
    say()

    # ------------------------------------------------ расхождения и проверки
    say("## 9. Проверки против документа")
    say()
    straight = V.experiment_straight_flight()
    say(f"* **прямолинейный полёт (§2.2)**: A_1 = {straight['A_1']:g} (нуль: "
        f"{straight['A_1_is_zero']}), A_2 = v² : {straight['A_2_equals_v2']}, "
        f"mu_3 = {straight['mu_3']:g} (нуль: {straight['mu_3_is_zero']}). "
        f"Общие формулы (4-83)/(4-84) дали ровно три числа §2.2 и три нуля: "
        f"**{straight['all_match'] and straight['zeros_are_zero']}**, "
        f"наибольшее расхождение {straight['max_abs_diff']:.3e}")
    criterion = V.experiment_block_criterion()
    say(f"* **критерий (5-20)**: по азимуту |ΔR| = {criterion['dR_azimuth_only']:.6f} м, "
        f"по дальности {criterion['dR_range_only']:.6f} м при r_a/2 = "
        f"{criterion['r_a_over_2']:g} м — оба выполнены: "
        f"{criterion['azimuth_ok'] and criterion['range_ok']}. Кубический член "
        f"{criterion['dR_cubic_term']:.3e} м, то есть меньше линейного в "
        f"{criterion['dR_azimuth_only'] / max(criterion['dR_cubic_term'], 1e-300):.0f} раз "
        "— в размер блока он не входит, как и сказано в §3.3 документа.")
    say("* **восстановление k_10, k_20, k_30**: производная по координате против "
        "напечатанного в (4-83)/(4-84)")
    say()
    say("| коэффициент | численно | напечатано | отн. разность |")
    say("|---|---:|---:|---:|")
    for row in V.experiment_linearisation():
        say(f"| {row['coefficient']} | {row['numeric']:+.6e} | {row['printed']:+.6e} | "
            f"{row['relative_difference']:.2e} |")
    say()
    say("Для k_1 совпадение точное. Для k_2 и k_3 книга отбрасывает члены "
        "высшего порядка по A_1, поэтому расхождение есть, и оно названо числом.")
    say()
    ambiguity = V.experiment_ambiguity_basis()
    say(f"* **мера §7.2**: энтропия к постоянной фазе слепа точно "
        f"({ambiguity['constant_phase_entropy_change']:.1e}); к линейной по k — "
        f"лишь при сдвиге на целое число отсчётов "
        f"({ambiguity['integer_shift_entropy_change_max']:.1e}), а при дробном "
        f"сдвиге меняется на "
        f"{', '.join(f'{value:.3f}' for _, value in ambiguity['fractional_shift_entropy_change'])}. "
        "Снятие линии остаётся консервативной поправкой в пользу алгоритма.")
    say(f"* **центрированный номер бина**: на кубической ошибке снятие линии по "
        f"натуральному k = 0…M-1 даёт {ambiguity['cubic_rms_natural_k']:.3f} рад, "
        f"по центрированному — {ambiguity['cubic_rms_centred_k']:.3e} рад. "
        "Ядро (5-3) периодично по k, поэтому представитель обязан быть центрированным.")
    say()
    say("* **член -(q - q_k)/2 в (5-29)** — §8 документа, расхождение пятое. "
        "Реализована рабочая форма основного текста, то есть член оставлен. "
        "Чего он стоит:")
    say()
    say("| (5-29) | eta | max СКО phi(0), рад | нач. расхождение, рад | итераций (ср.) | сошлось блоков |")
    say("|---|---|---:|---:|---:|---|")
    for label, row in V.experiment_eta_term().items():
        say(f"| {label} | [{row['eta_min']:.2f}, {row['eta_max']:.2f}] | "
            f"{row['phi0_rms_max_rad']:.2f} | {row['mean_initial_rms_rad']:.3f} | "
            f"{row['mean_iterations']:.1f} | {row['converged_blocks']}/{row['q']} |")
    say()

    say("* **длина преобразования (5-3)** — расхождение книги СЕДЬМОЕ, в её "
        "собственный список §8 не попавшее. §4.5 и блок-схема §7 называют "
        "phi(0) «вектором длины M», где M — размер сцены; §5, открывая этап C, "
        "объявляет «далее M, N — размеры блока». Взято второе: длина суммы по "
        "k в (5-3) и есть длина преобразования. Чего это стоит:")
    say()
    say("| способ | длина (5-3) | сошлось | остаток, рад | S (5-6) | контраст | пик | энергии вне блока |")
    say("|---|---|---|---:|---:|---:|---:|---:|")
    for label, row in V.experiment_block_transform().items():
        converged_cell = "—" if row["converged_blocks"] is None else f"{row['converged_blocks']}/12"
        say(f"| {label} | {row['transform_length']} | {converged_cell} | "
            f"{row['mean_residual_rms_rad']:.4f} | {row['S']:.4f} | "
            f"{row['contrast']:.2f} | {row['peak']:.1f} | {row['leak_percent']:.1f} % |")
    say()
    say("Блочное преобразование круговое ВНУТРИ плитки, поэтому энергия за "
        "границу блока не выходит вовсе. У сценного она выходит в занулённую "
        "область — на чужую землю, — и попадает в энтропию (5-5) этого блока.")
    say()

    # ---------------------------------------------------------------- §7.5
    say("## 10. Края (§7.5)")
    say()
    say("| случай | итераций | сошлось | остаток, рад |")
    say("|---|---:|---|---:|")
    edge_rows = V.experiment_edge_cases()
    for row in edge_rows:
        if "residual_rms_rad" in row:
            say(f"| {row['case']} | {row['iterations']} | {row['converged']} | "
                f"{row['residual_rms_rad']:.3e} |")
    special = next(r for r in edge_rows if "purely_azimuth" in r)
    say()
    say(f"* **§3.7 документа**: {special['case']}; разбиение чисто азимутальное: "
        f"{special['purely_azimuth']}, номера блоков по (5-26) {special['block_numbers']}")
    first_block, last_block = full["blocks"][0], full["blocks"][-1]
    truncated = [b for b in full["blocks"] if b.shape != first_block.shape]
    say(f"* **блок на границе**: при M = {full['M']}, N = {full['N']}, "
        f"M_k = {full['M_k']}, N_k = {full['N_k']} номинальный блок "
        f"{first_block.shape[0]}x{first_block.shape[1]}, обрезанных блоков "
        f"{len(truncated)} из {full['q']}; самый малый — "
        f"{min(b.shape[0] for b in full['blocks'])}x"
        f"{min(b.shape[1] for b in full['blocks'])} (q_k={last_block.q_k}), "
        f"обработан наравне с остальными: сошёлся "
        f"{next(b['converged'] for b in full['per_block'] if b['q_k'] == last_block.q_k)}")
    say()

    # ------------------------------------------------------------- графики
    say("## Графики")
    say()
    sweep = V.experiment_magnitude_sweep()
    written = [
        plots.plot_entropy_per_iteration(full, out_dir / "01_entropy_per_iteration.png"),
        plots.plot_stop_criterion(full, V.MU_MEASURED, out_dir / "02_stop_criterion.png"),
        plots.plot_phase_true_found(main_case, out_dir / "03_phase_true_found.png"),
        plots.plot_image_before_after(
            full["_image_before"], full["_image_after"],
            "Собранное изображение до и после (полный прогон A->B->C->D)",
            out_dir / "04_image_before_after.png"),
        plots.plot_azimuth_cut(main_case, out_dir / "05_azimuth_cut.png"),
        plots.plot_block_map(full, out_dir / "06_block_map.png"),
        plots.plot_residual_versus_magnitude(sweep, out_dir / "07_residual_vs_magnitude.png"),
        plots.plot_power_floor_effect(floor_rows, out_dir / "08_power_floor_effect.png"),
    ]
    for path in written:
        say(f"* `{path.name}`")
    say()

    # ------------------------------------------------------------- приёмка
    say("## Приёмка §10 задания")
    say()
    checks = [
        ("1. Этап C в одиночку, остаточное СКО названо числом",
         main_case["converged"] and main_case["residual_rms_rad"] < 0.1,
         f"{main_case['residual_rms_rad']:.3e} рад за {main_case['iterations']} итераций"),
        ("2. То же для кубической, дрожания и смеси, таблицей",
         all(r["residual_rms_rad"] < 0.5 for r in error_rows),
         "остатки: " + ", ".join(f"{r['error']} {r['residual_rms_rad']:.2e}" for r in error_rows)),
        ("3. Сцена без точечных целей: сказано замером, что происходит",
         clutter["frozen_bins_total"] > 0 and not clutter["converged"],
         f"блуждает: (5-9) не выполнено, E''<=0 {clutter['frozen_bins_total']} раз, "
         f"энтропия сдвинулась на {clutter['entropy_after'] - clutter['entropy_before']:+.4f}"),
        ("4. Полный прогон A->B->C->D на синтетике",
         full["converged_blocks"] >= 1 and full["q"] > 1,
         f"q = {full['q']} блоков (M_k={full['M_k']}, N_k={full['N_k']}), "
         f"сошлось {full['converged_blocks']}"),
        ("5. Два бэкенда сверены, расхождение названо числом",
         len(backends["comparisons"]) >= 1,
         "; ".join(f"{c['reference']}/{c['other']}: {c['phi_max_abs_diff_rad']:.3e} рад"
                   for c in backends["comparisons"])),
        ("6. Стоимость итерации — ровно два азимутальных БПФ, счётчиком",
         cost["exactly_two_ffts"],
         f"замерено {cost['fft_per_iteration_set']} на каждой из {cost['iterations']} итераций"),
        ("7. W считается один раз за итерацию, тем же счётчиком",
         cost["exactly_one_w"],
         f"замерено {cost['w_per_iteration_set']}"),
        ("8. Четыре решения из §6 записаны в README.md словами",
         (pathlib.Path(__file__).parent / "README.md").exists()
         and "решение реализации №4"
         in (pathlib.Path(__file__).parent / "README.md").read_text(encoding="utf-8").lower(),
         "см. README.md, раздел «Четыре решения реализации»"),
        ("9. Шесть расхождений книги помечены в коде по месту",
         all(_discrepancy_marks().values()),
         "помечены: " + ", ".join(tag for tag, found in _discrepancy_marks().items() if found)
         + (("; НЕ помечены: " + ", ".join(
             tag for tag, found in _discrepancy_marks().items() if not found))
            if not all(_discrepancy_marks().values()) else "")),
        ("10. Владелец находит каждую формулу — проверяет он сам", None,
         "каждая функция называет номер формулы в докстроке; "
         "таблица «какая формула где» — в README.md"),
    ]
    say("| пункт | замер | чем подтверждено |")
    say("|---|---|---|")
    failed = 0
    for title, ok, evidence in checks:
        if ok is None:
            mark = "за владельцем"
        elif ok:
            mark = "да"
        else:
            mark = "**НЕТ**"
            failed += 1
        say(f"| {title} | {mark} | {evidence} |")
    say()
    if failed:
        say(f"**Не выполнено пунктов: {failed}.**")
    else:
        say("Все проверяемые замером пункты выполнены; пункт 10 проверяет владелец.")

    report = out_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nотчёт: {report}")
    print(f"графиков записано: {len(written)} в {out_dir}")
    return 1 if failed else 0


#: Шесть расхождений книги с собой, перечисленных в §8 документа. Каждое
#: обязано быть помечено в коде ПО МЕСТУ (§10.9 задания).
BOOK_DISCREPANCIES = (
    "(5-9)",       # без модуля
    "(5-13)",      # знак перед двойкой
    "(5-20)/модуль",
    "(5-20)/t_a",
    "(5-23)",      # n_p через Delta R
    "(5-29)",      # член -(q - q_k)/2
    "(5-19)",      # коэффициент (2 + ln|g|^2)
)


def _discrepancy_marks() -> dict[str, bool]:
    """§10.9 задания: какие из расхождений §8 документа помечены в коде.

    Возвращает словарь {метка: найдена ли}. Ищется строка вида
    «РАСХОЖДЕНИЕ <метка>: ... см. §8» в файлах этапов.
    """
    here = pathlib.Path(__file__).parent
    text = "".join(
        (here / name).read_text(encoding="utf-8")
        for name in ("stage_a_blocks.py", "stage_b_initial.py",
                     "stage_c_iterate.py", "stage_d_assemble.py")
    )
    return {tag: f"РАСХОЖДЕНИЕ {tag}" in text and "см. §8" in text
            for tag in BOOK_DISCREPANCIES}


def _count_discrepancy_marks() -> int:
    """Сколько из расхождений §8 помечено в коде по месту."""
    return sum(_discrepancy_marks().values())


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "out"
    raise SystemExit(main(target))

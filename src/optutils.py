#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for optimizing recommendation parameters.
"""

from itertools import product
import logging
import time

import numpy as np
import pandas as pd

import sfutils as sutil


logger = logging.getLogger(__name__)


###################################
def format_duration(seconds):
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


###################################
def _get_event_attendance_type_weights(
    attendance_type_weights,
    use_default_event_attendance_weights,
    default_attendance_type_weights,
):
    if use_default_event_attendance_weights:
        if default_attendance_type_weights is None:
            default_attendance_type_weights = {
                'default': 0,
                'scheduled': 1,
                'waitlisted': 2,
                'attended': 3,
                'bookmark': 2,
            }
        return default_attendance_type_weights.copy()

    return default_attendance_type_weights.copy()


###################################
def _evaluate_attendance_type_weights(
    sample_users_id,
    past_attendance,
    event_attendance,
    events,
    event_session_similarity,
    today,
    attendance_type_weights,
    result_context,
    ret_recs_size,
    t_half,
    user_sessions_filter,
    recs_top_n,
    recs_top_n_list,
    use_default_event_attendance_weights,
    default_attendance_type_weights,
):
    past_attendance_test = past_attendance.copy()
    event_attendance_test = event_attendance.copy()
    event_attendance_type_weights = _get_event_attendance_type_weights(
        attendance_type_weights,
        use_default_event_attendance_weights,
        default_attendance_type_weights,
    )
    recs_params = {
        'time_half_decay': t_half,
        'attendance_type_weights': attendance_type_weights,
        'ret_recs_size': ret_recs_size,
    }
    event_recs_params = {
        **recs_params,
        'attendance_type_weights': event_attendance_type_weights,
    }

    past_attendance_test['rating'] = sutil.calc_rating(
        past_attendance_test,
        events,
        today=today,
        recs_params=recs_params,
    )
    event_attendance_test['rating'] = sutil.calc_rating(
        event_attendance_test,
        events,
        today=today,
        recs_params=event_recs_params,
    )

    ret_recs = sutil.calc_users_recs(
        sample_users_id,
        past_attendance_test,
        event_session_similarity,
        user_sessions_filter,
        events=None,
        recs_params=recs_params,
    )

    attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = sutil.calc_recs_stats(
        ret_recs,
        event_attendance_test,
        recs_top_n=recs_top_n,
        recs_top_n_list=recs_top_n_list,
    )
    ndcg = sutil.calc_ndcg(ret_recs, event_attendance_test, recs_top_n=recs_top_n)

    attendee_top = attendee_hit_ratio.loc[attendee_hit_ratio['top_n_recs'] == recs_top_n]
    session_top = session_hit_ratio.loc[session_hit_ratio['top_n'] == recs_top_n]

    result_row = {
        **result_context,
        'ndcg': ndcg,
        'recs_top_n': recs_top_n,
        'ret_recs_size': len(ret_recs),
        'attendee_hit_pct': attendee_top['attendees_pct'].iloc[0] if not attendee_top.empty else np.nan,
        'session_recs_size': session_top['recs_size'].iloc[0] if not session_top.empty else np.nan,
        'session_recs_hit': session_top['recs_hit'].iloc[0] if not session_top.empty else np.nan,
        'session_recs_hit_ratio': session_top['recs_hit_ratio'].iloc[0] if not session_top.empty else np.nan,
        'session_recs_hit_rating_sum': session_top['recs_hit_rating_sum'].iloc[0] if not session_top.empty else np.nan,
        'session_recs_hit_rating_avg': session_top['recs_hit_rating_avg'].iloc[0] if not session_top.empty else np.nan,
        'attendance_type_weights': attendance_type_weights.copy(),
        'event_attendance_type_weights': event_attendance_type_weights.copy(),
        'use_default_event_attendance_weights': use_default_event_attendance_weights,
    }

    attendee_hit_ratio = attendee_hit_ratio.assign(**result_context, ndcg=ndcg)
    attendee_hit_cnt = attendee_hit_cnt.assign(**result_context, ndcg=ndcg)
    session_hit_ratio = session_hit_ratio.assign(**result_context, ndcg=ndcg)

    return result_row, attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio


###################################
def _build_optimizer_result(results, attendee_hit_ratio_results, attendee_hit_cnt_results, session_hit_ratio_results):
    results = pd.DataFrame(results)
    if not results.empty:
        results = results.sort_values('ndcg', ascending=False).reset_index(drop=True)

    attendee_hit_ratio_results = pd.concat(attendee_hit_ratio_results, ignore_index=True) if attendee_hit_ratio_results else pd.DataFrame()
    attendee_hit_cnt_results = pd.concat(attendee_hit_cnt_results, ignore_index=True) if attendee_hit_cnt_results else pd.DataFrame()
    session_hit_ratio_results = pd.concat(session_hit_ratio_results, ignore_index=True) if session_hit_ratio_results else pd.DataFrame()

    return {
        'results': results,
        'attendee_hit_ratio': attendee_hit_ratio_results,
        'attendee_hit_cnt': attendee_hit_cnt_results,
        'session_hit_ratio': session_hit_ratio_results,
    }


###################################
def optimize_attendance_type_weights(
    sample_users_id,
    past_attendance,
    event_attendance,
    events,
    event_session_similarity,
    event_date,
    ret_recs_size=30,
    t_half=365,
    user_sessions_filter=None,
    recs_top_n=15,
    recs_top_n_list=None,
    weight_min=1,
    weight_max=5,
    weight_values=None,
    weight_keys=('scheduled', 'waitlisted', 'attended', 'bookmark'),
    fixed_weights=None,
    use_default_event_attendance_weights=False,
    default_attendance_type_weights=None,
    progress_every=100,
):
    if recs_top_n_list is None:
        recs_top_n_list = [3, 5, 10, 15, 20, 25, 30]
    recs_top_n_list = list(dict.fromkeys(list(recs_top_n_list) + [recs_top_n]))

    base_weights = {
        'default': 0,
        'scheduled': 1,
        'waitlisted': 2,
        'attended': 3,
        'bookmark': 2,
    }
    if fixed_weights is not None:
        base_weights.update(fixed_weights)

    if weight_values is None:
        weight_values = range(weight_min, weight_max + 1)

    weight_values = list(weight_values)
    weight_keys = list(weight_keys)
    today = pd.Timestamp(event_date).normalize()

    results = []
    attendee_hit_ratio_results = []
    attendee_hit_cnt_results = []
    session_hit_ratio_results = []
    total_runs = len(weight_values) ** len(weight_keys)
    start_time = time.monotonic()

    for run_id, weight_tuple in enumerate(product(weight_values, repeat=len(weight_keys)), start=1):
        attendance_type_weights = base_weights.copy()
        attendance_type_weights.update(dict(zip(weight_keys, weight_tuple)))
        weight_cols = {f"weight_{key}": attendance_type_weights[key] for key in base_weights}
        result_context = {
            'run_id': run_id,
            **weight_cols,
        }

        if progress_every and (run_id == 1 or run_id % progress_every == 0 or run_id == total_runs):
            logger.info(
                "Optimizing attendance weights run %s/%s: %s",
                run_id,
                total_runs,
                attendance_type_weights,
            )
            logger.info(
                "Event attendance weights: %s",
                _get_event_attendance_type_weights(
                    attendance_type_weights,
                    use_default_event_attendance_weights,
                    default_attendance_type_weights,
                ),
            )
            completed_runs = run_id - 1
            if completed_runs:
                elapsed_seconds = time.monotonic() - start_time
                seconds_per_run = elapsed_seconds / completed_runs
                remaining_seconds = seconds_per_run * (total_runs - completed_runs)
                logger.info("Estimated time remaining: %s", format_duration(remaining_seconds))
            else:
                logger.info("Estimated time remaining: available after first completed run")

        result_row, attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = _evaluate_attendance_type_weights(
            sample_users_id,
            past_attendance,
            event_attendance,
            events,
            event_session_similarity,
            today,
            attendance_type_weights,
            result_context,
            ret_recs_size,
            t_half,
            user_sessions_filter,
            recs_top_n,
            recs_top_n_list,
            use_default_event_attendance_weights,
            default_attendance_type_weights,
        )

        results.append(result_row)
        attendee_hit_ratio_results.append(attendee_hit_ratio)
        attendee_hit_cnt_results.append(attendee_hit_cnt)
        session_hit_ratio_results.append(session_hit_ratio)

    return _build_optimizer_result(
        results,
        attendee_hit_ratio_results,
        attendee_hit_cnt_results,
        session_hit_ratio_results,
    )


###################################
def optimize_attendance_type_weights_genetic(
    sample_users_id,
    past_attendance,
    event_attendance,
    events,
    event_session_similarity,
    event_date,
    ret_recs_size=30,
    t_half=365,
    user_sessions_filter=None,
    recs_top_n=15,
    recs_top_n_list=None,
    weight_min=1,
    weight_max=5,
    weight_values=None,
    weight_keys=('scheduled', 'waitlisted', 'attended', 'bookmark'),
    fixed_weights=None,
    population_size=10,
    epochs=10,
    elite_size=2,
    mutation_rate=0.2,
    tournament_size=3,
    random_state=None,
    use_default_event_attendance_weights=False,
    default_attendance_type_weights=None,
    progress_every=1,
):
    if population_size <= 0:
        raise ValueError("population_size must be greater than 0")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if elite_size <= 0:
        raise ValueError("elite_size must be greater than 0")
    if mutation_rate < 0 or mutation_rate > 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be greater than 0")

    if recs_top_n_list is None:
        recs_top_n_list = [3, 5, 10, 15, 20, 25, 30]
    recs_top_n_list = list(dict.fromkeys(list(recs_top_n_list) + [recs_top_n]))

    base_weights = {
        'default': 0,
        'scheduled': 1,
        'waitlisted': 2,
        'attended': 3,
        'bookmark': 2,
    }
    if fixed_weights is not None:
        base_weights.update(fixed_weights)

    if weight_values is None:
        weight_values = range(weight_min, weight_max + 1)

    weight_values = [int(value) for value in weight_values]
    if not weight_values:
        raise ValueError("weight_values must contain at least one value")

    weight_keys = list(weight_keys)
    today = pd.Timestamp(event_date).normalize()
    rng = np.random.default_rng(random_state)
    elite_size = min(elite_size, population_size)
    tournament_size = min(tournament_size, population_size)

    results = []
    attendee_hit_ratio_results = []
    attendee_hit_cnt_results = []
    session_hit_ratio_results = []
    total_runs = population_size * epochs
    start_time = time.monotonic()
    run_id = 0
    search_space_size = len(weight_values) ** len(weight_keys)

    def random_individual():
        if not weight_keys:
            return ()
        return tuple(int(value) for value in rng.choice(weight_values, size=len(weight_keys), replace=True))

    def initial_population():
        population = []
        seen = set()
        unique_target = min(population_size, search_space_size)

        while len(population) < unique_target:
            individual = random_individual()
            if individual in seen:
                continue
            population.append(individual)
            seen.add(individual)

        while len(population) < population_size:
            population.append(random_individual())

        return population

    def weights_from_individual(individual):
        attendance_type_weights = base_weights.copy()
        attendance_type_weights.update(dict(zip(weight_keys, individual)))
        return attendance_type_weights

    def select_parent(epoch_results):
        selected_idx = rng.choice(len(epoch_results), size=tournament_size, replace=True)
        selected = [epoch_results[idx] for idx in selected_idx]
        selected = sorted(selected, key=lambda item: item['ndcg'], reverse=True)
        return selected[0]['individual']

    def crossover(parent_a, parent_b):
        if not weight_keys:
            return ()
        return tuple(
            parent_a[idx] if rng.random() < 0.5 else parent_b[idx]
            for idx in range(len(weight_keys))
        )

    def mutate(individual):
        if not weight_keys:
            return ()
        return tuple(
            int(rng.choice(weight_values)) if rng.random() < mutation_rate else int(value)
            for value in individual
        )

    def make_next_population(epoch_results):
        epoch_results = sorted(epoch_results, key=lambda item: item['ndcg'], reverse=True)
        next_population = [item['individual'] for item in epoch_results[:elite_size]]
        seen = set(next_population)
        attempts = 0

        while len(next_population) < population_size:
            parent_a = select_parent(epoch_results)
            parent_b = select_parent(epoch_results)
            child = mutate(crossover(parent_a, parent_b))

            if len(seen) < search_space_size and child in seen and attempts < population_size * 20:
                attempts += 1
                continue

            next_population.append(child)
            seen.add(child)

        return next_population

    population = initial_population()

    for epoch in range(1, epochs + 1):
        epoch_results = []

        for candidate_id, individual in enumerate(population, start=1):
            run_id += 1
            attendance_type_weights = weights_from_individual(individual)
            weight_cols = {f"weight_{key}": attendance_type_weights[key] for key in base_weights}
            result_context = {
                'run_id': run_id,
                'epoch': epoch,
                'candidate_id': candidate_id,
                **weight_cols,
            }

            if progress_every and (run_id == 1 or run_id % progress_every == 0 or run_id == total_runs):
                logger.info(
                    "Optimizing attendance weights genetic run %s/%s epoch %s/%s candidate %s/%s: %s",
                    run_id,
                    total_runs,
                    epoch,
                    epochs,
                    candidate_id,
                    population_size,
                    attendance_type_weights,
                )
                logger.info(
                    "Event attendance weights: %s",
                    _get_event_attendance_type_weights(
                        attendance_type_weights,
                        use_default_event_attendance_weights,
                        default_attendance_type_weights,
                    ),
                )
                completed_runs = run_id - 1
                if completed_runs:
                    elapsed_seconds = time.monotonic() - start_time
                    seconds_per_run = elapsed_seconds / completed_runs
                    remaining_seconds = seconds_per_run * (total_runs - completed_runs)
                    logger.info("Estimated time remaining: %s", format_duration(remaining_seconds))
                else:
                    logger.info("Estimated time remaining: available after first completed run")

            result_row, attendee_hit_ratio, attendee_hit_cnt, session_hit_ratio = _evaluate_attendance_type_weights(
                sample_users_id,
                past_attendance,
                event_attendance,
                events,
                event_session_similarity,
                today,
                attendance_type_weights,
                result_context,
                ret_recs_size,
                t_half,
                user_sessions_filter,
                recs_top_n,
                recs_top_n_list,
                use_default_event_attendance_weights,
                default_attendance_type_weights,
            )

            results.append(result_row)
            attendee_hit_ratio_results.append(attendee_hit_ratio)
            attendee_hit_cnt_results.append(attendee_hit_cnt)
            session_hit_ratio_results.append(session_hit_ratio)
            epoch_results.append({
                'individual': individual,
                'ndcg': result_row['ndcg'],
            })

        if epoch < epochs:
            population = make_next_population(epoch_results)

    return _build_optimizer_result(
        results,
        attendee_hit_ratio_results,
        attendee_hit_cnt_results,
        session_hit_ratio_results,
    )

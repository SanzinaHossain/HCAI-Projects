import random
import time
from pathlib import Path

from django.http import FileResponse
from django.shortcuts import redirect, render

from .services.data_service import load_movies, movie_card
from .services.feature_service import extract_features
from .services.preference_model import (
    fit_bradley_terry,
    fit_plackett_luce,
    pairwise_accuracy,
    recommend,
)
from .services.study_service import (
    PAIRWISE_TASKS,
    RANKING_TASKS,
    VALIDATION_TASKS,
    build_plan,
)


def landing(request):
    try:
        count = len(load_movies())
        err = None
    except Exception as exc:
        count = None
        err = str(exc)
    return render(
        request,
        'project4/landing.html',
        {'movie_count': count, 'dataset_error': err},
    )


def start(request):
    try:
        n_movies = len(load_movies())
        plan = build_plan(n_movies, random.randrange(1_000_000_000))
    except Exception:
        return redirect('project4:landing')

    request.session['p4'] = {
        'plan': plan,
        'consented': False,
        'condition_pos': 0,
        'task_pos': 0,
        'pairwise': [],
        'rankings': [],
        'surveys': {},
        'validation': [],
        'timings': {},
        'condition_started': None,
        'started': time.time(),
    }
    return redirect('project4:consent')


def consent(request):
    study = request.session.get('p4')
    if not study:
        return redirect('project4:start')

    if request.method == 'POST':
        study['consented'] = True
        study['condition_started'] = time.time()
        request.session.modified = True
        return redirect('project4:task')

    return render(request, 'project4/consent.html')


def _cards(ids):
    df = load_movies()
    return [movie_card(df.iloc[i], i) for i in ids]


def _task_context(study, kind, ids, pos, total, error=None):
    return {
        'kind': kind,
        'movies': _cards(ids),
        'index': pos + 1,
        'total': total,
        'method_number': study['condition_pos'] + 1,
        'progress_pct': round(((pos + 1) / total) * 100),
        'error': error,
    }


def task(request):
    study = request.session.get('p4')
    if not study or not study.get('consented'):
        return redirect('project4:start')

    kind = study['plan']['order'][study['condition_pos']]
    pos = study['task_pos']
    total = PAIRWISE_TASKS if kind == 'pairwise' else RANKING_TASKS

    if pos >= total:
        return redirect('project4:survey')

    ids = study['plan']['pairs'][pos] if kind == 'pairwise' else study['plan']['rankings'][pos]

    if request.method == 'POST':
        if kind == 'pairwise':
            raw_winner = request.POST.get('winner')
            try:
                winner = int(raw_winner)
            except (TypeError, ValueError):
                winner = None

            if winner not in ids:
                return render(
                    request,
                    'project4/task.html',
                    _task_context(study, kind, ids, pos, total, 'Please choose one of the two movies.'),
                )

            loser = ids[1] if winner == ids[0] else ids[0]
            study['pairwise'].append([winner, loser])
        else:
            try:
                order = [int(x) for x in request.POST.get('ranking', '').split(',') if x.strip()]
            except ValueError:
                order = []

            if len(order) != len(ids) or sorted(order) != sorted(ids):
                return render(
                    request,
                    'project4/task.html',
                    _task_context(
                        study,
                        kind,
                        ids,
                        pos,
                        total,
                        'Please rank all ten movies exactly once before continuing.',
                    ),
                )
            study['rankings'].append(order)

        study['task_pos'] += 1

        # Record elicitation time immediately after the final task in a method.
        if study['task_pos'] >= total and kind not in study['timings']:
            started = study.get('condition_started')
            if started is not None:
                study['timings'][kind] = max(0.0, time.time() - started)

        request.session.modified = True
        return redirect('project4:task')

    return render(
        request,
        'project4/task.html',
        _task_context(study, kind, ids, pos, total),
    )


def survey(request):
    study = request.session.get('p4')
    if not study:
        return redirect('project4:start')

    kind = study['plan']['order'][study['condition_pos']]

    if request.method == 'POST':
        ease = request.POST.get('ease')
        confidence = request.POST.get('confidence')
        if ease not in {'1', '2', '3', '4', '5'} or confidence not in {'1', '2', '3', '4', '5'}:
            return render(
                request,
                'project4/survey.html',
                {
                    'kind': kind,
                    'method_number': study['condition_pos'] + 1,
                    'method_time': study.get('timings', {}).get(kind),
                    'error': 'Please answer both rating questions.',
                },
            )

        study['surveys'][kind] = {
            'ease': ease,
            'confidence': confidence,
            'comment': request.POST.get('comment', '').strip(),
        }

        if study['condition_pos'] == 0:
            study['condition_pos'] = 1
            study['task_pos'] = 0
            study['condition_started'] = time.time()
            request.session.modified = True
            return redirect('project4:task')

        request.session.modified = True
        return redirect('project4:validation')

    return render(
        request,
        'project4/survey.html',
        {
            'kind': kind,
            'method_number': study['condition_pos'] + 1,
            'method_time': study.get('timings', {}).get(kind),
        },
    )


def validation(request):
    study = request.session.get('p4')
    if not study:
        return redirect('project4:start')

    pos = len(study['validation'])
    if pos >= VALIDATION_TASKS:
        return redirect('project4:complete')

    ids = study['plan']['validation'][pos]
    context = {
        'movies': _cards(ids),
        'index': pos + 1,
        'total': VALIDATION_TASKS,
        'progress_pct': round(((pos + 1) / VALIDATION_TASKS) * 100),
    }

    if request.method == 'POST':
        try:
            winner = int(request.POST.get('winner'))
        except (TypeError, ValueError):
            winner = None

        if winner not in ids:
            context['error'] = 'Please choose one of the two movies.'
            return render(request, 'project4/validation.html', context)

        loser = ids[1] if winner == ids[0] else ids[0]
        study['validation'].append([winner, loser])
        request.session.modified = True
        return redirect('project4:validation')

    return render(request, 'project4/validation.html', context)


def complete(request):
    study = request.session.get('p4')
    if not study or len(study.get('validation', [])) < VALIDATION_TASKS:
        return redirect('project4:start')

    df = load_movies()
    X, _ = extract_features(df)

    pair_w = fit_bradley_terry(X, study['pairwise'])
    rank_w = fit_plackett_luce(X, study['rankings'])
    pair_acc = pairwise_accuracy(pair_w, X, study['validation'])
    rank_acc = pairwise_accuracy(rank_w, X, study['validation'])

    seen = set(
        sum(study['plan']['pairs'], [])
        + sum(study['plan']['rankings'], [])
        + sum(study['plan']['validation'], [])
    )
    recs = [
        movie_card(df.iloc[i], i)
        for i, _ in recommend((pair_w + rank_w) / 2, X, df, seen, 5)
    ]

    names = {
        'pairwise': 'Two-movie choices',
        'ranking': 'Ten-movie ranking',
    }
    order_display = ' → '.join(names[x] for x in study['plan']['order'])

    return render(
        request,
        'project4/complete.html',
        {
            'pair_acc': pair_acc,
            'rank_acc': rank_acc,
            'recs': recs,
            'order_display': order_display,
            'pair_time': study.get('timings', {}).get('pairwise'),
            'rank_time': study.get('timings', {}).get('ranking'),
        },
    )


def reset(request):
    request.session.pop('p4', None)
    return redirect('project4:landing')


def report_pdf(request):
    """Download the submitted Project 4 report as a static PDF.

    The report is intentionally not generated from survey responses or Python
    report-building code. Replacing HCAI_Project_4_Report.pdf replaces the
    report downloaded by every user.
    """
    report_path = Path(__file__).resolve().parent / 'HCAI_Project_4_Report.pdf'
    return FileResponse(
        report_path.open('rb'),
        as_attachment=True,
        filename='HCAI_Project_4_Report.pdf',
        content_type='application/pdf',
    )

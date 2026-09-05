import random

# Both elicitation methods expose 20 unique training movies:
# 10 pairwise tasks x 2 movies and 2 rankings x 10 movies.
PAIRWISE_TASKS = 10
RANKING_TASKS = 2
RANKING_SIZE = 10
VALIDATION_TASKS = 8


def build_plan(n_movies, seed=None):
    rng = random.Random(seed)
    needed = PAIRWISE_TASKS * 2 + RANKING_TASKS * RANKING_SIZE + VALIDATION_TASKS * 2
    if n_movies < needed:
        raise ValueError(
            f'The study needs at least {needed} unique movies, but only {n_movies} are available.'
        )

    # Sampling without replacement means the two training methods and the
    # validation set do not accidentally reuse a movie in the same session.
    ids = rng.sample(range(n_movies), needed)
    p = 0

    pairs = []
    for _ in range(PAIRWISE_TASKS):
        pairs.append(ids[p:p + 2])
        p += 2

    rankings = []
    for _ in range(RANKING_TASKS):
        rankings.append(ids[p:p + RANKING_SIZE])
        p += RANKING_SIZE

    validation = []
    for _ in range(VALIDATION_TASKS):
        validation.append(ids[p:p + 2])
        p += 2

    # Random order is used in the demonstration. In a real deployment, the
    # recruitment protocol described in the report would enforce equal AB/BA
    # allocation across the final sample.
    order = rng.choice([['pairwise', 'ranking'], ['ranking', 'pairwise']])
    return {
        'order': order,
        'pairs': pairs,
        'rankings': rankings,
        'validation': validation,
    }

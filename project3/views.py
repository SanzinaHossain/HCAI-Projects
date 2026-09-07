def index(request):

    selected_category = request.GET.get("category", "All")
    selected_news = request.GET.get("news", "0")

    # --------------------------------------------------------
    # OVERALL ACCURACY
    # --------------------------------------------------------

    classifier_accuracy = round(
        _classifier_accuracy * 100,
        2
    )

    expert_accuracy = round(
        _expert_accuracy * 100,
        2
    )

    # --------------------------------------------------------
    # CATEGORY ACCURACY
    # --------------------------------------------------------

    category_data = []

    for category in LABEL_NAMES.values():

        category_data.append({
            "name": category,
            "classifier": _classifier_per_class[category],
            "expert": _expert_per_class[category],
        })

    # --------------------------------------------------------
    # SELECTED NEWS
    # --------------------------------------------------------

    try:
        news_index = int(selected_news)
    except ValueError:
        news_index = 0

    if news_index < 0 or news_index >= len(_test_texts):
        news_index = 0

    news_text = _test_texts[news_index]
    true_label = _test_labels[news_index]

    classifier_prediction = _classifier_predictions[news_index]

    # Use a fixed seed so the selected expert result is reproducible
    random.seed(42 + news_index)
    expert_prediction = simulated_expert(true_label)

    news_result = {
        "text": news_text,
        "true_category": LABEL_NAMES[true_label],
        "classifier": LABEL_NAMES[classifier_prediction],
        "expert": LABEL_NAMES[expert_prediction],

        "classifier_correct":
            classifier_prediction == true_label,

        "expert_correct":
            expert_prediction == true_label,
    }

    # --------------------------------------------------------
    # NEWS OPTIONS
    # --------------------------------------------------------

    news_options = []

    for i, text in enumerate(_test_texts[:100]):

        news_options.append({
            "index": i,
            "text": text[:100] + "..."
        })

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = {

        "selected_category": selected_category,

        "categories": [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ],

        "classifier_accuracy":
            classifier_accuracy,

        "expert_accuracy":
            expert_accuracy,

        "category_data":
            category_data,

        "news_options":
            news_options,

        "selected_news":
            news_index,

        "news_result":
            news_result,

        "train_samples":
            _train_samples,

        "test_samples":
            _test_samples,
    }

    return render(
        request,
        "project3/index.html",
        context
    )
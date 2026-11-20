from llm.training.mixture import adjust_for_languages, group_language, language_scores


def test_languages_with_lower_scores_receive_more_weight_without_touching_other_groups():
    weights = {"chat_en": 0.1, "chat_ru": 0.1, "reading_kk": 0.1, "math": 0.1}
    adjusted = adjust_for_languages(weights, {"en": 0.8, "ru": 0.4, "kk": 0.0})
    assert adjusted["math"] == weights["math"]
    assert adjusted["chat_en"] < adjusted["chat_ru"] < adjusted["reading_kk"]
    assert adjusted["reading_kk"] == 0.2


def test_group_language_detection_and_score_extraction():
    assert group_language("chat_kk") == "kk" and group_language("chat_other") is None
    assert group_language("math") is None
    assert language_scores({"conv_lang_en": 0.5, "conv_pass_rate": 1.0}) == {"en": 0.5}

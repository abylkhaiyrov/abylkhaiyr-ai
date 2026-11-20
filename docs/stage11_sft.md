# Этап 11. Сохранение языковых способностей при SFT

## Диагноз legacy SFT

Оба веса проверены на одних и тех же окнах отложенного корпуса v2 и на одном Conversational Benchmark с фиксированным декодированием. Итог 6000 шагов предобучения и 1500 шагов SFT-v1:

| Метрика | Pretrain | SFT-v1 |
|---|---:|---:|
| Test perplexity, все токены | 19,66 | 46,55 |
| Test perplexity, EN | 25,53 | 33,50 |
| Test perplexity, RU | 18,80 | 50,37 |
| Test perplexity, KK | 18,96 | 284,57 |
| Conversational Benchmark | 5/53 | 10/53 |

SFT-v1 улучшил диалоговый результат, но ухудшил моделирование обычного текста. Особенно сильно пострадал казахский. Рост test perplexity в 2,37 раза нарушил действовавший порог 1,5× для DPO.

- Пик learning rate legacy SFT составлял `2e-4` после 100 шагов warmup. За прогон обработано 3 457 366 токенов с метками ассистента.
- Assistant-only validation loss на собственной выборке v2 снижался с 1,718 на шаге 150 до 1,510 на шаге 1500. Этот ряд не подтверждает переобучение на той же функции потерь. Он не измеряет сохранение языковых способностей и не сравнивается с full-token perplexity.
- В legacy смеси на код приходилось 15% батчей при 382 примерах группы, тогда как на математику приходилось 35% при 39 154 примерах. Частое повторение кода и отсутствие replay создавали риск смещения. Исторические метрики не позволяют разложить регрессию по отдельным причинам.

## Бенчмарк и подбор параметров

Apple MPS доступен. Перед измерением активного процесса обучения не было. Studio работала в фоне. FP32, 3 повтора, 2 шага прогрева и 4 измеряемых шага на повтор.

| Вариант | Медиана целевых токенов/с | Пик устройства, GiB | Доля полезных позиций |
|---|---:|---:|---:|
| Без packing и foreach | 2 839 | 4,14 | 21,6% |
| Multi-tensor AdamW | 3 095 | 4,14 | 21,6% |
| Packing + multi-tensor AdamW | 5 068 | 3,08 | 35,3% |

В четырёх проверенных шагах AdamW не дал расхождения loss с baseline. Комбинация packing и foreach ускорила обработку целевых токенов примерно в 1,79 раза. До измерения уже использовалось около 12 GiB swap, поэтому абсолютную скорость нельзя переносить на незагруженную машину. Native GQA, BF16 и другие непроверенные в этом прогоне настройки не включены.

Пилоты стартовали из одного `checkpoints/tiny_v2/tiny_best`. Цель — около 120 000 токенов с метками за прогон. Все варианты оценены на едином unpacked assistant-only validation и отдельно на одинаковых окнах обычного текста.

| Пилот | Токенов с метками | Assistant validation loss | Corpus validation perplexity |
|---|---:|---:|---:|
| Взвешенная смесь, LR 3e-5 | 118 005 | 2,873 | 20,62 |
| Взвешенная смесь, LR 8e-5 | 118 005 | 2,654 | 21,63 |
| Равные веса категорий, LR 8e-5 | 117 816 | 2,609 | 22,30 |
| Взвешенная смесь + 3% replay батчей, LR 8e-5 | 116 669 | 2,694 | 21,19 |

Исходная corpus validation perplexity — 19,97. Для полного прогона выбраны веса v3.1 и LR 8e-5 с replay: при том же пике LR replay лучше сохранил full-token качество, с небольшим ухудшением assistant-only loss. Равные веса дали меньший assistant-only loss и больший рост corpus perplexity. Четыре пилота с одним сидом не устанавливают глобальный оптимум.

## SFT-v3.1 и выбор checkpoint

Конфигурация `configs/tiny_v31_sft_stage11.yaml` сохраняет отдельные checkpoints на шагах 40, 80 и 120. Assistant-only validation не использует replay. Replay-батчи формируются из уже проверенных токенов только `train.en.bin`, `train.ru.bin` и `train.kk.bin` корпуса v2, с равными вероятностями языков. Каждый replay-токен получает цель полного next-token prediction.

Прогон завершён на 120 шагах за 138 с. Из 810 843 токенов цели **741 347** приходятся на ответы ассистента, **69 496** на replay. Доля replay по батчам задана как 3%; фактическая доля по токенам — 8,57%. Проверка v3.1 не нашла пересечений разбиваемых примеров и eval prompts по применённым правилам аудита.

| Checkpoint | Assistant validation loss, v3.1 | Corpus validation PPL | EN | RU | KK |
|---|---:|---:|---:|---:|---:|
| Pretrain | — | 19,97 | 20,24 | 16,30 | 16,68 |
| SFT-v1 | — | 51,68 | 28,14 | 51,77 | 355,38 |
| v3.1, 40 | 2,447 | 22,11 | 22,09 | 18,68 | 19,54 |
| v3.1, 80 | 2,269 | 22,48 | 22,24 | 19,06 | 20,42 |
| v3.1, 120 | **2,203** | 22,41 | 22,30 | 18,99 | 20,87 |

До оценки test выбрали шаг 120 по минимальному assistant-only loss среди checkpoint, сохранивших общую validation perplexity в пределах 1,25× pretrained и каждый язык в пределах 1,5×.

## Sampling на validation

Пять вариантов декодирования, шесть примеров на группу, лимит 64 новых токена. Отбор учитывает token F1, завершённость и повторы. Настройки посчитаны на validation и не использованы для подбора по test.

| Категория | Лучший вариант | Token F1 |
|---|---|---:|
| Диалог | greedy, repetition penalty 1,2 | 0,419 |
| Объяснения | greedy, repetition penalty 1,2 | 0,302 |
| Навыки | greedy, repetition penalty 1,0 | 0,781 |
| Математика | temperature 0,5, top-p 0,9, repetition penalty 1,1 | 0,545 |
| Код | temperature 0,5, top-p 0,9, repetition penalty 1,1 | 0,451 |

Token F1 проверяет совпадение ответа с эталоном по токенам. Для математики и кода семантическая проверка проводится отдельно. Результаты sweep на малом числе примеров не меняют настройки продукта.

## Независимый test и решение

На тесте использовали фиксированные параметры Conversational Benchmark для всех трёх моделей. Sampling sweep на это решение не влиял.

| Метрика | Pretrain | SFT-v1 | SFT-v3.1, шаг 120 |
|---|---:|---:|---:|
| Test perplexity | 19,66 | 46,55 | **22,16** |
| Test perplexity EN | 25,53 | 33,50 | 28,24 |
| Test perplexity RU | 18,80 | 50,37 | 21,89 |
| Test perplexity KK | 18,96 | 284,57 | 23,78 |
| Conversational Benchmark | 5/53 | 10/53 | **17/53** |
| Завершённость ответов | 1,89% | 84,91% | 94,34% |
| Повторяемость триграмм в разговорном тесте | 35,51% | 16,77% | 5,18% |

У выбранного checkpoint full-token perplexity на 12,7% выше pretrained. На отдельных языках рост укладывается в 1,5×. Средняя корпусная perplexity и метрики моноязычных текстов используют разные выборки, поэтому языковые показатели не усреднялись для восстановления общей оценки.

На 60 задачах математического набора режимы оценивались отдельно: **3,33%** для модели, **6,67%** с проверкой и **85%** с инструментами. На отдельном наборе логики модель дала **0%**, режим с инструментами — **16,67%**. Фактические ответы и выполнение кода по `pass@1` и `pass@4` получили **0%**. Оценка выполнена на локальных наборах. В этом прогоне внешний GSM8K не загружался.

DPO пропущен: 17/53 (**32,08%**) ниже порога **50%** для conversational gate. В локальном `checkpoints/production.json` поле `production` равно `null`, а шаг 120 записан в `experimental` с измеренными метриками. Legacy SFT не прошёл также общий, русский и казахский пороги retention. Файл `experiments/stage11/dpo_decision.json` содержит причину пропуска.

## Воспроизведение

Из корня проекта, с локальными данными v2, v3.1 и исходным checkpoint:

```bash
.venv/bin/python scripts/profile_training.py --config configs/tiny_v31_sft.yaml \
  --variants baseline foreach packed --steps 4 --repeats 3 --stability-steps 4 \
  --output experiments/stage11/mps.json
.venv/bin/python scripts/tune_sft.py --budget 120000
.venv/bin/llm finetune --config configs/tiny_v31_sft_stage11.yaml

stage=checkpoints/tiny_v31_sft_stage11
base=checkpoints/tiny_v2/tiny_best
legacy=checkpoints/tiny_v2_sft/tiny_sft_best
selected="$stage/step-00000120"
.venv/bin/python scripts/evaluate_sft.py --split validation \
  --output experiments/stage11/validation.json base="$base" legacy="$legacy" \
  stage40="$stage/step-00000040" stage80="$stage/step-00000080" stage120="$selected"
.venv/bin/python scripts/sampling_sweep.py "$selected" --validation-dir data/v3_1 \
  --per-group 6 --max-new-tokens 64 --validation-only \
  --variants greedy_rp1.0 greedy_rp1.1 greedy_rp1.2 t0.3_p0.9 t0.5_p0.9_rp1.1 \
  --output experiments/stage11/sampling.json
.venv/bin/python scripts/evaluate_sft.py --split test --retention-only \
  --output experiments/stage11/test_reference.json base="$base" legacy="$legacy"
.venv/bin/python scripts/evaluate_sft.py --split test \
  --output experiments/stage11/test_selected.json stage11="$selected"
```

Для повторной записи манифеста объедините верхнеуровневые записи двух test JSON в `final_eval.json` и передайте их в `scripts/promote_model.py` с `--reference base` и аргументами `stage11=$selected=sft` и `legacy=$legacy=sft`.

Артефакты эксперимента, включая веса и `checkpoints/production.json`, остаются локальными. В Git хранятся конфигурация, код и агрегированные результаты.

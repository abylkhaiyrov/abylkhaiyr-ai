# Математика модели

Все формулы реализованы вручную в `src/llm/model`, `src/llm/optimization`, `src/llm/training/losses.py`
и сверяются в `tests/` с эталонными значениями.

## Softmax со стабилизацией
`softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`. Вычитание максимума не меняет результат,
но исключает переполнение `exp`. Функция: `model/attention.py::stable_softmax`.

## Scaled Dot-Product Attention
`Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V`, где `M[i, j] = -inf` при `j > i` (причинная маска).
Эталон: `reference_attention` (вычисления не ниже float32). Быстрая версия: `torch SDPA`.
Тесты проверяют совпадение, `gradcheck` в float64 и отсутствие утечки будущих токенов
(изменение токенов с позиции t не меняет логиты позиций < t).

## Grouped Query Attention
`H` голов запроса делят `G` голов ключей и значений (`H % G == 0`). Каждая KV-голова повторяется `H/G` раз
(`repeat_kv`). При `G = H` получается Multi-Head Attention, при `G = 1` — Multi-Query Attention.

## RoPE
Для пары координат `(x_{2i}, x_{2i+1})` на позиции `p` применяется поворот на угол `p * theta^(-2i/d)`.
Скалярное произведение повёрнутых `q` и `k` зависит только от разности позиций `p_q - p_k`
(проверено в `test_rope_scores_depend_only_on_relative_distance`). Реализация в стиле `rotate_half`.

## RMSNorm
`RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g`. Нормировка считается в float32 при любой точности модели.

## SiLU и SwiGLU
`SiLU(x) = x * sigmoid(x)`, `SwiGLU(x) = W_down (SiLU(W_gate x) * (W_up x))`.

## Функция потерь
`L = -(1/N) * sum_t log p(y_t | y_<t)`, где `N` — число токенов с меткой, отличной от `-100`.
При накоплении градиентов сумма потерь каждого микробатча делится на общее `N` всего шага, поэтому результат
не зависит от разбиения на микробатчи (`test_gradient_accumulation_equals_large_batch`).

## AdamW
`m = b1 m + (1-b1) g`, `v = b2 v + (1-b2) g^2`, `theta = theta (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)`.
Собственная реализация сверяется с `torch.optim.AdamW`. Weight decay не применяется к одномерным параметрам.

## Расписание LR
Линейный прогрев до `lr`, затем косинусный спад до `lr * min_lr_ratio`.

## Клиппинг градиента
`g <- g * min(1, max_norm / (||g||_2 + 1e-6))`, норма глобальная по всем параметрам. Нефинитная норма пропускает шаг.

## DPO
`L = -log sigmoid(beta * ((log pi(y_w) - log pi(y_l)) - (log ref(y_w) - log ref(y_l))))`,
где `ref` — замороженная копия SFT-модели.

## Pass@k
Несмещённая оценка: `1 - prod_{i=0}^{c-1} (1 - k / (n - i))` при `n - c >= k`, иначе 1.

## Численная стабильность
Softmax, RMSNorm, логиты и потери считаются в float32 внутри autocast (FP16/BF16). GradScaler используется
только для FP16 на CUDA.

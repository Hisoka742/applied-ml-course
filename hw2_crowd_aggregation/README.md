# ДЗ #2. Агрегация асессорских оценок

Сгенерировать или найти датасет асессорских оценок, посчитать по нему разные статистики и описать, почему они
похожи на реальные данные. Применить методы агрегации Majority Vote, Weighted Majority Vote и Dawid-Skene,
замерить качество, сделать выводы.

**Решение:** [hw2_crowd_aggregation.ipynb](hw2_crowd_aggregation.ipynb). Там описаны генеративная модель
данных, статистики и почему они похожи на реальные, сами методы, результаты и выводы.

## Структура

| Файл | Содержимое |
|------|------------|
| `crowd/simulation.py` | генератор краудсорсинговой разметки с известной истинной разметкой |
| `crowd/statistics.py` | статистики разметки: активность исполнителей, Gini, Krippendorff's alpha, согласованность |
| `crowd/aggregation.py` | Majority Vote, Weighted Majority Vote, Dawid-Skene (EM), оценки качества исполнителей |
| `crowd/experiments.py` | запуск всех методов, метрики, серии экспериментов |
| `crowd/plotting.py` | общие графики для всех методов |
| `tests/` | тесты методов агрегации, статистик и генератора |

## Запуск

Все команды выполняются из папки `hw2_crowd_aggregation`. Зависимости — `requirements.txt` в корне репозитория.

```bash
pip install -r ../requirements.txt
python -m pytest                     # tests, about 3 s
jupyter lab hw2_crowd_aggregation.ipynb
```

Перезапустить ноутбук целиком без открытия Jupyter (около 1.5 минут на CPU):

```bash
jupyter nbconvert --to notebook --execute --inplace hw2_crowd_aggregation.ipynb
```

Датасет не хранится в репозитории: он генерируется кодом с фиксированным `SEED`, поэтому результаты
воспроизводятся один в один.

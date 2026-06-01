## Выводы по baseline

На этапе baseline были обучены и сравнены четыре модели:

- `DummyRegressor`
- `LinearRegression`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`

Baseline строился на исходных признаках без feature engineering, логарифмирования и дополнительной обработки гипотез из EDA. Единственная обработка данных на этом этапе — one-hot encoding категориальных признаков `cut`, `color` и `clarity`.

Качество моделей оценивалось только на train-части данных с помощью 5-fold cross-validation. Тестовая выборка не использовалась и остается отложенной до финальной оценки модели.

По результатам cross-validation лучшую метрику RMSE показал `HistGradientBoostingRegressor`. Немного хуже сработал `RandomForestRegressor`. `LinearRegression` заметно уступила ансамблевым моделям, что ожидаемо для задачи с нелинейными зависимостями между признаками и ценой. `DummyRegressor` показал самый слабый результат и используется только как нижняя граница качества.

Таким образом, в качестве основной модели для дальнейших экспериментов выбран `HistGradientBoostingRegressor`. Именно на нем дальше будут проверяться гипотезы feature engineering и подбираться гиперпараметры.
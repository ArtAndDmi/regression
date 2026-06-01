## Выводы по hyperparameter tuning

На этапе hyperparameter tuning была зафиксирована лучшая схема feature engineering из предыдущего этапа: `geometry_without_xyz_plus_ordinal_clarity`.

Для модели `HistGradientBoostingRegressor` был выполнен подбор гиперпараметров с помощью `GridSearchCV` и 5-fold cross-validation. Метрика качества — RMSE.

Тестовая выборка на этом этапе не использовалась.

Лучший результат после tuning составил RMSE = 546.09. Для сравнения, лучший результат после feature engineering до подбора гиперпараметров составлял RMSE = 551.67.

Таким образом, tuning улучшил качество модели примерно на 5.58 RMSE.

Лучшие параметры модели:

- `learning_rate = 0.04`
- `max_iter = 250`
- `max_leaf_nodes = 63`
- `min_samples_leaf = 20`

В финальную модель стоит брать tuned-версию pipeline с выбранной схемой feature engineering и найденными гиперпараметрами.
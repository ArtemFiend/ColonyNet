# Веса признаков для оценки аномальности колоний

В текущем алгоритме веса задаются не для каждого отдельного сырого признака, а для групп признаков и ансамблевых оценок. Внутри каждой группы признаки нормализуются через устойчивую нормализацию, затем сворачиваются в один group score.

Финальный score считается как взвешенное среднее рангов:

`final_anomaly_score_raw = weighted_average(score_rank_i, weight_i)`

| Признак / компонент | Описание на русском | Вес | Как вычисляется |
| --- | --- | ---: | --- |
| `lof_score_rank` | Локальная выбросность объекта в общем пространстве признаков. Показывает, насколько колония плохо вписывается в локальную плотность похожих объектов. | 0.18 | Все числовые признаки нормализуются через `RobustScaler`, затем применяется `LocalOutlierFactor`. Чем выше LOF-score, тем сильнее объект отличается от локального окружения. Затем score переводится в rank от 0 до 1. |
| `size_shape_score_rank` | Размер и форма колонии: площадь, диаметр, округлость, вытянутость, выпуклость, неровность контура. | 0.15 | Используются `log_area`, `equivalent_diameter`, `area_to_median_ratio`, `aspect_ratio`, `circularity`, `eccentricity`, `solidity`, `convexity_defect_ratio`, `radial_contour_cv`. Признаки нормализуются, затем считается `0.7 * mean(abs(robust_z)) + 0.3 * max(abs(robust_z))`, после чего результат переводится в rank. |
| `texture_score_rank` | Текстурная неоднородность колонии: зернистость, контраст, локальный рисунок внутри маски. | 0.15 | Используются `laplacian_var`, GLCM-признаки (`glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`, `glcm_dissimilarity`, `glcm_ASM`) и LBP-признаки (`lbp_mean`, `lbp_std`, `lbp_entropy`). Считается robust group score и rank. |
| `isolation_forest_score_rank` | Глобальная выбросность объекта по ансамблю случайных разбиений. | 0.12 | Все числовые признаки нормализуются, затем применяется `IsolationForest`. Объекты, которые легче изолируются, получают более высокий score. Затем score переводится в rank. |
| `robust_z_score_rank` | Общая статистическая удаленность колонии от медианного объекта по всем числовым признакам. | 0.12 | По всем числовым признакам после robust scaling считается общий outlier score: `0.7 * mean(abs(robust_z)) + 0.3 * max(abs(robust_z))`. Затем результат переводится в rank. |
| `intensity_score_rank` | Яркость внутри колонии и распределение интенсивности: неоднородность, разброс яркости, отличие центра от края. | 0.10 | Используются `intensity_mean`, `intensity_median`, `intensity_iqr`, `intensity_p95_p05_range`, `intensity_cv`, `intensity_entropy`, `center_rim_intensity_delta`, `center_rim_intensity_ratio`, `radial_intensity_slope`, `radial_intensity_std`. Считается robust group score и rank. |
| `color_background_score_rank` | Цвет колонии и отличие от локального фона вокруг нее. | 0.08 | Используются Lab/HSV признаки: `mean_L_lab`, `mean_a_lab`, `mean_b_lab`, `std_L_lab`, `std_a_lab`, `std_b_lab`, `mean_S`, `mean_V`, `mean_H_sin`, `mean_H_cos`, а также локальные отличия `local_delta_L`, `local_delta_a`, `local_delta_b`, `local_color_delta_lab`. Считается robust group score и rank. |
| `histogram_score_rank` | Отличие формы гистограммы яркости/контраста от типичной гистограммы чашки и ближайших соседей. | 0.08 | Используются `hist_intensity_entropy`, `hist_intensity_width`, `hist_intensity_skewness`, `hist_intensity_kurtosis`, `dark_fraction`, `bright_fraction`, `contrast_hist_entropy`, `intensity_hist_js_to_plate_median`, `intensity_hist_wasserstein_to_plate_median`, `intensity_hist_js_to_neighbors`, `L_hist_js_to_plate_median`, `contrast_hist_js_to_plate_median`. Считается robust group score и rank. |
| `neighbor_difference_score_rank` | Отличие колонии от ближайших соседей по размеру, форме, цвету, текстуре и гистограмме. | 0.05 | Для каждой колонии берутся ближайшие соседи, затем считаются относительные отклонения: `relative_area_vs_neighbors`, `relative_intensity_vs_neighbors`, `relative_texture_vs_neighbors`, `relative_circularity_vs_neighbors`, `relative_solidity_vs_neighbors`, `relative_color_delta_lab_vs_neighbors`, `relative_local_contrast_vs_neighbors`, `relative_entropy_vs_neighbors`, `relative_histogram_vs_neighbors`, `feature_knn_distance`. Считается robust group score и rank. |
| `morphotype_score_rank` | Отклонение от своего морфотипа или принадлежность к редкому морфотипу. | 0.08 | Колонии сначала группируются в морфотипы через KMeans/DBSCAN, затем сравниваются с похожими колониями внутри своего морфотипа. Считаются `within_morphotype_anomaly_score`, `size_within_morphotype_z`, `texture_within_morphotype_z`, `color_within_morphotype_z`, `histogram_within_morphotype_z`. Редкий морфотип сам по себе отправляет объект скорее на review, а не автоматически в аномалии. |
| `spatial_context_score_rank` | Пространственная изолированность или необычная локальная плотность колонии на чашке. | 0.03 | Используются `nearest_neighbor_distance`, `mean_5nn_distance`, `log_nearest_neighbor_distance`, `log_mean_5nn_distance`, `edge_nearest_distance`, `mean_5nn_edge_distance`, `nearest_distance_to_median_diameter_ratio`, `edge_distance_to_median_diameter_ratio`, `local_density_r`. Это слабый контекстный признак: он повышает score, но не считается самостоятельным сильным доказательством. |
| `cluster_outlier_score_rank` | Удаленность объекта от кластеров в общем пространстве признаков. | 0.03 | После robust scaling выполняется DBSCAN. Если DBSCAN не дает устойчивых кластеров, используется KMeans. Score равен удаленности от центра кластера или штрафу за noise-объект, затем переводится в rank. |

## Технические фильтры без веса

Эти признаки не имеют веса в финальном score. Они используются как фильтр качества, чтобы не принять ошибку сегментации или нестабильный результат за настоящую аномальную колонию.

| Признак | Описание на русском | Вес | Как вычисляется |
| --- | --- | ---: | --- |
| `yolo_conf` / `low_yolo_conf` | Уверенность YOLO в найденной маске. | 0 | Если `yolo_conf < min_yolo_conf_for_analysis`, объект получает техническое предупреждение. |
| `is_too_small` | Маска слишком маленькая. | 0 | Площадь сравнивается с абсолютным минимумом и минимумом относительно медианной площади по чашке. |
| `is_too_large` | Маска слишком большая. | 0 | Площадь сравнивается с долей площади изображения и с максимумом относительно медианной площади по чашке. |
| `touches_image_border` | Маска касается границы изображения. | 0 | Проверяется пересечение маски с краем изображения в пределах `edge_margin_px`. |
| `suspicious_aspect_ratio` | Подозрительно вытянутая маска. | 0 | Сравнивается отношение ширины bbox к высоте bbox с порогом. |
| `mask_fragment_after_overlap` | Маска стала фрагментом после удаления пересечений с другими масками. | 0 | Сравнивается площадь до/после overlap cleanup и минимальная площадь компоненты. |
| `invalid_geometry` | Некорректная геометрия маски. | 0 | Флаг ставится, если после очистки объект не может надежно интерпретироваться как отдельная колония. |
| `possible_segmentation_artifact` | Объект похож на технический артефакт сегментации. | 0 | Комбинация фрагментации, некорректной геометрии, подозрительной формы и большой потери площади. |
| `review_segmentation` | Отдельный статус для слипшихся, обрезанных, низкоуверенных или подозрительных масок. | 0 | Если есть признаки плохой сегментации (`low_yolo_conf`, `is_too_large`, `mask_fragment_after_overlap`, `invalid_geometry`, `possible_segmentation_artifact`, касание края), объект не смешивается с биологическими аномалиями, а уходит на ручной просмотр сегментации. |
| `perturbation_stability_score` | Устойчивость кандидата к легким изменениям изображения. | 0 | Для кандидатов выполняются дополнительные прогоны YOLO при `brightness_plus_5pct`, `contrast_minus_5pct` и `slight_blur`. Кандидат остается автоматической аномалией, если повторно попадает в top-кандидаты минимум в 2 из 3 проверок (`score >= 0.66`). Иначе статус меняется на `unstable_candidate`. |

## Правило отбора

Высокого веса и высокого `final_anomaly_score_raw` недостаточно. Чтобы объект попал в `selected_anomalies.csv`, дополнительно нужны:

| Условие | Текущее значение |
| --- | ---: |
| `final_anomaly_score_raw` | >= 0.65 |
| `absolute_evidence_strength` | >= 0.35 |
| `independent_evidence_count` | >= 2 |
| `overall_reliability_score` | >= 0.50 |
| `consensus_score` | >= 0.30 |
| `perturbation_stability_score` | >= 0.66 |
| `technical_warning` | False |
| Доля визуально выделенных объектов | не больше 20% колоний, но не более 20 объектов, суммарно по всем визуальным статусам |
| Diversity визуализации | не больше 1-2 близко расположенных объектов из одной плотной зоны или одного морфотипа |
| Краевая зона чашки в визуализации | не больше 25% highlight-набора для объектов ближе 2 медианных диаметров колонии к краю чашки; сильные `select_candidate` не скрываются |
| `review_segmentation` в визуализации | не больше 20% highlight-набора и не больше 4 объектов; на картинке подписывается как `SEG`, а не числом `0.00` |

Пространственная удаленность (`spatial_context_score`) сейчас не засчитывается как самостоятельная сильная evidence-группа. Она усиливает итоговый score, но автоматический выбор требует подтверждения другими группами: форма, текстура, яркость, цвет, гистограмма или отличие от соседей.

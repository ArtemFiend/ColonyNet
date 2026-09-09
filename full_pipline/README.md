# Полный pipeline ColonyNet

Рабочая реализация: `full_pipeline.py`. Windows-интерфейс: `colony_pipeline_app.py`.

## Этапы анализа

1. Детекция чашки моделью YOLO26s.
2. Обрезка области чашки и приведение к 736 × 736.
3. Instance segmentation колоний моделью YOLO26x-seg.
4. Очистка масок и разрешение пересечений.
5. Вычисление морфологических, текстурных и пространственных признаков.
6. Поиск аномалий, проверка устойчивости, выделение объектов для проверки.
7. Экспорт CSV/Excel и изображений этапов.

## Модели

В папку `models/` нужно отдельно положить доверенные веса:

| Файл | SHA-256 проверенной локальной модели |
| --- | --- |
| `petri_detector_yolo26s_best.pt` | `adc76b3226de7f219d8da9a3afecd45d9d4729b80ce6e44a9f99341d5108223f` |
| `colony_yolo26x_seg_best.pt` | `4aa01bb1c43c1484820c22faafbfa5553cc255c3813595beee887a787f67e2f5` |

Веса не включены в Git. Без них inference не запускается. Контрольные суммы относятся к локальным файлам, использованным при проверке; при переобучении моделей их нужно обновить.

## Запуск из корня проекта

```powershell
python -m full_pipline.colony_pipeline_app --self-test
python -m full_pipline.colony_pipeline_app
python -m full_pipline.colony_pipeline_app --run-once image.jpg --output outputs/demo
```

Python API:

```python
from pathlib import Path
from full_pipline.full_pipeline import (
    make_full_pipeline_config, load_pipeline_models, run_single_image_pipeline,
)

config = make_full_pipeline_config(output_dir='outputs/demo')
segmenter, detector = load_pipeline_models(config)
result = run_single_image_pipeline(Path('image.jpg'), segmenter, config, detector)
```

Прямой API и CLI сохраняют результаты в указанный каталог: используйте новый путь для каждого запуска. GUI создаёт отдельный каталог автоматически. Имена снимков без расширения должны быть уникальны в пределах запуска.

## Интерпретация

`selected` — отобранные аномалии; `review_candidates` — объекты для ручной проверки; `technical_warnings` — технические проблемы масок. `visual_highlights` — объекты, выбранные для отображения: их число не обязано совпадать с числом отобранных аномалий. Score не является вероятностью биологической аномалии.

Исследовательские notebooks сохранены как исходный код без вычисленных выводов. Они могут требовать дополнительных библиотек и локальных датасетов. Достоверную оценку качества получают на независимой тестовой выборке, разделённой по исходным чашкам.

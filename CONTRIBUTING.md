# Работа с репозиторием

## Подготовка к публикации

Чистая копия исходников создаётся отдельно от рабочей папки экспериментов:

```powershell
python tools/prepare_release.py github-ready
```

Команда требует новую папку: существующий экспорт не перезаписывается. Копируются исходники, конфигурации, документация и notebooks без сохранённых выводов и вложений. Датасеты, веса и результаты не копируются. Оригинальные notebooks остаются неизменными. Копия не содержит `.git`, токенов доступа или настроек подключения к GitHub.

Для существующего GitHub-репозитория перенесите содержимое экспорта в отдельный свежий clone, просмотрите diff и опубликуйте обычным коммитом. Не используйте `git add .` из папки с экспериментами и не переписывайте историю ради обновления файлов.

```powershell
git clone https://github.com/ArtemFiend/ColonyNet.git ColonyNet-publish
# Скопируйте содержимое github-ready в ColonyNet-publish, включая скрытые файлы.
cd ColonyNet-publish
git status --short
git diff --stat
python -m unittest discover -s tests -v
python tools/check_repository.py
git add --all
git diff --cached --stat
git commit -m "Update ColonyNet application and repository safety"
git push origin main
```

## Проверки

`python -m unittest discover -s tests -v` проверяет разделение выборок и защиту файловых операций без ML-зависимостей. `python tools/check_repository.py` проверяет отслеживаемые файлы на запрещённые артефакты, выводы notebooks и распространённые форматы секретов. Значения найденных секретов не выводятся.

Для изменений inference дополнительно выполните `--self-test`, `tools/smoke_desktop.py` и полный `--run-once` на локальном тестовом изображении. CI не содержит приватных моделей и поэтому не проверяет точность сегментации.

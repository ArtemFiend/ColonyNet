# Windows-сборка ColonyNet

Сборка требует Windows, Python 3.10, установленных зависимостей приложения и двух локальных моделей, описанных в [README](README.md).

```powershell
py -3.10 -m venv .venv-app
.\.venv-app\Scripts\python.exe -m pip install -r requirements-app.txt pyinstaller
.\full_pipline\build_exe.ps1
```

Скрипт выбирает `.venv-app`, затем `.venv`, затем системный Python. Ошибка PyInstaller возвращается как ошибка сборки.

Результат: `dist/ColonyNetPipeline/ColonyNetPipeline.exe`. Это сборка **onedir**: переносите всю папку `ColonyNetPipeline`, включая `_internal`, а не один EXE. Сборка включает локальные веса; перед распространением отдельно проверьте разрешения на распространение моделей и лицензии зависимостей.

```powershell
.\dist\ColonyNetPipeline\ColonyNetPipeline.exe --self-test
.\dist\ColonyNetPipeline\ColonyNetPipeline.exe --run-once image.jpg --output outputs/demo
```

Результаты GUI по умолчанию сохраняются в `%LOCALAPPDATA%/ColonyNet/outputs`. Каждый запуск GUI получает отдельную папку. Исходные снимки и отчёты сохраняются локально и могут содержать чувствительные данные.

Готовые сборки не хранятся в Git. После изменения Python-кода EXE нужно пересобрать. GitHub Actions выполняет проверки исходников; он не собирает приложение с приватными весами и не проверяет точность моделей.

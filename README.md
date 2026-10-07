# Speaker verification distillation

## Запуск на GPU-сервере через uv

Нужны Linux x86_64, NVIDIA GPU, драйвер с поддержкой CUDA 12.4 и установленный
[uv](https://docs.astral.sh/uv/getting-started/installation/).
PyTorch и torchaudio закреплены на версии 2.6.0; для Linux используются
сборки CUDA 12.4 из официального индекса PyTorch по
[схеме настройки uv для PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
Python 3.11 uv скачает автоматически, если его нет на сервере.

Из корня репозитория:

```bash
uv sync --locked

# Проверка окружения и доступности GPU
uv run python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

cd egs/voxceleb-v2
uv run python distill.py
```

`uv sync` автоматически создаёт окружение `.venv/` в корне репозитория.
`uv run` находит его и из каталога `egs/voxceleb-v2` по родительскому
`pyproject.toml`. Активировать окружение не требуется. `voicesdk` устанавливается
в него в editable-режиме, поэтому `PYTHONPATH` задавать не нужно.

Скрипты используют относительные пути `data/...`: запускайте их из
`egs/voxceleb-v2`. До обучения настройте в выбранном скрипте пути к датасетам,
конфигурациям, весам учителя и центроидам. Эти данные не устанавливаются uv.

## Добавление пакетов

Из корня репозитория:

```bash
uv add PACKAGE
```

Команда обновит `pyproject.toml`, `uv.lock` и установит пакет в `.venv/`.
На другой машине достаточно `uv sync --locked`.

Скриптам `distill-mic-aug.py`, `distill-mic-aug-campp.py` и
`train-antispoof.py` дополнительно нужен `audimentation` с API
`DataSample`, `SequentialCompose`, `FileListAudioProvider`, `AddNoise`, `Reverb`.
Его источник пока не указан в репозитории. Если это отдельный локальный проект,
добавьте его на сервере из корня этого репозитория:

```bash
uv add --editable /absolute/path/to/audimentation
```

Если пакет находится в Git, используйте `uv add 'audimentation @ git+https://…'`
с реальным URL. После добавления можно запускать скрипты с аугментациями через
`uv run python distill-mic-aug.py` из `egs/voxceleb-v2`.

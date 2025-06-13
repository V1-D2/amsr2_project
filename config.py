import pathlib

# Настройки для локального запуска в PyCharm
BASE_DIR = pathlib.Path("./data")  # Локальная папка для данных
TEMP_DIR = pathlib.Path("./temp")  # Временная папка

# G-Portal credentials
GPORTAL_USERNAME = "Vlad_Dia"
GPORTAL_PASSWORD = "GIAgia12345@"

# Создаем папки если их нет
BASE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
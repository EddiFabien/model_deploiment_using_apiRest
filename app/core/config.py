from functools import lru_cache
import os
from pathlib import Path
from decouple import config, Csv

class Settings:
    # Application settings
    app_name: str = config("APP_NAME", default="OptionClass API")
    app_version: str = config("APP_VERSION", default="1.0.0")
    debug: bool = config("DEBUG", default=False, cast=bool)
    
    # Server settings
    host: str = config("HOST", default="0.0.0.0")
    port: int = config("PORT", default=8000, cast=int)
    
    # Model settings
    model_path: str = config("MODEL_PATH", default="models/optionclass_model.pkl")
    
    # Data settings
    data_path: str = config("DATA_PATH", default="data/raw/base.xlsx")
    processed_data_path: str = config("PROCESSED_DATA_PATH", default="data/processed/processed_data.csv")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Create directories if they don't exist
for path in [
    os.path.dirname(Settings().model_path),
    os.path.dirname(Settings().data_path),
    os.path.dirname(Settings().processed_data_path)
]:
    os.makedirs(path, exist_ok=True)

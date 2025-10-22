from pydantic_settings import BaseSettings

class ObservabilitySettings(BaseSettings):
    PORT: int = 8000
    NEO4J_URL: str = ""
    DB_USER: str = ""
    DB_PASSWORD: str = ""
    COGNITO_USER_POOL_ID: str = ""
    AWS_REGION: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    IMAGE_TAG: str = ""
    SERVICE_NAME: str = ""
    OTLP_ENDPOINT: str = ""
    ENABLE_METRICS: bool = True
    METRICS_PORT: str = ""
    LOG_LEVEL: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = ObservabilitySettings()

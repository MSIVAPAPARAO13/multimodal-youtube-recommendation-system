import os
from db.config import Settings, DatabaseConfig, get_settings


def test_settings_default_values():
    """Test that Settings loads default values correctly."""
    settings = Settings()
    assert settings.POSTGRES_PORT == 5432
    assert settings.POSTGRES_HOST == "localhost"
    assert settings.POSTGRES_DB == "video_recommender"
    assert settings.DB_POOL_SIZE == 5
    assert settings.DB_MAX_OVERFLOW == 10
    assert "postgresql+psycopg2://" in settings.DATABASE_URL
    assert "postgresql+asyncpg://" in settings.ASYNC_DATABASE_URL
    assert isinstance(settings.CORS_ORIGINS, list)
    assert len(settings.CORS_ORIGINS) >= 1


def test_settings_environment_override(monkeypatch):
    """Test environment variable overrides."""
    monkeypatch.setenv("POSTGRES_HOST", "custom-host")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_USER", "custom_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "custom_pass")
    monkeypatch.setenv("POSTGRES_DB", "custom_db")
    monkeypatch.setenv("APP_ENV", "production")

    custom_settings = Settings()
    assert custom_settings.POSTGRES_HOST == "custom-host"
    assert custom_settings.POSTGRES_PORT == 5433
    assert custom_settings.POSTGRES_USER == "custom_user"
    assert custom_settings.APP_ENV == "production"
    assert "custom-host:5433/custom_db" in custom_settings.DATABASE_URL


def test_database_config_backward_compatibility():
    """Test DatabaseConfig class methods."""
    url = DatabaseConfig.get_database_url()
    assert "postgresql+psycopg2://" in url
    async_url = DatabaseConfig.get_database_url(async_driver=True)
    assert "postgresql+asyncpg://" in async_url

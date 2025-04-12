from typing import Optional
from pydantic import BaseModel # Keep BaseModel for potential future use
from pydantic_settings import BaseSettings, SettingsConfigDict
import os # Keep os for now, might be needed for other settings/keys

# Removed ModelConfig class

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Load provider and name directly as top-level settings
    model_provider: str = "openai" # Default provider
    model_name: str = "gpt-4o"   # Default model name

    # API Keys (loaded automatically by BaseSettings from env vars)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    api_ninja_api_key: Optional[str] = None
    weatherapi_api_key: Optional[str] = None # Key for WeatherAPI.com
    ipgeolocation_api_key: Optional[str] = None # Key for ipgeolocation.io
    # Removing tool specific keys/URLs from settings
    # exchangerate_api_key: Optional[str] = None
    # jina_api_key: Optional[str] = None
    # brightdata_proxy_url: Optional[str] = None

    # Add property to get the combined model string
    @property
    def model_string(self) -> str:
        """Get the model string in the format expected by PydanticAI."""
        return f"{self.model_provider}:{self.model_name}"

    # Simplified config: No env_file (rely on explicit load in script)
    model_config = SettingsConfigDict(env_prefix='', env_file_encoding="utf-8", extra="ignore")

# Global settings instance
settings = Settings()
# Removed debug prints 
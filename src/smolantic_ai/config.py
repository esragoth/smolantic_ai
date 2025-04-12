from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider: str = Field(..., description="The model provider (openai, anthropic, gemini)")
    model_name: str = Field(..., description="The specific model name")
    
    @property
    def model_string(self) -> str:
        """Get the model string in the format expected by PydanticAI."""
        return f"{self.provider}:{self.model_name}"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # Model configurations
    multistep_model: ModelConfig = Field(
        default=ModelConfig(provider="openai", model_name="gpt-4o"),
        description="Model configuration for MultistepAgent"
    )
    tool_calling_model: ModelConfig = Field(
        default=ModelConfig(provider="anthropic", model_name="claude-3-opus-latest"),
        description="Model configuration for ToolCallingAgent"
    )
    code_model: ModelConfig = Field(
        default=ModelConfig(provider="openai", model_name="gpt-4o"),
        description="Model configuration for CodeAgent"
    )
    
    # API Keys
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

# Global settings instance
settings = Settings() 
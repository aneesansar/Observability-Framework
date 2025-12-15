"""
Application Configuration.

Centralized settings management using Pydantic.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application-wide configuration settings.
    
    All settings can be overridden via environment variables.
    """
    
    # Service Identity
    service_name: str = Field(
        default="Speedy-Observability",
        alias="SERVICE_NAME",
    )
    api_title: str = Field(
        default="RAG Observability Demo",
        alias="API_TITLE",
    )
    
    # CORS
    cors_origins_str: str = Field(
        default="http://localhost:3000",
        alias="CORS_ORIGINS",
    )
    
    # Keycloak (auth settings are in app.auth.config)
    keycloak_server_url: Optional[str] = Field(
        default=None,
        alias="KEYCLOAK_SERVER_URL",
    )
    realm_name: Optional[str] = Field(
        default=None,
        alias="REALM_NAME",
    )
    client_id: Optional[str] = Field(
        default=None,
        alias="CLIENT_ID",
    )
    client_secret_key: Optional[str] = Field(
        default=None,
        alias="CLIENT_SECRET_KEY",
    )
    callback_uri: str = Field(
        default="http://localhost:8001/callback",
        alias="CALLBACK_URI",
    )
    
    # MongoDB
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        alias="MONGO_URI",
    )
    mongo_db: str = Field(
        default="rag",
        alias="MONGO_DB",
    )
    mongo_coll: str = Field(
        default="trips",
        alias="MONGO_COLL",
    )
    
    # Models / Embeddings
    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBED_MODEL",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_URL",
    )
    llm_model: str = Field(
        default="gemma3:latest",
        alias="LLM_MODEL",
    )
    
    # Role-based telemetry (strings, parsed as properties)
    admin_roles_str: str = Field(
        default="admin,Admin",
        alias="ADMIN_ROLES",
    )
    manager_roles_str: str = Field(
        default="manager,Manager",
        alias="MANAGER_ROLES",
    )
    developer_roles_str: str = Field(
        default="developer,Developer",
        alias="DEVELOPER_ROLES",
    )
    
    # Attribute prefixes per role group
    admin_attr_prefixes_str: str = Field(
        default="Admin,Manager,Developer",
        alias="ADMIN_ATTR_PREFIXES",
    )
    manager_attr_prefixes_str: str = Field(
        default="Manager,Developer",
        alias="MANAGER_ATTR_PREFIXES",
    )
    developer_attr_prefixes_str: str = Field(
        default="Developer",
        alias="DEVELOPER_ATTR_PREFIXES",
    )
    
    # Trace Display Configuration
    trace_span_depth: str = Field(
        default="complete",
        alias="TRACE_SPAN_DEPTH",
        description="Control span visibility: 'parent' (only root spans), 'has_child' (parents with children, no leaf nodes), 'complete' (all spans)",
    )
    
    # Attribute Categories Configuration
    telemetry_categories: str = Field(
        default="basic,llm,deepeval,rag,quality",
        alias="TELEMETRY_CATEGORIES",
        description="Comma-separated list of enabled attribute categories: basic,llm,deepeval,rag,quality,system",
    )
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins as list."""
        return [o.strip() for o in self.cors_origins_str.split(",") if o.strip()]
    
    @property
    def admin_roles(self) -> List[str]:
        """Get admin roles as list."""
        return [r.strip() for r in self.admin_roles_str.split(",") if r.strip()]
    
    @property
    def manager_roles(self) -> List[str]:
        """Get manager roles as list."""
        return [r.strip() for r in self.manager_roles_str.split(",") if r.strip()]
    
    @property
    def developer_roles(self) -> List[str]:
        """Get developer roles as list."""
        return [r.strip() for r in self.developer_roles_str.split(",") if r.strip()]
    
    @property
    def admin_attr_prefixes(self) -> List[str]:
        """Get admin attribute prefixes as list."""
        return [p.strip() for p in self.admin_attr_prefixes_str.split(",") if p.strip()]
    
    @property
    def manager_attr_prefixes(self) -> List[str]:
        """Get manager attribute prefixes as list."""
        return [p.strip() for p in self.manager_attr_prefixes_str.split(",") if p.strip()]
    
    @property
    def developer_attr_prefixes(self) -> List[str]:
        """Get developer attribute prefixes as list."""
        return [p.strip() for p in self.developer_attr_prefixes_str.split(",") if p.strip()]
    
    @property
    def enabled_telemetry_categories(self) -> List[str]:
        """Get enabled telemetry categories as list."""
        return [c.strip().lower() for c in self.telemetry_categories.split(",") if c.strip()]
    
    @property
    def keycloak_server_url_normalized(self) -> Optional[str]:
        """Get normalized Keycloak server URL with trailing slash."""
        if self.keycloak_server_url:
            return self.keycloak_server_url.rstrip("/") + "/"
        return None
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience alias (backwards compatible)
settings = get_settings()

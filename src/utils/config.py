# Configuration management utilities.
#
# Provides functions for loading application configuration from YAML files
# and setting up logging based on configuration settings.

import yaml
import logging
import os

class ConfigError(Exception):
    # Custom exception for configuration-related errors.
    #
    # Raised when configuration file is missing, invalid, or contains
    # incorrect values that prevent application from functioning properly.
    pass

# Configuration utilities for loading and managing application settings.
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ConfigError(Exception):
    # Custom exception for configuration-related errors.
    pass

def _get_app_dir() -> Path:
    """Get the application directory (works for both script and frozen exe)."""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        return Path(sys.executable).parent
    else:
        # Running as script - go up from src/utils to project root
        return Path(__file__).parent.parent.parent

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    # Load and validate the application configuration from YAML file.
    #
    # Args:
    #     config_path: Path to the YAML configuration file (default: config.yaml)
    #                  If relative, resolved from application directory
    #
    # Returns:
    #     Dict containing the parsed configuration
    #
    # Raises:
    #     ConfigError: If the config file is not found or is invalid
    
    try:
        # Resolve config path relative to app directory if not absolute
        config_file = Path(config_path)
        if not config_file.is_absolute():
            app_dir = _get_app_dir()
            config_file = app_dir / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {config_path}. Searched: {config_file if 'config_file' in locals() else 'unknown'}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")

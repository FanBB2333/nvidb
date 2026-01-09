"""
nvidb configuration module.
Centralized configuration for working directory and other settings.
"""
import os
from pathlib import Path

# Package version
VERSION = "1.6.6"

# Default working directory for nvidb data (config, logs, database)
# Can be overridden by NVIDB_HOME environment variable
WORKING_DIR = Path(os.environ.get('NVIDB_HOME', '~/.nvidb')).expanduser()

def get_config_path():
    """Get the path to the config file."""
    return WORKING_DIR / 'config.yml'

def get_db_path():
    """Get the default path to the SQLite database."""
    return WORKING_DIR / 'gpu_log.db'

def ensure_working_dir():
    """Ensure the working directory exists."""
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

"""
nvidb configuration module.
Centralized configuration for working directory and other settings.
"""
import os

# Default working directory for nvidb data (config, logs, database)
# Can be overridden by NVIDB_HOME environment variable
WORKING_DIR = os.path.expanduser(os.environ.get('NVIDB_HOME', '~/.nvidb'))

def get_config_path():
    """Get the path to the config file."""
    return os.path.join(WORKING_DIR, 'config.yml')

def get_db_path():
    """Get the default path to the SQLite database."""
    return os.path.join(WORKING_DIR, 'gpu_log.db')

def ensure_working_dir():
    """Ensure the working directory exists."""
    os.makedirs(WORKING_DIR, exist_ok=True)

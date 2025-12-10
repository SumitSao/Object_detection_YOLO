"""
Authentication Package
Provides user authentication and management functionality.
"""

from auth.user_manager import UserManager
from auth.auth_ui import AuthUI

__all__ = ['UserManager', 'AuthUI']

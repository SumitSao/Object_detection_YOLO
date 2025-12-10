"""
User Manager Module
Handles user authentication, registration, and database operations using SQLite.
"""

import sqlite3
import bcrypt
import os
from datetime import datetime
from typing import Optional, Tuple


class UserManager:
    """Manages user authentication and database operations."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize UserManager with database path.
        
        Args:
            db_path: Path to SQLite database file. Defaults to auth/users.db
        """
        if db_path is None:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, 'users.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database and create users table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password as string
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password to verify
            password_hash: Stored password hash
            
        Returns:
            True if password matches, False otherwise
        """
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """
        Create a new user account.
        
        Args:
            username: Desired username
            email: User's email address
            password: Plain text password
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Validate input
        if not username or not email or not password:
            return False, "All fields are required"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        # Hash the password
        password_hash = self._hash_password(password)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            conn.commit()
            conn.close()
            
            return True, "Account created successfully!"
            
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return False, "Username already exists"
            elif 'email' in str(e):
                return False, "Email already registered"
            else:
                return False, "Error creating account"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str, Optional[dict]]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username or email
            password: Plain text password
            
        Returns:
            Tuple of (success: bool, message: str, user_data: dict or None)
        """
        if not username or not password:
            return False, "Username and password are required", None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if username is email or username
            cursor.execute('''
                SELECT id, username, email, password_hash, created_at
                FROM users
                WHERE username = ? OR email = ?
            ''', (username, username))
            
            user = cursor.fetchone()
            conn.close()
            
            if user is None:
                return False, "Invalid username or password", None
            
            user_id, username, email, password_hash, created_at = user
            
            # Verify password
            if self._verify_password(password, password_hash):
                user_data = {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'created_at': created_at
                }
                return True, "Login successful!", user_data
            else:
                return False, "Invalid username or password", None
                
        except Exception as e:
            return False, f"Error: {str(e)}", None
    
    def user_exists(self, username: str = None, email: str = None) -> bool:
        """
        Check if a user exists by username or email.
        
        Args:
            username: Username to check
            email: Email to check
            
        Returns:
            True if user exists, False otherwise
        """
        if not username and not email:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if username and email:
                cursor.execute('''
                    SELECT COUNT(*) FROM users
                    WHERE username = ? OR email = ?
                ''', (username, email))
            elif username:
                cursor.execute('''
                    SELECT COUNT(*) FROM users
                    WHERE username = ?
                ''', (username,))
            else:
                cursor.execute('''
                    SELECT COUNT(*) FROM users
                    WHERE email = ?
                ''', (email,))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception:
            return False
    
    def get_user_count(self) -> int:
        """
        Get the total number of registered users.
        
        Returns:
            Number of users in database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM users')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception:
            return 0

"""
Authentication UI Module
Provides Streamlit-based login and sign-up interface.
"""

import streamlit as st
from auth.user_manager import UserManager
import re
from typing import Tuple


class AuthUI:
    """Handles authentication UI components for Streamlit."""
    
    def __init__(self):
        """Initialize AuthUI with UserManager."""
        self.user_manager = UserManager()
        
        # Initialize session state
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'user_data' not in st.session_state:
            st.session_state.user_data = None
        if 'auth_page' not in st.session_state:
            st.session_state.auth_page = 'login'
    
    def _validate_email(self, email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one number"
        
        return True, "Password is strong"
    
    def show_login_page(self):
        """Display the login page."""
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔐 Login</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Welcome back! Please login to your account.</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username and password:
                    success, message, user_data = self.user_manager.authenticate_user(username, password)
                    
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_data = user_data
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Don't have an account? Sign Up", use_container_width=True):
                st.session_state.auth_page = 'signup'
                st.rerun()
    
    def show_signup_page(self):
        """Display the sign-up page."""
        st.markdown("<h1 style='text-align: center; color: #2196F3;'>📝 Create Account</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Join us! Create your account to get started.</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        with st.form("signup_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
            
            # Password requirements
            with st.expander("Password Requirements"):
                st.markdown("""
                - At least 6 characters long
                - At least one uppercase letter
                - At least one lowercase letter
                - At least one number
                """)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_button:
                # Validation
                if not all([username, email, password, confirm_password]):
                    st.warning("Please fill in all fields")
                elif not self._validate_email(email):
                    st.error("Please enter a valid email address")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Validate password strength
                    is_valid, message = self._validate_password(password)
                    if not is_valid:
                        st.error(message)
                    else:
                        # Create user
                        success, message = self.user_manager.create_user(username, email, password)
                        
                        if success:
                            st.success(message)
                            st.info("Please login with your credentials")
                            st.session_state.auth_page = 'login'
                            st.rerun()
                        else:
                            st.error(message)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Already have an account? Login", use_container_width=True):
                st.session_state.auth_page = 'login'
                st.rerun()
    
    def show_auth_page(self):
        """
        Display the appropriate authentication page based on session state.
        
        Returns:
            True if user is logged in, False otherwise
        """
        if st.session_state.logged_in:
            return True
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.session_state.auth_page == 'login':
            self.show_login_page()
        else:
            self.show_signup_page()
        
        return False
    
    def logout(self):
        """Logout the current user."""
        st.session_state.logged_in = False
        st.session_state.user_data = None
        st.session_state.auth_page = 'login'
        st.rerun()
    
    def show_logout_button(self):
        """Display logout button in sidebar."""
        if st.session_state.logged_in:
            with st.sidebar:
                st.markdown("---")
                st.markdown(f"**Logged in as:** {st.session_state.user_data['username']}")
                if st.button("🚪 Logout", use_container_width=True):
                    self.logout()

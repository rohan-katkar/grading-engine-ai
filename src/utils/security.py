import hashlib
import secrets

def hash_password(password: str) -> str:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with a unique random salt.
    Format stored: salt$hash
    """
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    )
    return f"{salt}${key.hex()}"

def verify_password(stored_password_hash: str, provided_password: str) -> bool:
    """
    Verifies a provided plaintext password against the stored salt$hash string.
    """
    try:
        salt, key_hex = stored_password_hash.split("$")
        new_key = hashlib.pbkdf2_hmac(
            'sha256', 
            provided_password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        )
        return secrets.compare_digest(new_key.hex(), key_hex)
    except (ValueError, AttributeError):
        return False
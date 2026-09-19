import pytest

from src.utils.security import hash_password, verify_password


def test_password_hash_round_trip_and_unique_salts():
    first_hash = hash_password("correct horse battery staple")
    second_hash = hash_password("correct horse battery staple")

    assert first_hash != second_hash
    assert verify_password(first_hash, "correct horse battery staple")
    assert not verify_password(first_hash, "wrong password")


@pytest.mark.parametrize("malformed_hash", ["", "missing-separator", "a$b$c", None])
def test_malformed_password_hash_is_rejected(malformed_hash):
    assert not verify_password(malformed_hash, "password")
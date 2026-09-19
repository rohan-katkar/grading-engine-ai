from src.utils import sanitizer


def test_sanitizer_redacts_regex_pii_without_requiring_nlp_results(monkeypatch):
    class EmptyAnalyzer:
        def analyze(self, **kwargs):
            return []

    monkeypatch.setattr(sanitizer, "analyzer", EmptyAnalyzer())

    clean, pii_found = sanitizer.sanitize_student_answer(
        "Contact me at alex@example.com or 555-123-4567. Student ID: 123456.",
    )

    assert pii_found is True
    assert "alex@example.com" not in clean
    assert "555-123-4567" not in clean
    assert "[REDACTED_ID]" in clean


def test_domain_allowlist_extracts_reference_terms():
    allowlist = sanitizer.extract_domain_allowlist(
        "Jonas Salk developed the polio vaccine."
    )

    assert "Jonas Salk" in allowlist
    assert "Jonas" in allowlist
    assert "Salk" in allowlist
"""Tests for the PatternDetector class"""

from pandas import Series

from pandera_forge import PatternDetector


class TestPatternDetector:
    """Test suite for PatternDetector"""

    def test_email_pattern_detection(self):
        """Test detection of email addresses"""
        series = Series(
            ["user@example.com", "admin@test.org", "contact@company.co.uk", "support@domain.net"]
        )

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "email"

    def test_url_pattern_detection(self):
        """Test detection of URLs"""
        series = Series(
            [
                "https://example.com",
                "http://test.org",
                "https://www.company.com/page",
                "http://subdomain.site.net",
            ]
        )

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "url"

    def test_uuid_pattern_detection(self):
        """Test detection of UUIDs"""
        series = Series(
            [
                "123e4567-e89b-12d3-a456-426614174000",
                "987fcdeb-51a2-43d1-9876-543210fedcba",
                "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            ]
        )

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "uuid"

    def test_ipv4_pattern_detection(self):
        """Test detection of IPv4 addresses"""
        series = Series(["192.168.1.1", "10.0.0.1", "172.16.0.1", "8.8.8.8"])

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "ipv4"

    def test_date_iso_pattern_detection(self):
        """Test detection of ISO format dates"""
        series = Series(["2023-01-01", "2023-12-31", "2024-06-15", "2022-03-20"])

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "date_iso"

    def test_no_pattern_detection(self):
        """Test when no pattern is detected"""
        series = Series(
            ["random string 1", "another value 2", "something else 3", "no pattern here 4"]
        )

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is None

    def test_partial_pattern_detection(self):
        """Test when pattern matches less than threshold"""
        series = Series(["user@example.com", "not an email", "another non-email", "random string"])

        # With default threshold (0.9), this should not detect email pattern
        pattern_result = PatternDetector.detect_pattern(series, min_match_ratio=0.9)
        assert pattern_result is None

        # With lower threshold, it should detect
        pattern_result = PatternDetector.detect_pattern(series, min_match_ratio=0.2)
        assert pattern_result is not None
        assert pattern_result[0] == "email"

    def test_string_constraints_inference(self):
        """Test inference of string constraints"""
        series = Series(["ABC123", "ABC456", "ABC789"])

        constraints = PatternDetector.infer_string_constraints(series)

        assert constraints.min_length
        assert constraints.max_length
        assert constraints.min_length == 6
        assert constraints.max_length == 6

    def test_string_constraints_with_prefix(self):
        """Test detection of common prefix"""
        series = Series(["PREFIX_value1", "PREFIX_value2", "PREFIX_value3"])

        constraints = PatternDetector.infer_string_constraints(series)

        assert constraints.starts_with
        assert constraints.starts_with == "PRE"

    def test_string_constraints_with_suffix(self):
        """Test detection of common suffix"""
        series = Series(["value1_SUFFIX", "value2_SUFFIX", "value3_SUFFIX"])

        constraints = PatternDetector.infer_string_constraints(series)

        assert constraints.ends_with
        assert constraints.ends_with == "FIX"

    def test_empty_series_handling(self):
        """Test handling of empty series"""
        series = Series([], dtype=object)

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is None

        constraints = PatternDetector.infer_string_constraints(series)
        assert constraints.model_dump(exclude_unset=True, exclude_none=True) == {}

    def test_null_values_handling(self):
        """Test handling of series with null values"""
        series = Series(["user@example.com", None, "admin@test.org", None, "contact@company.com"])

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "email"

    def test_numeric_string_pattern(self):
        """Test detection of numeric strings"""
        series = Series(["12345", "67890", "11111", "99999"])

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "numeric_string"

    def test_alphanumeric_pattern(self):
        """Test detection of alphanumeric strings"""
        series = Series(["ABC123", "XYZ789", "TEST456", "CODE001"])

        pattern_result = PatternDetector.detect_pattern(series)
        assert pattern_result is not None
        assert pattern_result[0] == "alphanumeric"

    def test_custom_regex_generation(self):
        """Test custom regex pattern generation"""
        series = Series(["A-123", "B-456", "C-789", "D-012"])

        custom_pattern = PatternDetector.generate_custom_regex(series)
        assert custom_pattern is not None
        # Should generate pattern like "^[A-Z]-\d\d\d$"
        assert "[A-Z]" in custom_pattern
        assert r"\d" in custom_pattern

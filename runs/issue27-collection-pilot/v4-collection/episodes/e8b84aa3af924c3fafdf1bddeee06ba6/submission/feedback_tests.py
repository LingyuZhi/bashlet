def tests():
    def check(result, expected):
        assert result == expected, f"Failed: expected {expected!r}, got {result!r}"

    s1 = "aabaabc"
    s1_expected = "a"
    check(first_repeated_char(s1), s1_expected)

    s2 = "abcdefgh"
    s2_expected = None
    check(first_repeated_char(s2), s2_expected)

    s3 = "aaa"
    s3_expected = "a"
    check(first_repeated_char(s3), s3_expected)

    s4 = "hello"
    s4_expected = "l"
    check(first_repeated_char(s4), s4_expected)

    print("All tests passed!")

if __name__ == "__main__":
    from solution import first_repeated_char
    tests()

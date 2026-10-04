def first_repeated_char(str1):
    """
    Find the first repeated character in a given string.
    
    Args:
        str1: Input string to search for repeated character
        
    Returns:
        The first character that appears more than once in the string,
        moving from left to right, or None if no such character exists.
    """
    seen = set()
    for char in str1:
        if char in seen:
            return char
        seen.add(char)
    return None

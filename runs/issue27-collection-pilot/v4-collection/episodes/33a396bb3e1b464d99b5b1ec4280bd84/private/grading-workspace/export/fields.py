def csv_field(text):
    needs_quote = ',' in text or '"' in text or '\n' in text
    if needs_quote:
        # Double any embedded quotes and wrap in quotes
        return '"' + text.replace('"', '\"\"') + '"'
    return text

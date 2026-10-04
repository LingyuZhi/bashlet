from export.fields import csv_field
assert csv_field(*['a,b']) == '"a,b"'
print('feedback passed')

import re

re_whitespace = re.compile(r'\s+', re.MULTILINE)
re_multiple_whitespace = re.compile(r'  +', re.MULTILINE)
re_paragraph = re.compile(r'\n{2,}', re.MULTILINE)
re_line_punctuation = re.compile(r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|\||~|»|«|“|”|-|_)+$', re.MULTILINE)
re_line_punctuation_wo_fs = re.compile(r'^(?:!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|\||~|»|«|“|”|-|_)+$', re.MULTILINE)
re_line_punctuation_wo_underscore = re.compile(r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|`|\{|\||\}|\||~|»|«|“|”|-)+$', re.MULTILINE)
re_ds_punctuation_wo_underscore = re.compile(r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|`|\{|\||\}|\||~|»|«|“|”|-)+')
re_fullstop = re.compile(r'^(?:\.)+$', re.MULTILINE)
re_newline_in_text = re.compile(r'(?<=\w)\n(?=\w)', re.MULTILINE)
re_incomplete_sentence_at_end = re.compile(r'(?<=\.)[^\.]+$', re.DOTALL)
re_item_element  = r'(?:-|\. |\*|•|\d+ |\d+\.|\d\)|\(\d+\)|\d\)\.|o |# )'
re_heading_general = r'[^\.\:\n]*(?::\n{1,2}|\?\n{1,2}|[^,]\n)'
import re
test_text = "abc"
test_pattern = r"(b)"
result = re.split(test_pattern, test_text)
print(result) # Expected: ['a', 'b', 'c']

# Try a more complex pattern like yours
test_text_complex = "--- Page 1 --- In case of discrepancies between the French and the English text. Luxembourg, 23 August 2018 To all investment fund managers and entities carrying out the activity of registrar agent CIRCULAR CSSF 18/698 Re: Authorisation"
test_patterns = [r'(Article\s+\d+[a-z]?)', r'(Section\s+\d+(?:\.\d+)?)', r'(Chapter\s+\d+)', r'(Part\s+[IVXLC]+)', r'(Regulation\s+\d+)', r'(Directive\s+\d+)', r'(\d+\.\s+[A-Z][^.]*)', r'([A-Z][A-Z\s]{10,})']
test_combined_pattern = '|'.join(test_patterns)
result_complex = re.split(f'({test_combined_pattern})', test_text_complex, flags=re.IGNORECASE)
print(result_complex)
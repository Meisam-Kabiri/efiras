#!/usr/bin/env python3
"""
Text Processing Utilities
Common text processing functions used across the project
"""

import re
from typing import List


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text based on punctuation and formatting patterns.
    
    Rules:
    - Split on '.' followed by space and uppercase/list patterns
    - Split on '\n' followed by uppercase/list patterns  
    - List patterns: (a), a), 1., 1), 1-, 1_
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Clean up extra whitespace but keep structure
    text = re.sub(r'[ \t]+', ' ', text.strip())
    
    # Find all split positions
    sentences = []
    current_pos = 0
    
    # Pattern: . followed by space and sentence starter OR \n followed by sentence starter
    pattern = r'(\.\s+(?=[A-Z]|\([a-z]\)|[a-z]\)|\d+[\.\)\-_])|\n\s*(?=[A-Za-z]|\([a-z]\)|[a-z]\)|\d+[\.\)\-_])|\s+(?=\d+\))|\s+(?=\(\d+\)))'
    
    prev_sentence = ''
    for match in re.finditer(pattern, text):
        # Add text before the split
        sentence = prev_sentence+text[current_pos:match.start()].strip()
        if re.match(r'^\d+[.,:;!?]?\s*$', sentence): ## only-number sentence
            prev_sentence += sentence+' '
        else:
            sentences.append(sentence)
            prev_sentence = ''
        current_pos = match.end() - 1  # Keep the starter character
    
    # Add remaining text
    if current_pos < len(text):
        sentence = text[current_pos:].strip()
        if sentence:
            sentences.append(sentence)
    
    return sentences


def clean_whitespace(text: str) -> str:
    """
    Clean and normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_newlines(text: str) -> str:
    """
    Remove all newlines from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without newlines
    """
    if not text:
        return ""
    
    return re.sub(r'\n+', ' ', text)


def is_list_item(text: str) -> bool:
    """
    Check if text starts with a list pattern.
    
    Patterns: (a), a), 1., 1), 1-, 1_
    
    Args:
        text: Text to check
        
    Returns:
        True if text starts with list pattern
    """
    if not text:
        return False
    
    return bool(re.match(r'^(\([a-z]\)|[a-z]\)|\d+[\.\)\-_])', text.strip()))


def is_definition(text: str) -> bool:
    """
    Check if text looks like a definition.
    
    Pattern: number) "word"
    
    Args:
        text: Text to check
        
    Returns:
        True if text looks like a definition
    """
    if not text:
        return False
    
    return bool(re.match(r'^\d+\)\s*"', text.strip()))


def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text by splitting on double newlines or newline-space-newline patterns.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    if not text:
        return []
    
    # Split on \n\n or \n \n patterns
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    # Clean each paragraph and filter out empty ones
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if cleaned:
            cleaned_paragraphs.append(cleaned)
    
    return cleaned_paragraphs


# Test function
def test_functions():
    """Test the utility functions"""
    
    # test_text = 'First sentence. Second sentence. 1) First item 2) Second item. (a) sub item'
    test_text = """
    1. For the purposes of this circular:\n1) “EBA” means the European Banking Authority. 2) “EIOPA” means the European Insurance and Occupational Pensions Authority. 3) “ESMA” means the European Securities and Markets Authority. 4) “ML/TF” means money laundering and terrorist financing. 5) “Circular CSSF 07/290” means Circular CSSF 07/290, as amended by Circular CSSF 10/451 on the definition of capital ratios pursuant to Article 56 of the Law of 5 April 1993 on the financial sector, as amended, (the circular is currently being updated). 6) “Circular CSSF 17/661” means Circular CSSF 17/661 adopting the joint guidelines issued by the three European Supervisory Authorities (EBA/ESMA/EIOPA) on money laundering and terrorist financing risk factors. 7) “Circular CSSF 11/512” means Circular 11/512 presenting the main regulatory changes in risk management following the publication of CSSF Regulation 10-04 and ESMA clarifications, laying down further clarifications from the CSSF on risk management rules and defining the content and format of the risk management process to be communicated to the CSSF. 8) “CRR” means Regulation (EU) No 575/2013 of 26 June 2013 on prudential requirements for credit institutions and investment firms and amending Regulation (EU) No 648/2012. 9) “delegate” means any third party carrying out on behalf of an IFM: \n one or more functions included in the activity of collective portfolio management as defined in Annex II of the 2010 Law as well as part of the risk management activities in accordance with point 222 or functions included in Annex I of the 2013 Law, respectively. \n for an AIFM, the external valuer. 10) “AIFMD” means Directive 2011/61/EU of the European Parliament and of the Council of 8 June 2011 on Alternative Investment Fund Managers. 11) “UCITS Directive” means Directive 2009/65/EC of the European Parliament and of the Council of 13 July 2009 on the coordination of laws, regulations and administrative provisions relating to undertakings for collective investment in transferable securities (UCITS). 12) “FTE” means full-time equivalent. 13) “AIF” means an alternative investment fund as defined in Article 1 of the 2013 Law including the European long-term investment fund (ELTIF), the European social entrepreneurship fund (EuSEF) and the European venture capital fund (EuVECA). 14) “FIAAG” means a self-managed alternative investment fund: internally managed AIF within the meaning of point (b) of Article 4(1) of the 2013 Law. 15) “key functions” means functions included in the activity of collective portfolio management as defined in Annex II of the 2010 Law or functions included in Annex I of the 2013 Law, respectively, including monitoring delegates of the above-mentioned functions, permanent compliance, risk management and internal audit functions as well as, the valuation function for the AIFM. 16) “required own funds” means \n the own funds required under Articles 101(4) and 102(1)(a) of the 2010 Law as well as Article 8 of the 2013 Law. \n where appropriate, the own funds referred to in Articles 12 to 15 of Delegated Regulation (EU) 231/2013 for an AIFM. \n where the IFM is also authorised to provide the services referred to in Article 101(3) of the 2010 Law and/or in Article 5(4) of the 2013 Law, the own funds required under Circular CSSF 07/290. 17) “AIFM” means an alternative investment fund manager authorised under Chapter 2 of the 2013 Law.
    """

    # test_text = "hello my name is: Meisam 1)this is the fist item \n and here I am"
    
    # print("Test text:", test_text)
    # print("Sentences:", extract_sentences(test_text))
    # print("Is '1) item' a list item:", is_list_item('1) item'))
    # print("Is '1) \"word\"' a definition:", is_definition('1) "word"'))

    sentences = extract_sentences(test_text)
    for i, sentence in enumerate(sentences):
        print(f"{i}th sentence is: {sentence} \n -------------------------------------- \n")


if __name__ == "__main__":
    test_functions()
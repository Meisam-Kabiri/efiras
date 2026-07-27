"""
buckets_config.py
The two parsing buckets for the 23-document regulatory corpus.

Bucket A: EUR-Lex HTML, tree-walked like fetch_chunk.py (deterministic, no LLM needed for hierarchy).
Bucket B: flat text - PDF or plain <pre> HTML (regex-on-lines + LLM stack tracking for hierarchy).

Every URL below was fetched and verified (HTTP 200 + title/content matched against the
expected document) on 2026-07-04, not guessed from memory. Two exceptions found during
verification, called out where they occur:

  - SOLVENCY2_L2: EUR-Lex serves the JS-only "eurlex-frontoffice" shell for this CELEX on
    the normal TXT/HTML endpoint (curl gets an empty app shell, no article text at all -
    confirmed with and without cookies). Using the TXT/PDF endpoint instead, which does
    return the real 797-page document. format = "pdf", not "html", for this one entry.
  - DODD_FRANK_P2: there is no second Dodd-Frank document. "Part 2" is the same Public Law
    111-203 text as DODD_FRANK, just chunked as a second pass (e.g. later titles). Same URL,
    do not fetch it twice.

2026-07-09 update: added 4 gap-fill EU frameworks to BUCKET_A (DORA, MAR, MIFID2_L2, AMLR),
each verified the same way -- real "id=\"art_N\"" EUR-Lex markup confirmed present, not just
that the page loads. Also dropped BASEL2 and both DODD_FRANK entries from BUCKET_B -- see the
comment above BUCKET_B for why.

Each tuple: (doc_id, name, url, format)
"""

BUCKET_A = (
    ("MIFID2", "Markets in Financial Instruments Directive II",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32014L0065", "html"),
    ("AIFMD", "Alternative Investment Fund Managers Directive",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32011L0061", "html"),
    ("GDPR", "General Data Protection Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679", "html"),
    ("SFDR", "Sustainable Finance Disclosure Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R2088", "html"),
    ("CRR", "Capital Requirements Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02013R0575-20260101", "html"),
    ("5AMLD", "Fifth Anti-Money Laundering Directive",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32018L0843", "html"),
    ("PSD2", "Payment Services Directive 2",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32015L2366", "html"),
    ("SOLVENCY2", "Solvency II Directive",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32009L0138", "html"),
    ("EMIR", "European Market Infrastructure Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32012R0648", "html"),
    ("UCITS", "Undertakings for Collective Investment in Transferable Securities",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32009L0065", "html"),
    ("CRD5", "Capital Requirements Directive V",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02013L0036-20240729", "html"),
    ("MIFIR", "Markets in Financial Instruments Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32014R0600", "html"),
    ("SFTR", "Securities Financing Transactions Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32015R2365", "html"),
    ("4AMLD", "Fourth Anti-Money Laundering Directive",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32015L0849", "html"),
    ("EUTAXONOMY", "EU Taxonomy Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32020R0852", "html"),
    ("AIFMD_L2", "AIFMD Level 2 Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32013R0231", "html"),
    # -- added 2026-07-09, gap-fill picks, verified real id="art_N" markup present --
    ("DORA", "Digital Operational Resilience Act",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554", "html"),
    ("MAR", "Market Abuse Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02014R0596-20241204", "html"),
    ("MIFID2_L2", "MiFID II Delegated Regulation (org. requirements, product governance, costs & charges)",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02017R0565-20220802", "html"),
    # not yet applicable (from 2027-07-10) -- adopted/in force now, kept for forward coverage
    ("AMLR", "EU Anti-Money Laundering Regulation",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1624", "html"),
)

# Removed 2026-07-09:
#   - BASEL2 ("International Convergence of Capital Measurement..."): fully superseded by
#     Basel III, and CRR (BUCKET_A) already carries the actually-binding EU-law version of the
#     relevant capital mechanics -- a superseded, non-binding 347-page international standard
#     text was low real query value for the processing cost.
#   - DODD_FRANK / DODD_FRANK_P2: dropped as out of scope -- the only non-EU/non-global
#     framework in the whole 23-doc list. DODD_FRANK_P2 was never a real second document to
#     begin with (see the note above).
BUCKET_B = (
    ("BASEL3", "Basel III: International Framework",
     "https://www.bis.org/publ/bcbs189.pdf", "pdf"),
    ("FATF2012", "FATF Recommendations 2012",
     "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/FATF%20Recommendations%202012.pdf.coredownload.inline.pdf", "pdf"),
    ("CSSF_18_698", "CSSF Circular 18/698",
     "https://www.cssf.lu/wp-content/uploads/cssf18_698eng.pdf", "pdf"),
)

BUCKETS = (
    ("A", BUCKET_A),
    ("B", BUCKET_B),
)

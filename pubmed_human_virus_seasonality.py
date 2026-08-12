#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:31:40 2026

@author: jaanajurvansuu

Search PubMed for publications explicitly relevant to seasonality or recurring
temporal patterns for a fixed list of human viruses.

Important:
- A publication is retained only when a virus name/alias and a seasonality
  concept occur in the SAME title/abstract sentence.
- The matched sentence is classified as positive, negative, or unclear using
  transparent rule-based language patterns.
- "Negative" means the sentence explicitly reports absent/lacking seasonality.
- "Positive" requires explicit language that a seasonal pattern was observed.
- "Unclear" includes methods, hypotheses, or ambiguous statements.
- The matched sentence is written to the output for manual audit.
- No scores, weights, or JSON output files are produced.

Outputs
-------
1. virus_seasonality_publications.csv
   Columns:
   virus, pmid, year, title, relation, matched_sentence

2. virus_seasonality_summary.csv
   Columns:
   virus, relevant_publications, positive, negative, unclear
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import requests


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# Fixed list of 37 human viruses.
VIRUSES = ['Alphapapillomavirus 1',
 'Alphapapillomavirus 11',
 'Alphapapillomavirus 4',
 'Alphapapillomavirus 9',
 'Betacoronavirus 1',
 'Betapapillomavirus 3',
 'Betapolyomavirus secuhominis',
 'Bocaparvovirus primate1',
 'Bocaparvovirus primate2',
 'Cytomegalovirus humanbeta5',
 'Deltapolyomavirus sextihominis',
 'Enterovirus A',
 'Enterovirus B',
 'Enterovirus C',
 'Gammapapillomavirus 1',
 'Gammapapillomavirus 7',
 'HMO Astrovirus A',
 'Hepatitis E virus',
 'Human circovirus VS6600022',
 'Human coronavirus NL63',
 'Human mastadenovirus A',
 'Human mastadenovirus B',
 'Human mastadenovirus D',
 'Human mastadenovirus F',
 'Mupapillomavirus 1',
 'Mupapillomavirus 2',
 'Rhinovirus A',
 'Rhinovirus B',
 'Rhinovirus C',
 'Rotavirus A',
 'Salivirus FHB',
 'Sapovirus Sapozj-9',
 'Sapporo virus',
 'Severe acute respiratory syndrome-related coronavirus',
 'Simplexvirus humanalpha2',
 'TTV-like mini virus',
 'Torque teno virus']


# PubMed aliases used to find literature under common or historical names.
# Standalone "CMV" is intentionally excluded; HCMV is used for human
# cytomegalovirus.
ALIASES: dict[str, list[str]] = {'Alphapapillomavirus 1': ['Alphapapillomavirus 1',
                           'human papillomavirus 32',
                           'human papillomavirus type 32',
                           'HPV32',
                           'HPV-32',
                           'human papillomavirus 42',
                           'human papillomavirus type 42',
                           'HPV42',
                           'HPV-42'],
 'Alphapapillomavirus 4': ['Alphapapillomavirus 4',
                           'human papillomavirus 2',
                           'HPV2',
                           'HPV-2',
                           'human papillomavirus 3',
                           'HPV3',
                           'HPV-3',
                           'human papillomavirus 10',
                           'HPV10',
                           'HPV-10',
                           'human papillomavirus 27',
                           'HPV27',
                           'HPV-27',
                           'human papillomavirus 28',
                           'HPV28',
                           'HPV-28',
                           'human papillomavirus 29',
                           'HPV29',
                           'HPV-29',
                           'human papillomavirus 57',
                           'HPV57',
                           'HPV-57',
                           'human papillomavirus 77',
                           'HPV77',
                           'HPV-77',
                           'human papillomavirus 94',
                           'HPV94',
                           'HPV-94',
                           'human papillomavirus 117',
                           'HPV117',
                           'HPV-117',
                           'human papillomavirus 125',
                           'HPV125',
                           'HPV-125'],
 'Alphapapillomavirus 9': ['Alphapapillomavirus 9',
                           'human papillomavirus 16',
                           'human papillomavirus type 16',
                           'HPV16',
                           'HPV-16',
                           'human papillomavirus 31',
                           'human papillomavirus type 31',
                           'HPV31',
                           'HPV-31',
                           'human papillomavirus 33',
                           'human papillomavirus type 33',
                           'HPV33',
                           'HPV-33',
                           'human papillomavirus 35',
                           'human papillomavirus type 35',
                           'HPV35',
                           'HPV-35',
                           'human papillomavirus 52',
                           'human papillomavirus type 52',
                           'HPV52',
                           'HPV-52',
                           'human papillomavirus 58',
                           'human papillomavirus type 58',
                           'HPV58',
                           'HPV-58'],
 'Alphapapillomavirus 11': ['Alphapapillomavirus 11',
                            'human papillomavirus 34',
                            'human papillomavirus type 34',
                            'HPV34',
                            'HPV-34',
                            'human papillomavirus 73',
                            'human papillomavirus type 73',
                            'HPV73',
                            'HPV-73'],
 'Betacoronavirus 1': ['Betacoronavirus 1', 'human coronavirus OC43', 'HCoV-OC43', 'coronavirus OC43'],
 'Betapapillomavirus 3': ['Betapapillomavirus 3',
                          'human papillomavirus 49',
                          'human papillomavirus type 49',
                          'HPV49',
                          'HPV-49',
                          'human papillomavirus 75',
                          'human papillomavirus type 75',
                          'HPV75',
                          'HPV-75',
                          'human papillomavirus 76',
                          'human papillomavirus type 76',
                          'HPV76',
                          'HPV-76'],
 'Betapolyomavirus secuhominis': ['Betapolyomavirus secuhominis',
                                  'human polyomavirus 2',
                                  'JC polyomavirus',
                                  'JC virus',
                                  'JCPyV'],
 'Bocaparvovirus primate1': ['Bocaparvovirus primate1', 'human bocavirus 1', 'HBoV1'],
 'Bocaparvovirus primate2': ['Bocaparvovirus primate2', 'human bocavirus 2', 'HBoV2'],
 'Cytomegalovirus humanbeta5': ['Cytomegalovirus humanbeta5',
                                'human cytomegalovirus',
                                'human herpesvirus 5',
                                'HHV-5',
                                'HCMV'],
 'Deltapolyomavirus sextihominis': ['Deltapolyomavirus sextihominis', 'human polyomavirus 6', 'HPyV6'],
 'Enterovirus A': ['Enterovirus A', 'human enterovirus A', 'enterovirus species A', 'HEV-A'],
 'Enterovirus B': ['Enterovirus B', 'human enterovirus B', 'enterovirus species B', 'HEV-B'],
 'Enterovirus C': ['Enterovirus C', 'human enterovirus C', 'enterovirus species C', 'HEV-C'],
 'Gammapapillomavirus 1': ['Gammapapillomavirus 1',
                           'human papillomavirus 4',
                           'human papillomavirus type 4',
                           'HPV4',
                           'HPV-4',
                           'human papillomavirus 65',
                           'human papillomavirus type 65',
                           'HPV65',
                           'HPV-65'],
 'Gammapapillomavirus 7': ['Gammapapillomavirus 7',
                           'human papillomavirus 109',
                           'human papillomavirus type 109',
                           'HPV109',
                           'HPV-109',
                           'human papillomavirus 123',
                           'human papillomavirus type 123',
                           'HPV123',
                           'HPV-123'],
 'HMO Astrovirus A': ['HMO Astrovirus A', 'HMO-A astrovirus', 'human mink ovine-like astrovirus'],
 'Hepatitis E virus': ['hepatitis E virus'],
 'Human circovirus VS6600022': ['Human circovirus VS6600022', 'human circovirus', 'HCirV'],
 'Human coronavirus NL63': ['human coronavirus NL63', 'HCoV-NL63', 'coronavirus NL63'],
 'Human mastadenovirus A': ['Human mastadenovirus A',
                            'human adenovirus A',
                            'adenovirus type 12',
                            'adenovirus 12',
                            'HAdV-12',
                            'adenovirus type 18',
                            'adenovirus 18',
                            'HAdV-18',
                            'adenovirus type 31',
                            'adenovirus 31',
                            'HAdV-31',
                            'adenovirus type 61',
                            'adenovirus 61',
                            'HAdV-61'],
 'Human mastadenovirus B': ['Human mastadenovirus B', 'human adenovirus B'],
 'Human mastadenovirus D': ['Human mastadenovirus D',
                            'human adenovirus D',
                            'adenovirus type 8',
                            'HAdV-8',
                            'adenovirus type 19',
                            'HAdV-19',
                            'adenovirus type 37',
                            'HAdV-37',
                            'adenovirus type 53',
                            'HAdV-53',
                            'adenovirus type 54',
                            'HAdV-54',
                            'adenovirus type 56',
                            'HAdV-56',
                            'adenovirus type 64',
                            'HAdV-64'],
 'Human mastadenovirus F': ['Human mastadenovirus F',
                            'human adenovirus F',
                            'adenovirus type 40',
                            'adenovirus 40',
                            'HAdV-40',
                            'adenovirus type 41',
                            'adenovirus 41',
                            'HAdV-41'],
 'Mupapillomavirus 1': ['Mupapillomavirus 1',
                        'human papillomavirus 1',
                        'human papillomavirus type 1',
                        'HPV1',
                        'HPV-1'],
 'Mupapillomavirus 2': ['Mupapillomavirus 2',
                        'human papillomavirus 63',
                        'human papillomavirus type 63',
                        'HPV63',
                        'HPV-63'],
 'Rhinovirus A': ['Rhinovirus A', 'human rhinovirus A', 'HRV-A'],
 'Rhinovirus B': ['Rhinovirus B', 'human rhinovirus B', 'HRV-B'],
 'Rhinovirus C': ['Rhinovirus C', 'human rhinovirus C', 'HRV-C'],
 'Rotavirus A': ['Rotavirus A', 'group A rotavirus', 'human rotavirus A'],
 'Salivirus FHB': ['Salivirus FHB', 'human salivirus', 'salivirus', 'klassevirus'],
 'Sapovirus Sapozj-9': ['Sapovirus Sapozj-9', 'Sapozj-9'],
 'Sapporo virus': ['Sapporo virus', 'human sapovirus', 'sapovirus'],
 'Severe acute respiratory syndrome-related coronavirus': ['Severe acute respiratory syndrome-related '
                                                           'coronavirus',
                                                           'SARS-related coronavirus',
                                                           'SARS coronavirus',
                                                           'SARS-CoV',
                                                           'SARS-CoV-1',
                                                           'SARS-CoV-2',
                                                           '2019-nCoV'],
 'Simplexvirus humanalpha2': ['Simplexvirus humanalpha2',
                              'human alphaherpesvirus 2',
                              'herpes simplex virus type 2',
                              'herpes simplex virus 2',
                              'HSV-2'],
 'TTV-like mini virus': ['TTV-like mini virus', 'torque teno mini virus', 'TTMV'],
 'Torque teno virus': ['torque teno virus', 'transfusion transmitted virus', 'TT virus']}


# Specific seasonality concepts only.
# Broad terms such as "winter", "summer", "surveillance", and "epidemiology"
# are intentionally excluded because they create many false positives.
SEASONALITY_TERMS = [
    "seasonality",
    "seasonal",
    "seasonal pattern",
    "seasonal patterns",
    "seasonal variation",
    "seasonal variations",
    "seasonal distribution",
    "seasonal distributions",
    "seasonal trend",
    "seasonal trends",
    "seasonal incidence",
    "seasonal prevalence",
    "seasonal circulation",
    "seasonal dynamics",
    "seasonal differences",
    "seasonal effect",
    "seasonal effects",
    "annual cycle",
    "annual cycles",
    "annual pattern",
    "annual patterns",
    "monthly pattern",
    "monthly patterns",
    "monthly distribution",
    "monthly distributions",
    "year-round",
    "year round",
    "no seasonal pattern",
    "no seasonality",
    "nonseasonal",
    "non-seasonal",
]


class PubMedClient:
    def __init__(self) -> None:
        self.email = os.getenv("NCBI_EMAIL", "anonymous@example.com")
        self.api_key = os.getenv("NCBI_API_KEY")
        self.tool = os.getenv("NCBI_TOOL", "human_virus_seasonality_search")
        self.session = requests.Session()
        self.last_request_time = 0.0
        self.minimum_interval = 0.11 if self.api_key else 0.34

    def request(
        self,
        endpoint: str,
        params: dict,
        retries: int = 5,
    ) -> requests.Response:
        params = {
            **params,
            "email": self.email,
            "tool": self.tool,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(retries):
            delay = self.minimum_interval - (
                time.monotonic() - self.last_request_time
            )
            if delay > 0:
                time.sleep(delay)

            try:
                response = self.session.get(
                    f"{EUTILS_BASE}/{endpoint}",
                    params=params,
                    timeout=90,
                )
                self.last_request_time = time.monotonic()

                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"Temporary NCBI response: {response.status_code}",
                        response=response,
                    )

                response.raise_for_status()
                return response

            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError("NCBI request failed")

    def search(
        self,
        query: str,
        max_records: int,
    ) -> tuple[int, list[str]]:
        response = self.request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_records,
                "sort": "pub date",
            },
        )

        data = response.json()["esearchresult"]
        return int(data["count"]), data.get("idlist", [])

    def fetch(self, pmids: list[str]) -> list[dict]:
        records: list[dict] = []

        for start in range(0, len(pmids), 200):
            batch = pmids[start:start + 200]

            response = self.request(
                "efetch.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "xml",
                },
            )

            records.extend(parse_pubmed_xml(response.content))

        return records


def xml_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def extract_year(article: ET.Element) -> int | None:
    paths = [
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
        ".//PubMedPubDate[@PubStatus='entrez']/Year",
    ]

    for path in paths:
        value = xml_text(article.find(path))
        if value.isdigit():
            return int(value)

    medline_date = xml_text(
        article.find(".//JournalIssue/PubDate/MedlineDate")
    )
    match = re.search(r"\b(?:19|20)\d{2}\b", medline_date)
    return int(match.group()) if match else None


def parse_pubmed_xml(xml_content: bytes) -> list[dict]:
    root = ET.fromstring(xml_content)
    records: list[dict] = []

    for article in root.findall(".//PubmedArticle"):
        title = xml_text(article.find(".//ArticleTitle"))

        abstract_parts = [
            xml_text(node)
            for node in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(
            part for part in abstract_parts if part
        )

        records.append(
            {
                "pmid": xml_text(article.find(".//PMID")),
                "title": title,
                "abstract": abstract,
                "combined_text": f"{title} {abstract}",
                "year": extract_year(article),
            }
        )

    return records


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014-]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_tiab(term: str) -> str:
    clean = term.replace('"', "")
    return f'"{clean}"[Title/Abstract]'


def build_or_group(terms: Iterable[str]) -> str:
    return "(" + " OR ".join(
        quote_tiab(term) for term in terms
    ) + ")"


def build_virus_clause(virus: str) -> str:
    terms = list(
        dict.fromkeys(ALIASES.get(virus, [virus]))
    )
    return build_or_group(terms)


def build_pubmed_query(
    virus: str,
    year_from: int | None,
) -> str:
    query = (
        f"{build_virus_clause(virus)} "
        f"AND {build_or_group(SEASONALITY_TERMS)}"
    )

    if year_from:
        query += (
            f' AND ("{year_from}/01/01"[Date - Publication] '
            f': "3000"[Date - Publication])'
        )

    return query


def alias_patterns(alias: str) -> list[re.Pattern[str]]:
    normalized = normalize_text(alias)
    tokens = normalized.split()

    if not tokens:
        return []

    compact = "".join(tokens)
    spaced = r"\s*".join(
        re.escape(token) for token in tokens
    )

    patterns = [
        re.compile(
            rf"(?<![a-z0-9]){spaced}(?![a-z0-9])"
        )
    ]

    if len(compact) >= 4:
        patterns.append(
            re.compile(
                rf"(?<![a-z0-9]){re.escape(compact)}(?![a-z0-9])"
            )
        )

    return patterns


def sentence_split(text: str) -> list[str]:
    """
    Split title/abstract text into approximate sentences.

    PubMed abstracts are usually well punctuated. This deliberately simple
    splitter is transparent and avoids adding an NLP-package dependency.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if sentence.strip()
    ]


def sentence_mentions_virus(sentence: str, virus: str) -> bool:
    text = normalize_text(sentence)

    for alias in ALIASES.get(virus, [virus]):
        for pattern in alias_patterns(alias):
            if pattern.search(text):
                return True

    return False


def sentence_mentions_seasonality(sentence: str) -> bool:
    text = normalize_text(sentence)

    return any(
        normalize_text(term) in text
        for term in SEASONALITY_TERMS
    )


NEGATIVE_PATTERNS = [
    r"\bno\b.{0,45}\bseasonal(?:ity| pattern| variation| variations| effect| effects)?\b",
    r"\bnot\b.{0,45}\bseasonal(?:ity| pattern| variation| variations| effect| effects)?\b",
    r"\bwithout\b.{0,45}\bseasonal(?:ity| pattern| variation| variations)?\b",
    r"\black(?:ed|ing|s)?\b.{0,45}\bseasonal(?:ity| pattern| variation| variations)?\b",
    r"\babsence of\b.{0,45}\bseasonal(?:ity| pattern| variation| variations)?\b",
    r"\bno evidence of\b.{0,45}\bseasonal(?:ity| pattern| variation| variations)?\b",
    r"\bdid not\b.{0,45}\b(?:show|demonstrate|exhibit|display|reveal)\b.{0,45}\bseasonal",
    r"\bseasonal(?:ity| pattern| variation| variations)?\b.{0,45}\b(?:was|were)?\s*not\b.{0,25}\b(?:observed|detected|found|significant|evident|apparent|discernible)\b",
    r"\bnon[\s-]?seasonal\b",
]


POSITIVE_PATTERNS = [
    r"\b(?:showed|shows|show|demonstrated|demonstrates|demonstrate|exhibited|exhibits|exhibit|displayed|displays|display|revealed|reveals|reveal|indicated|indicates|indicate)\b.{0,60}\bseasonal",
    r"\b(?:seasonality|seasonal pattern|seasonal variation|seasonal variations)\b.{0,45}\b(?:was|were)\b.{0,20}\b(?:observed|detected|found|evident|apparent|significant|clear|marked|pronounced)\b",
    r"\b(?:significant|clear|marked|pronounced|distinct|strong)\b.{0,25}\bseasonal(?:ity| pattern| variation| variations)?\b",
    r"\bseasonal(?:ity| pattern| variation| variations)?\b.{0,45}\b(?:occurred|exists|persisted|emerged)\b",
]


UNCLEAR_METHOD_PATTERNS = [
    r"\b(?:assess|assessed|assessing|evaluate|evaluated|evaluating|examine|examined|examining|investigate|investigated|investigating|test|tested|testing|analy[sz]e|analy[sz]ed|analy[sz]ing|explore|explored|exploring)\b.{0,60}\bseasonal",
    r"\bseasonal(?:ity| pattern| variation| variations)?\b.{0,60}\b(?:assess|evaluate|examine|investigate|test|analy[sz]e|explore)",
]


def classify_sentence(sentence: str) -> str:
    """
    Classify what the sentence says about seasonality.

    Negative rules are evaluated first so that statements such as
    "did not exhibit a discernible seasonal pattern" cannot be classified
    as positive merely because they contain "exhibit" and "seasonal".
    """
    text = sentence.lower()

    if any(re.search(pattern, text) for pattern in NEGATIVE_PATTERNS):
        return "negative"

    if any(re.search(pattern, text) for pattern in UNCLEAR_METHOD_PATTERNS):
        return "unclear"

    if any(re.search(pattern, text) for pattern in POSITIVE_PATTERNS):
        return "positive"

    return "unclear"


def relation_priority(relation: str) -> int:
    # If several matched sentences occur in one abstract, preserve the most
    # informative explicit conclusion while retaining all matched sentences.
    return {
        "positive": 2,
        "negative": 2,
        "unclear": 1,
    }[relation]


def classify_record(record: dict, virus: str) -> dict | None:
    matched = []

    # Treat the title as its own sentence, then process abstract sentences.
    sentences = []
    if record["title"]:
        sentences.append(record["title"])
    sentences.extend(sentence_split(record["abstract"]))

    for sentence in sentences:
        if (
            sentence_mentions_virus(sentence, virus)
            and sentence_mentions_seasonality(sentence)
        ):
            matched.append(
                {
                    "sentence": sentence,
                    "relation": classify_sentence(sentence),
                }
            )

    if not matched:
        return None

    relations = {item["relation"] for item in matched}

    # Explicit positive and negative statements in the same record are kept
    # as "unclear" at record level because the abstract requires manual review.
    if "positive" in relations and "negative" in relations:
        record_relation = "unclear"
    elif "positive" in relations:
        record_relation = "positive"
    elif "negative" in relations:
        record_relation = "negative"
    else:
        record_relation = "unclear"

    return {
        **record,
        "relation": record_relation,
        "matched_sentence": " || ".join(
            item["sentence"] for item in matched
        ),
    }


def find_relevant_publications(
    client: PubMedClient,
    virus: str,
    max_records: int,
    year_from: int | None,
) -> list[dict]:
    query = build_pubmed_query(
        virus=virus,
        year_from=year_from,
    )

    total_hits, pmids = client.search(
        query=query,
        max_records=max_records,
    )

    records = client.fetch(pmids) if pmids else []

    relevant = []
    for record in records:
        classified = classify_record(record, virus)
        if classified is not None:
            relevant.append(classified)

    if total_hits > len(records):
        print(
            f"    WARNING: PubMed returned {total_hits} hits, "
            f"but only {len(records)} were downloaded. "
            f"Increase --max-records.",
            file=sys.stderr,
        )

    counts = {
        relation: sum(
            record["relation"] == relation
            for record in relevant
        )
        for relation in ("positive", "negative", "unclear")
    }

    print(
        f"    Relevant publications: {len(relevant)} "
        f"(positive={counts['positive']}, "
        f"negative={counts['negative']}, "
        f"unclear={counts['unclear']})",
        file=sys.stderr,
        flush=True,
    )

    return relevant


def write_results(
    rows: list[dict],
    output_path: Path,
) -> None:
    fieldnames = [
        "virus",
        "pmid",
        "year",
        "title",
        "relation",
        "matched_sentence",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    rows: list[dict],
    output_path: Path,
) -> None:
    fieldnames = [
        "virus",
        "relevant_publications",
        "positive",
        "negative",
        "unclear",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve PubMed publications explicitly relevant "
            "to seasonality of selected human viruses."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "virus_seasonality_publications.csv"
        ),
        help="Output CSV containing relevant PubMed publications.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "virus_seasonality_summary.csv"
        ),
        help="Summary CSV with one row per virus and publication count.",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=3000,
        help=(
            "Maximum PubMed records downloaded per virus. "
            "Default: 3000."
        ),
    )

    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Optional earliest publication year.",
    )

    args = parser.parse_args()

    client = PubMedClient()
    output_rows: list[dict] = []
    summary_rows: list[dict] = []

    for index, virus in enumerate(
        VIRUSES,
        start=1,
    ):
        print(
            f"[{index}/{len(VIRUSES)}] {virus}",
            file=sys.stderr,
            flush=True,
        )

        try:
            records = find_relevant_publications(
                client=client,
                virus=virus,
                max_records=args.max_records,
                year_from=args.year_from,
            )

            summary_rows.append(
                {
                    "virus": virus,
                    "relevant_publications": len(records),
                    "positive": sum(
                        record["relation"] == "positive"
                        for record in records
                    ),
                    "negative": sum(
                        record["relation"] == "negative"
                        for record in records
                    ),
                    "unclear": sum(
                        record["relation"] == "unclear"
                        for record in records
                    ),
                }
            )

            for record in records:
                output_rows.append(
                    {
                        "virus": virus,
                        "pmid": record["pmid"],
                        "year": record["year"],
                        "title": record["title"],
                        "relation": record["relation"],
                        "matched_sentence": record["matched_sentence"],
                    }
                )

        except Exception as exc:
            print(
                f"    ERROR: {exc}",
                file=sys.stderr,
                flush=True,
            )
            summary_rows.append(
                {
                    "virus": virus,
                    "relevant_publications": "",
                    "positive": "",
                    "negative": "",
                    "unclear": "",
                }
            )

    write_results(
        rows=output_rows,
        output_path=args.output,
    )

    write_summary(
        rows=summary_rows,
        output_path=args.summary_output,
    )

    print(
        f"Wrote {len(output_rows)} publications to {args.output}",
        file=sys.stderr,
    )
    print(
        f"Wrote summary for {len(summary_rows)} viruses to {args.summary_output}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:18:08 2026

@author: jaanajurvansuu

Search PubMed for evidence of viral reactivation or persistence in
immunocompromised human hosts for a fixed list of 37 human viruses.

Logic
-----
A publication is retained only when:
1. the virus name or one of its aliases occurs in a title/abstract sentence;
2. a specific reactivation/persistence outcome occurs in that SAME sentence;
3. an immunocompromised-host term occurs in that sentence or an immediately
   adjacent sentence.

The matched context is classified as:
- positive: explicit evidence of reactivation/persistence/prolonged infection;
- negative: explicit evidence that such an outcome was absent/not observed;
- unclear: the outcome is discussed or assessed but the sentence does not
  clearly state a positive or negative result.

This is a literature-identification tool. It does not by itself establish that
a virus reactivates or persists in immunocompromised hosts.

Outputs
-------
1. virus_immunocompromised_publications.csv
   virus, pmid, year, title, relation, matched_context

2. virus_immunocompromised_summary.csv
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


# Human-virus aliases. Standalone "CMV" is intentionally excluded; HCMV is
# retained for human cytomegalovirus.
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


IMMUNOCOMPROMISED_TERMS = [
    "immunocompromised",
    "immunosuppressed",
    "immunosuppression",
    "immunodeficiency",
    "immune deficiency",
    "transplant",
    "transplantation",
    "solid organ transplant",
    "solid organ transplantation",
    "hematopoietic stem cell",
    "haematopoietic stem cell",
    "hematopoietic cell transplant",
    "stem cell transplant",
    "bone marrow transplant",
    "allogeneic",
    "autologous",
    "HIV",
    "AIDS",
    "chemotherapy",
    "rituximab",
    "corticosteroid",
    "biologic therapy",
    "malignancy",
    "leukemia",
    "leukaemia",
    "lymphoma",
    "neutropenia",
]


# Deliberately specific outcomes. Generic "viral load", "viremia", and
# "replication" alone are excluded because they do not establish persistence
# or reactivation.
OUTCOME_TERMS = [
    "reactivation",
    "reactivated",
    "reactivates",
    "persistent infection",
    "persistent viral infection",
    "viral persistence",
    "persistence",
    "chronic infection",
    "chronic viral infection",
    "chronic hepatitis",
    "prolonged infection",
    "prolonged viral infection",
    "prolonged shedding",
    "persistent shedding",
    "impaired clearance",
    "delayed clearance",
    "failure of clearance",
    "failure to clear",
    "failed to clear",
    "unable to clear",
    "latent infection",
    "latency",
    "recurrence",
    "recurrent infection",
    "relapse",
    "disseminated infection",
    "opportunistic infection",
]


class PubMedClient:
    def __init__(self) -> None:
        self.email = os.getenv("NCBI_EMAIL", "anonymous@example.com")
        self.api_key = os.getenv("NCBI_API_KEY")
        self.tool = os.getenv(
            "NCBI_TOOL",
            "human_virus_immunocompromised_context",
        )
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
    # PubMed performs a broad candidate retrieval. More stringent contextual
    # filtering is performed locally after the abstracts are downloaded.
    query = (
        f"{build_virus_clause(virus)} "
        f"AND {build_or_group(IMMUNOCOMPROMISED_TERMS)} "
        f"AND {build_or_group(OUTCOME_TERMS)}"
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
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text,
        )
        if sentence.strip()
    ]


def contains_term(text: str, terms: Iterable[str]) -> bool:
    normalized = normalize_text(text)

    return any(
        normalize_text(term) in normalized
        for term in terms
    )


def sentence_mentions_virus(
    sentence: str,
    virus: str,
) -> bool:
    text = normalize_text(sentence)

    for alias in ALIASES.get(virus, [virus]):
        for pattern in alias_patterns(alias):
            if pattern.search(text):
                return True

    return False


NEGATIVE_PATTERNS = [
    r"\bno\b.{0,45}\breactivation\b",
    r"\bwithout\b.{0,45}\breactivation\b",
    r"\bno evidence of\b.{0,45}\breactivation\b",
    r"\bdid not\b.{0,40}\breactivat",
    r"\breactivation\b.{0,40}\b(?:was|were)?\s*not\b.{0,25}\b(?:observed|detected|found)\b",
    r"\bno\b.{0,45}\b(?:persistence|persistent infection|chronic infection)\b",
    r"\bwithout\b.{0,45}\b(?:persistence|persistent infection|chronic infection)\b",
    r"\bno evidence of\b.{0,45}\b(?:persistence|persistent infection|chronic infection)\b",
    r"\bcleared\b.{0,30}\b(?:virus|infection)\b",
    r"\bviral clearance\b.{0,30}\b(?:occurred|achieved|successful|complete)\b",
]


METHOD_PATTERNS = [
    r"\b(?:assess|assessed|assessing|evaluate|evaluated|evaluating|examine|examined|examining|investigate|investigated|investigating|monitor|monitored|monitoring|study|studied|studying)\b.{0,70}\b(?:reactivation|persistence|persistent|chronic|clearance|shedding|latency|recurrence|relapse)\b",
    r"\b(?:reactivation|persistence|persistent|chronic|clearance|shedding|latency|recurrence|relapse)\b.{0,70}\b(?:assess|evaluate|examine|investigate|monitor|study)\b",
]


POSITIVE_PATTERNS = [
    r"\b(?:developed|experienced|had|showed|demonstrated|exhibited|displayed|presented with|was associated with|were associated with)\b.{0,70}\breactivation\b",
    r"\breactivation\b.{0,60}\b(?:occurred|observed|detected|documented|reported|common|frequent|increased)\b",
    r"\b(?:persistent|chronic|prolonged)\b.{0,35}\b(?:infection|shedding)\b",
    r"\b(?:viral persistence|persistence)\b.{0,60}\b(?:occurred|observed|detected|documented|reported)\b",
    r"\b(?:impaired|delayed)\b.{0,25}\bclearance\b",
    r"\b(?:failure|failed|unable)\b.{0,30}\bclear",
    r"\b(?:disseminated|opportunistic)\b.{0,35}\binfection\b",
    r"\b(?:recurrence|recurrent infection|relapse)\b.{0,60}\b(?:occurred|observed|reported|documented)\b",
]


def classify_outcome_sentence(sentence: str) -> str:
    text = sentence.lower()

    # Negative first to avoid false positives such as
    # "no CMV reactivation was observed."
    if any(
        re.search(pattern, text)
        for pattern in NEGATIVE_PATTERNS
    ):
        return "negative"

    if any(
        re.search(pattern, text)
        for pattern in METHOD_PATTERNS
    ):
        return "unclear"

    if any(
        re.search(pattern, text)
        for pattern in POSITIVE_PATTERNS
    ):
        return "positive"

    # An explicit outcome term without a clear assertion remains relevant,
    # but its direction is not inferred.
    return "unclear"


def classify_record(
    record: dict,
    virus: str,
) -> dict | None:
    # Treat title as its own sentence, then abstract sentences.
    sentences: list[str] = []
    if record["title"]:
        sentences.append(record["title"])
    sentences.extend(sentence_split(record["abstract"]))

    matches = []

    for index, sentence in enumerate(sentences):
        # Virus and outcome must be linked within the same sentence.
        if not sentence_mentions_virus(sentence, virus):
            continue

        if not contains_term(sentence, OUTCOME_TERMS):
            continue

        # Immunocompromised-host context must be in the same or adjacent
        # sentence. This is stricter than whole-abstract co-occurrence while
        # allowing common abstract constructions.
        context_start = max(0, index - 1)
        context_end = min(len(sentences), index + 2)
        context_sentences = sentences[context_start:context_end]
        context = " ".join(context_sentences)

        if not contains_term(
            context,
            IMMUNOCOMPROMISED_TERMS,
        ):
            continue

        matches.append(
            {
                "relation": classify_outcome_sentence(
                    sentence
                ),
                "context": context,
            }
        )

    if not matches:
        return None

    relations = {
        item["relation"]
        for item in matches
    }

    if "positive" in relations and "negative" in relations:
        relation = "unclear"
    elif "positive" in relations:
        relation = "positive"
    elif "negative" in relations:
        relation = "negative"
    else:
        relation = "unclear"

    return {
        **record,
        "relation": relation,
        "matched_context": " || ".join(
            item["context"]
            for item in matches
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
        classified = classify_record(
            record,
            virus,
        )

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
        for relation in (
            "positive",
            "negative",
            "unclear",
        )
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


def write_publications(
    rows: list[dict],
    output_path: Path,
) -> None:
    fieldnames = [
        "virus",
        "pmid",
        "year",
        "title",
        "relation",
        "matched_context",
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
            "Retrieve PubMed evidence of viral reactivation or "
            "persistence in immunocompromised human hosts."
        )
    )

    parser.add_argument(
        "--publications-output",
        type=Path,
        default=Path(
            "virus_immunocompromised_publications.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "virus_immunocompromised_summary.csv"
        ),
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
    publication_rows: list[dict] = []
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

            for record in records:
                publication_rows.append(
                    {
                        "virus": virus,
                        "pmid": record["pmid"],
                        "year": record["year"],
                        "title": record["title"],
                        "relation": record["relation"],
                        "matched_context": record[
                            "matched_context"
                        ],
                    }
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

        except Exception as exc:
            print(
                f"    ERROR: {exc}",
                file=sys.stderr,
                flush=True,
            )

            # Blank values distinguish a failed query from a true zero.
            summary_rows.append(
                {
                    "virus": virus,
                    "relevant_publications": "",
                    "positive": "",
                    "negative": "",
                    "unclear": "",
                }
            )

    write_publications(
        rows=publication_rows,
        output_path=args.publications_output,
    )

    write_summary(
        rows=summary_rows,
        output_path=args.summary_output,
    )

    print(
        f"Wrote publications: {args.publications_output}",
        file=sys.stderr,
    )
    print(
        f"Wrote summary: {args.summary_output}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

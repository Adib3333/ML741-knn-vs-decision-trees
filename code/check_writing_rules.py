"""Check the report against the writing rules.

Runs on the prose of the LaTeX source with mathematics, algorithms, tables,
citations, cross-references and generated macros stripped out, so only the
sentences a reader actually reads get examined.
"""
from __future__ import annotations

import os
import re
import sys

TEX = os.environ.get("A2_TEX", "report/26243881RW741assignment2.tex")
NUM = "report/numbers.tex"

FINDINGS: list[tuple[str, str, str]] = []


def flag(rule: str, detail: str, snippet: str) -> None:
    FINDINGS.append((rule, detail, " ".join(snippet.split())[:160]))


def load() -> tuple[str, str, dict]:
    tex = open(TEX).read()
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\n", open(NUM).read()))

    body = tex
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", body, flags=re.S)
    body = re.sub(r"\\appendices.*$", "", body, flags=re.S)
    body = re.sub(r"^.*?\\begin\{abstract\}", "", body, flags=re.S)

    prose = body
    prose = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", " ", prose, flags=re.S)
    prose = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", " ", prose, flags=re.S)
    prose = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", prose, flags=re.S)
    prose = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " EQUATION ", prose, flags=re.S)
    prose = re.sub(r"\$[^$]*\$", " MATH ", prose)
    prose = re.sub(r"\\cite\{[^}]*\}", " CITE ", prose)
    prose = re.sub(r"\\(?:ref|label)\{[^}]*\}", " REF ", prose)
    prose = re.sub(r"\\footnote\{[^}]*\}", " ", prose)
    prose = re.sub(r"\\url\{[^}]*\}", " URL ", prose)
    prose = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^}]*)\}", r" HEAD:\1. ", prose)
    prose = re.sub(r"\\emph\{([^}]*)\}", r"\1", prose)
    prose = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", prose)
    # expand the generated numeric macros to their values
    prose = re.sub(r"\\([A-Za-z]+)\{\}",
                   lambda m: macros.get(m.group(1), " VALUE "), prose)
    prose = re.sub(r"\\[A-Za-z]+", " ", prose)
    prose = re.sub(r"[{}]", " ", prose)
    prose = re.sub(r"%.*", " ", prose)
    prose = re.sub(r"\s+", " ", prose)
    return tex, prose, macros


def sentences(prose: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", prose)
    return [p.strip() for p in parts if len(p.split()) > 2 and not p.startswith("HEAD:")]


def check() -> None:
    tex, prose, macros = load()
    sents = sentences(prose)

    def ctx(m, s=70):
        return prose[max(0, m.start() - s):m.start() + s]

    # WR3 - third person
    for w in [r"\bwe\b", r"\bour\b", r"\bus\b", r"(?<![A-Za-z])I(?![A-Za-z])",
              r"\bmy\b", r"\bone (?:can|should|must|may|is able)\b"]:
        for m in re.finditer(w, prose):
            flag("WR3", f"first person or 'one': {m.group()}", ctx(m))

    # WR4d - present continuous used as a noun phrase
    for m in re.finditer(r"\b(?:Optimising|Analysing|Comparing|Evaluating|Removing|"
                         r"Applying|Selecting|Measuring|Encoding) (?:of|the)\b", prose):
        flag("WR4d", f"present continuous: {m.group()}", ctx(m))

    # WR6 - uncertain terms
    for w in [r"\bsome\b", r"\bcertain\b", r"\bvarious\b", r"\bseveral\b",
              r"\ba number of\b", r"\bquite\b", r"\bfairly\b"]:
        for m in re.finditer(w, prose, re.I):
            flag("WR6", f"uncertain term: {m.group()}", ctx(m))

    # WR7 - hedging
    for w in [r"\bmay be\b", r"\bcan be\b", r"\bcould be\b", r"\bmight be\b",
              r"\bpossibly\b", r"\bperhaps\b", r"\bprobably\b", r"\bseems\b"]:
        for m in re.finditer(w, prose, re.I):
            flag("WR7", f"hedge: {m.group()}", ctx(m))

    # WR8 - etc
    for m in re.finditer(r"\betc\b", prose, re.I):
        flag("WR8", "use of 'etc'", ctx(m))

    # WR13 - contractions, WR14 - possessive apostrophes
    for m in re.finditer(r"\b[A-Za-z]+'(?:s|t|re|ve|ll|d|m)\b", prose):
        flag("WR13/WR14", f"apostrophe: {m.group()}", ctx(m))

    # WR18 - single-digit numbers in prose
    for m in re.finditer(r"(?<![\w.\-/])([0-9])(?![\w.%\-/])", prose):
        c = ctx(m, 45)
        if re.search(r"(?:Fig|Table|Algorithm|Section|Appendix|equation)\.?\s*$",
                     prose[max(0, m.start() - 14):m.start()], re.I):
            continue
        flag("WR18", f"single digit in prose: {m.group()}", c)

    # WR19 - sentence-level rules
    for s in sents:
        if re.match(r"^[0-9]", s):
            flag("WR19c", "sentence starts with a number", s)
        if re.match(r"^(?:And|But)\b", s):
            flag("WR19g", "sentence starts with And or But", s)
        if re.match(r"^(?:It|This|These|They|Those|Them|Its)\b(?!\s+[a-z])", s):
            flag("WR5", "sentence opens with an ambiguous pronoun", s)
        commas = s.count(",")
        conj = len(re.findall(r"\b(?:and|but|because|although|while|whereas|"
                              r"since|however|therefore)\b", s, re.I))
        nwords = len(s.split())
        if nwords > 40 or (commas >= 3 and nwords > 30):
            flag("WR19a", f"long sentence: {nwords} words, {commas} commas, "
                          f"{conj} conjunctives", s)

    # WR19f - "In this section," openings
    for m in re.finditer(r"In this (?:section|chapter|report),", prose, re.I):
        flag("WR19f", "opening with 'In this section,'", ctx(m))

    # WR31 - consistent British spelling
    for w in [r"normaliz", r"standardiz", r"optimiz", r"analyz", r"behavior\b",
              r"modeling\b", r"\bcenter\b", r"labeled\b", r"minimiz", r"maximiz"]:
        for m in re.finditer(w, prose, re.I):
            c = ctx(m)
            if "SMOTE" in c or "Synthetic" in c or "Oversampling" in c:
                continue
            flag("WR31", f"American spelling: {m.group()}", c)

    # WR30 - Latin terms in italics
    for m in re.finditer(r"et al\.", tex):
        pre = tex[max(0, m.start() - 12):m.start()]
        if "emph{" not in pre and "textit{" not in pre:
            flag("WR30", "'et al.' not in italics", tex[max(0, m.start() - 60):m.start() + 30])

    # WR22c - acronyms in headings or captions
    for m in re.finditer(r"\\(?:section|subsection|caption)\{([^}]*)\}", tex):
        for acr in ["k-NN", "SMOTE", "IQR", " CT "]:
            if acr in m.group(1):
                flag("WR22c", f"acronym '{acr.strip()}' in heading or caption", m.group(1))

    # WR28a - headings must not contain a verb or be a question
    for m in re.finditer(r"\\(?:section|subsection)\{([^}]*)\}", tex):
        h = m.group(1)
        if h.endswith("?"):
            flag("WR28a", "heading is a question", h)
        if re.search(r"\b(?:is|are|was|were|does|do|can|will|using|showing)\b", h, re.I):
            flag("WR28a", "heading contains a verb", h)

    # WR28c - no section with exactly one subsection
    for m in re.finditer(r"\\section\{([^}]*)\}", tex):
        nxt = tex.find("\\section{", m.end())
        block = tex[m.end():nxt if nxt > 0 else len(tex)]
        n = len(re.findall(r"\\subsection\{", block))
        if n == 1:
            flag("WR28c", "section has exactly one subsection", m.group(1))

    # WR28f - every section ends with a summary
    secs = list(re.finditer(r"\\section\{([^}]*)\}", tex))
    for i, m in enumerate(secs):
        end = secs[i + 1].start() if i + 1 < len(secs) else len(tex)
        block = tex[m.end():end]
        name = m.group(1)
        if name in ("Introduction", "Conclusions", "Code Repository"):
            continue
        if "\\subsection{Summary}" not in block:
            flag("WR28f", "section does not end with a summary", name)

    # WR25/WR26 - float captions, references and placement
    for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
        blk = m.group(1)
        if blk.find("\\caption") < blk.find("\\includegraphics"):
            flag("WR25a", "figure caption above the figure", blk[:80])
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        blk = m.group(1)
        if 0 < blk.find("\\input") < blk.find("\\caption"):
            flag("WR26a", "table caption below the table", blk[:80])

    for kind, pat in (("figure", r"\\label\{(fig:[^}]+)\}"),
                      ("table", r"\\label\{(tab:[^}]+)\}"),
                      ("algorithm", r"\\label\{(alg:[^}]+)\}")):
        for m in re.finditer(pat, tex):
            lab = m.group(1)
            refs = [r.start() for r in re.finditer(r"\\ref\{" + re.escape(lab) + r"\}", tex)]
            if not refs:
                flag("WR25e", f"{kind} never referred to in the text", lab)
            elif min(refs) > m.start():
                flag("WR25g/WR26d", f"{kind} placed before its first reference", lab)

    # WR23a/b - bibliography sorted and fully cited
    for m in re.finditer(r"\\bibitem\{([^}]+)\}", tex):
        if not re.search(r"\\cite\{[^}]*" + re.escape(m.group(1)) + r"[,}]", tex):
            flag("WR23b", "bibliography entry never cited", m.group(1))
    entries = re.findall(r"\\bibitem\{[^}]+\}\s*\n([^\n]*)", tex)
    surnames = []
    for e in entries:
        e = re.sub(r"\\v\{?[A-Za-z]\}?", "", e)
        mm = re.match(r"\s*(?:[A-Z]\.)+\s*([A-Za-z\\'\- ]+?),", e)
        surnames.append(mm.group(1).strip().lower() if mm else e[:12].lower())
    bad = [f"'{a}' before '{b}'" for a, b in zip(surnames, surnames[1:]) if a > b]
    if bad:
        flag("WR23a", "bibliography not alphabetical", "; ".join(bad))

    # WR23d/e - no pre-prints, no blogs
    for w in ["arxiv", "researchsquare", "blog", "medium.com", "towardsdatascience"]:
        if w in tex.lower():
            flag("WR23d/e", f"disallowed source type: {w}", w)

    # WR11 - acronym defined before first use
    for acr, definition in (("k-NN", "abbreviated as k-NN"),
                            ("CT", "abbreviated as CT")):
        uses = [m.start() for m in re.finditer(re.escape(acr), prose)]
        d = prose.find(definition)
        if uses and (d < 0 or d > min(uses)):
            flag("WR11/WR22a", f"acronym '{acr}' used before definition",
                 prose[max(0, min(uses) - 80):min(uses) + 40])


def main() -> int:
    check()
    by_rule: dict[str, list] = {}
    for rule, detail, snip in FINDINGS:
        by_rule.setdefault(rule, []).append((detail, snip))

    total = 0
    for rule in sorted(by_rule):
        items = by_rule[rule]
        total += len(items)
        print(f"\n=== {rule}  ({len(items)}) ===")
        for detail, snip in items[:14]:
            print(f"  - {detail}\n      ...{snip}...")
        if len(items) > 14:
            print(f"  ... and {len(items) - 14} more")

    print(f"\nTOTAL AUTOMATED FINDINGS: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

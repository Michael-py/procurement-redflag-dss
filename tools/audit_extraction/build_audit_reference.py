"""Build the audit reference tables from OCR output of the Auditor-General's
2024 annual report (Volumes I and II).

    python build_audit_reference.py <ocr_dir_vol1> <ocr_dir_vol2> <warehouse.sqlite> <out_dir>

Writes:
  oaugf_2024_issues.csv                one row per audit issue, with its title
  oaugf_2024_procurement_findings.csv  one row per audited MDA, matched to a
                                       NOCOPO buyer where a confident match exists

An issue counts as procurement-related when its title matches PROCUREMENT_RE.
The rule is deliberately simple and is applied to titles only; every title is
kept in the issues file so the classification can be inspected.
"""
import csv
import glob
import json
import os
import re
import sqlite3
import sys
from collections import OrderedDict

PROCUREMENT_RE = re.compile(
    r"CONTRACT|PROCURE|AWARD|TENDER|DUE PROCESS|MARKET SURVEY|SPLITTING|NOT EXECUTED|"
    r"EVIDENCE OF EXECUTION|EVIDENCE EXECUTION|MOBILI[SZ]ATION|\bBPP\b")

# Audited MDA (start of the name as printed in the report) -> NOCOPO buyer id.
# Only matches that are unambiguous by name are listed; everything else stays
# unmatched. Reviewed by hand against the buyer table on 12 October 2026.
P = "nigeria_portal:NG-BPP-BPP-NOC-"
BUYER_MAP = {
    "Cocoa Research Institute of Nigeria": "215014001",
    "Division of Agricultural College, Zaria": "517022001",
    "Nigerian Institute for Oil Palm Research": "215008001",
    "Nigerian College of Aviation Technology": "229031003",
    "Nigeria Extractive Industries Transparency": "111011001",
    "Nigeria Geological Survey Agency": "233003001",
    "Nigerian Communication Satellite": "156003001",
    "Nigerian Institute of Social and Economic Research": "238002001",
    "Nigerian Press Council": "123009001",
    "Voice of Nigeria": "123006001",
    "Federal Ministry of Science Technology and Innovation": "228001001",
    "National Biotechnology Development Agency": "228008001",
    "National Biotechnology Research and Development Agency": "228008001",
    "Federal Medical Centre, Birnin Kebbi": "521027033",
    "Federal University, Gashua": "517021039",
    "Federal University, Oye-Ekiti": "517021029",
    "Obafemi Awolowo University": "517021005",
    "University of Abuja": "517021010",
    "University of Lagos": "517021002",
    "University of I-Jyo": "517021020",          # OCR reading of "University of Uyo"
    "Federal University of Technology, Akure": "517021017",
    "National Obstetrics and Fistula Centre, Ningi": "521048002",
    "New Partnership for Africa": "111052001",
    "Institute of Peace and Conflict Resolution": "119006001",
    "Nigerians in Diaspora Commission": "119009001",
    "University Teaching Hospital Ilorin": "521026007",
    "National Mathematical Centre": "517021025",
    "National Power Training Institute": "231010001",
    "Federal Medical Centre, Abeokuta": "521027019",
    "National Research Institute for Chemical Technology": "228044001",
    "National Agricultural Land Development Authority": "215059001",
    "Ministry of Petroleum Resources Headquarters": "232001001",
    "Federal Ministry of Women Affairs": "514001001",
    "Federal Ministry of Humanitarian Affairs": "553001001",
    "Bureau of Public Service Reforms": "125008001",
    "Federal Ministry of Aviation": "31",
    "Federal Road Safety Corps": "161007001",
    "National Identity Management Commission": "161005001",
    "Nigerian Correctional Service": "124002001",
    "Federal Ministry of Agriculture and Food Security": "3",
    "National Population Commission": "543001001",
    "Fiscal Responsibility Commission": "539001007",
    "Debt Management Office": "220002001",
    "Abubakar Tafawa Balewa Teaching Hospital": "521026016",
    "National Hospital, Abuja": "521049001",
    "University of Benin Teaching Hospital": "521026005",
    "University of Maiduguri Teaching Hospital": "521026011",
    "Nigerian Upstream Petroleum Regulatory Commission": "232005001",
    "Nigerian Midstream and Downstream Petroleum Regulatory Authority": "232008008",
}


def read_pages(ocr_dir):
    flat = []
    for pf in sorted(glob.glob(os.path.join(ocr_dir, "p*.json"))):
        page = int(os.path.basename(pf)[1:5])
        lines = sorted(json.load(open(pf, encoding="utf-8")), key=lambda l: (round(l["y"] / 12), l["x"]))
        flat += [(page, l["t"].strip()) for l in lines if l["t"].strip()]
    return flat


def parse_volume(flat, volume):
    """Find each 'At the <MDA>, for the period ..., the following observation(s)
    was/were made' opener and the 'ISSUE n:' headings that follow it."""
    sections, cur = [], None
    for i, (page, text) in enumerate(flat):
        if re.match(r"^At\s+the\s+", text):
            chunk = " ".join(t for _, t in flat[i:i + 5])
            m = re.match(r"^At\s+the\s+(.+?)[,.]?\s+(?:for|during|covering)\s+the\s+(?:period|year|financial)", chunk)
            if m and "observation" in " ".join(t for _, t in flat[i:i + 8]).lower():
                period = re.search(r"(?:period|year)\s+(.{0,90}?)(?:,?\s*the following|\s+the following)", chunk)
                cur = {"volume": volume, "page": page, "mda": m.group(1).strip(),
                       "period": period.group(1).strip() if period else "", "issues": []}
                sections.append(cur)
        m = re.match(r"^ISSUE\s*(\d*)\s*[:;.]?\s*(.*)", text, flags=re.I)
        if m and cur is not None and text[:5].isupper():
            title, j = [m.group(2)], i + 1
            while j < len(flat) and j < i + 9:
                nxt = flat[j][1]
                if re.match(r"^(Description|Audit observed|Risk|Paragraph|Section|Financial Regulation)", nxt, flags=re.I):
                    break
                letters = sum(c.isalpha() for c in nxt)
                if not (nxt.isupper() or re.search(r"\(=?[N#]", nxt) or (letters and sum(c.isupper() for c in nxt) > 0.6 * letters)):
                    break
                title.append(nxt)
                j += 1
            cur["issues"].append({"page": page, "no": int(m.group(1) or 1), "title": " ".join(t for t in title if t).strip()})
    return sections


def main(v1_dir, v2_dir, warehouse, out_dir):
    sections = parse_volume(read_pages(v1_dir), 1) + parse_volume(read_pages(v2_dir), 2)
    con = sqlite3.connect(warehouse)
    buyers = dict(con.execute("SELECT buyer_key, name FROM buyer WHERE source = 'nigeria_portal'"))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "oaugf_2024_issues.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["volume", "page", "mda", "audit_period", "issue_no", "title", "procurement_related"])
        for s in sections:
            for i in s["issues"]:
                w.writerow([s["volume"], i["page"], s["mda"], s["period"], i["no"], i["title"],
                            int(bool(PROCUREMENT_RE.search(i["title"].upper())))])

    mdas = OrderedDict()
    for s in sections:
        name = re.sub(r"\s*\([^)]*\)", "", s["mda"]).strip()
        key = re.sub(r"[^a-z]", "", name.lower())[:28]
        m = mdas.setdefault(key, {"mda": name, "refs": [], "periods": [], "total": 0, "proc": 0})
        m["refs"].append(f"Vol.{s['volume']} p.{s['page']}")
        m["periods"].append(s["period"])
        m["total"] += len(s["issues"])
        m["proc"] += sum(1 for i in s["issues"] if PROCUREMENT_RE.search(i["title"].upper()))

    with open(os.path.join(out_dir, "oaugf_2024_procurement_findings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mda", "report_refs", "audit_periods", "issues_total", "issues_procurement", "buyer_key", "nocopo_buyer_name"])
        matched = 0
        for m in mdas.values():
            key = next((P + v for k, v in BUYER_MAP.items() if m["mda"].startswith(k)), "")
            if key and key not in buyers:
                raise SystemExit(f"mapped buyer not in warehouse: {key}")
            matched += bool(key)
            w.writerow([m["mda"], "; ".join(m["refs"]), "; ".join(p for p in m["periods"] if p), m["total"], m["proc"],
                        key, buyers.get(key, "")])
    print(f"sections={len(sections)} mdas={len(mdas)} issues={sum(len(s['issues']) for s in sections)} matched_to_nocopo={matched}")


if __name__ == "__main__":
    main(*sys.argv[1:5])

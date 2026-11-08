"""Single source for the citation metadata: writes CITATION.cff and .zenodo.json so they always match.

    python tools/citation_metadata.py          # from the repository root

Edit the values in META below, then rerun. The script checks that both files carry identical metadata.
"""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

META = {
    "title": "TOMShapeFlow-DEM: particle shape and angle of repose, a DEM study for lunar regolith handling",
    "abstract": (
        "Discrete-element (Chrono DEM-Engine) study of how grain shape sets the angle of repose of a granular "
        "material relevant to lunar regolith handling. Twelve multi-sphere grain shapes (Wadell roundness 0.13 to "
        "1.0 by aspect ratio 0.6 to 1.0, 2 mm grains) are tested in a rough-floor slump test, calibrated by rolling "
        "resistance to a 40 deg angle of repose (lunar highland simulant LHS-1: 39.6 deg), with a lunar-gravity "
        "subset. Includes the simulation code, the clump templates, per-run results, a fitted design surface "
        "(angle = 46.0 - 4.3 R - 10.7 A, leave-one-out RMSE 1.6 deg) and the methods log. Code under MIT; shape "
        "templates and results under CC BY 4.0."),
    "family_names": "Torres",
    "given_names": "Luis",
    "affiliation": "Torres Orbital Mining Inc.",
    "orcid": "0009-0003-2626-6325",
    "keywords": ["lunar regolith", "ISRU", "DEM", "particle shape", "angle of repose"],
    "license": "MIT",
    "repository": "https://github.com/orbitaltom/TOMShapeFlow-DEM",
}


def cff(m):
    ab = "\n".join("  " + line for line in textwrap.wrap(m["abstract"], 100))
    kw = "".join(f"  - {k}\n" for k in m["keywords"])
    return (f'cff-version: 1.2.0\n'
            f'message: "If you use this software or data, please cite it using the metadata below."\n'
            f'type: software\n'
            f'title: "{m["title"]}"\n'
            f'abstract: >-\n{ab}\n'
            f'authors:\n'
            f'  - family-names: {m["family_names"]}\n'
            f'    given-names: {m["given_names"]}\n'
            f'    affiliation: "{m["affiliation"]}"\n'
            f'    orcid: "https://orcid.org/{m["orcid"]}"\n'
            f'keywords:\n{kw}'
            f'license: {m["license"]}\n'
            f'repository-code: "{m["repository"]}"\n')


def zenodo(m):
    return {"title": m["title"], "description": m["abstract"], "upload_type": "software", "access_right": "open",
            "creators": [{"name": f'{m["family_names"]}, {m["given_names"]}', "affiliation": m["affiliation"],
                          "orcid": m["orcid"]}],
            "keywords": m["keywords"], "license": m["license"],
            "related_identifiers": [{"identifier": m["repository"], "relation": "isSupplementTo", "scheme": "url"}]}


def check(cff_text, z):
    """Read the shared fields back from both files and compare them."""
    lines = cff_text.splitlines()
    get = lambda key: next(line.split(": ", 1)[1].strip('"') for line in lines if line.strip().startswith(key + ":"))  # noqa: E731
    a0 = lines.index("abstract: >-") + 1
    a1 = next(i for i in range(a0, len(lines)) if not lines[i].startswith("  "))
    k0 = lines.index("keywords:") + 1
    k1 = next(i for i in range(k0, len(lines)) if not lines[i].startswith("  - "))
    c = {"title": get("title"), "abstract": " ".join(line.strip() for line in lines[a0:a1]),
         "name": f'{get("- family-names")}, {get("given-names")}', "affiliation": get("affiliation"),
         "orcid": get("orcid").rsplit("/", 1)[1], "keywords": [line[4:] for line in lines[k0:k1]],
         "license": get("license")}
    cr = z["creators"][0]
    zz = {"title": z["title"], "abstract": z["description"], "name": cr["name"], "affiliation": cr["affiliation"],
          "orcid": cr["orcid"], "keywords": z["keywords"], "license": z["license"]}
    diff = [k for k in c if c[k] != zz[k]]
    if diff:
        raise SystemExit(f"CITATION.cff and .zenodo.json differ in: {diff}")
    return c


def main():
    text, z = cff(META), zenodo(META)
    (ROOT / "CITATION.cff").write_text(text)
    (ROOT / ".zenodo.json").write_text(json.dumps(z, indent=2, ensure_ascii=False) + "\n")
    c = check((ROOT / "CITATION.cff").read_text(), json.load(open(ROOT / ".zenodo.json")))
    print("CITATION.cff and .zenodo.json written; identical fields: " + ", ".join(c))


if __name__ == "__main__":
    main()

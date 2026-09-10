"""
decode_radc_codebook.py -- make RADC_codebook_*.pdf readable.

WHY THIS EXISTS. The RADC codebook is the authoritative source for what every variable in
the two .xlsx files means and how its values are coded -- and tokenize_radc.py depends on
getting those codings right. Get one backwards and the model trains happily on an inverted
label: `r_stroke` is the trap, because 4 (the value 80% of visits carry) means "Not present",
not "definite stroke".

THE PROBLEM. The PDF's text cannot be extracted. It embeds Type3 fonts -- glyphs are drawing
procedures, not characters -- with no /ToUnicode map, so every extractor emits glyph ids:

    /0/1/2/3/i255/5/6/7/3/8/9/10/8/11/12/2      ("Rush Alzheimer's")

and each of the 29 pages defines its OWN font subset, so page 2's "/5" is not page 1's "/5".
That is 29 different substitution ciphers.

THE FIX. Hash each glyph's CharProcs content stream. The same letter drawn at the same size
is the same byte sequence on every page, so the hash unifies the 29 numberings into one
alphabet of 246 distinct bitmaps (letters appear more than once because the document mixes
font sizes/weights). Solving one cipher then decodes the whole document. GLYPH_TO_CHAR below
is that solution, obtained from known-plaintext cribs ("Rush Alzheimer's Disease Center",
"Apolipoprotein E (APOE) genotype") plus dictionary constraint propagation.

Unmapped glyphs render as {hexhash} -- mostly digits and punctuation in the medication
tables. All 48 variable definitions and value codings are legible.

Output: <repo>/data/RADC/codebook_decoded.txt  (stays beside the PDF, under data/, ignored
by git -- it is the source PDF's content and carries the same data-use terms.)

Run:  python data_prep/decode_radc_codebook.py
Needs: pypdf  (pip install pypdf)
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Delphi-2M/
REPO = os.path.dirname(HERE)
RADC = os.path.join(REPO, "data", "RADC")
OUT = os.path.join(RADC, "codebook_decoded.txt")

# glyph-bitmap md5 (first 10 hex chars) -> character
GLYPH_TO_CHAR = {
    '009d58024f': 'E', '0213096aec': 'b', '05c3e985d0': 'd', '06eda5487b': '5',
    '07fd8568bd': 'x', '08ac2ba046': 'h', '09cde34ae4': 's', '0abbc76c42': 'l',
    '0b00a0e73d': 's', '0d70c6a997': 'z', '0f18e01e57': 'a', '129a027c66': 's',
    '1481a6fcc2': ',', '177e885424': 'p', '17bd904a48': 'C', '1838d4a9ce': '-',
    '18aa1a3bd5': 'h', '1b7576451c': 'p', '1bf8bd72d8': '9', '1c0c20506f': 'x',
    '1ddfab7c1f': 'l', '22ec63be1d': '&', '23901ec276': 'F', '23a3b89ea8': 'I',
    '2419dd6c50': 'N', '2427c703ab': 't', '24546097a2': 'r', '26aa209413': 'm',
    '2976a7b39b': 'm', '2a113bf3e2': '8', '2ea77f2ece': 'M', '334d7a2d46': ' ',
    '355760fe47': 'B', '369d1b0d44': 'z', '37354c1fdd': 'o', '37b190416d': 'i',
    '385e3fbd0c': '.', '38dd625670': 'h', '3a1b4e7be5': 'i', '3e1d19d3a3': '1',
    '3e96ecd145': 'O', '3f1b15de8a': 'h', '3f404dfd64': '(', '4272c21b1d': 'y',
    '488d8f3f88': 'n', '4c347a01c1': 'v', '4f0db7a3d9': 'h', '4fef4380c5': 'g',
    '504fdb8964': 'G', '51b7c2e38e': 'n', '545a661d5f': 'r', '55da435e7c': 'w',
    '57d94cad60': 'c', '5811118a9b': ')', '5e2e35b3e8': 'd', '63a788de63': 'y',
    '63e6f1c5d5': 'u', '64c1f83d6a': 'L', '6636ee20fc': 'q', '67e00e53bb': 'p',
    '68e3a00cad': '(', '6ceb0f786e': "'", '6d2ec262fb': 's', '6e322bc432': 'E',
    '6e6e91071d': '4', '72b3fc5c77': 'm', '73cf6234e3': 'j', '7b2b0b6cc8': 'd',
    '7f3db7e659': 'q', '7f7455e896': 'k', '7fa57acc59': 'd', '7fe0143d51': 'g',
    '80f94219c9': 'A', '82363d59fe': 'w', '8400d0d8dc': 'o', '844df11464': 'q',
    '845d88001a': '&', '84dac3e0cc': 'w', '85c9acff35': 'g', '8732c7f122': 'v',
    '8c236177eb': 's', '8d4552bb4a': 'i', '8f2bafab39': 'f', '9099f9797f': '6',
    '952b32c8d2': 'c', '963f3273bf': ')', '97c3740eb9': 's', '987db42fad': 'n',
    '990ac90379': 'q', '998ba6ccb8': '_', '9b14399b96': 'c', '9c6963dcfb': 'C',
    '9e1f7a20e2': '0', 'a13cf62030': 't', 'a17d5efa25': 'z', 'a4a492a969': 'u',
    'a703cbd2e5': 'O', 'a990732344': 'T', 'a9a2ab13a9': '/', 'a9cc59a721': 'd',
    'aa468f3065': 'e', 'b035c00edc': 'b', 'b399fde45b': 'f', 'b916779e71': 'c',
    'b9e3c2ce36': 'g', 'ba16982e75': 'r', 'bd9d1920f4': '2', 'c1997828f1': 'j',
    'c2fb557afd': 'o', 'c3a5ac4cba': 'R', 'c50f095f78': 'a', 'c986215863': 'p',
    'ca7bace157': 'A', 'cd7df62bd8': 'v', 'd0d09ac0c6': 'r', 'd8b6dd6c40': 'r',
    'da10081ce7': '7', 'da291c8bc0': 'r', 'db9d288ba5': 'n', 'dd618264b1': 'V',
    'ddd6072cfe': 'D', 'e084cf2101': ':', 'e50d8b602f': 'n', 'e7c0eec230': 'e',
    'e7e8cf2614': 's', 'e87f982f21': 's', 'e88da09f15': 'r', 'e92f0f3f27': 'n',
    'e9d9c1e65b': '3', 'eceba86426': 'P', 'eff38f4686': 'f', 'f32739ac1d': 'u',
    'f891f09cce': 'a', 'f9c8b45337': 'd', 'f9d8006bd1': 'b', 'f9de6407d4': 'p',
    'fa3aebccb0': 'b', 'fd2f7e395d': 'e', 'ff505f5b85': 'a', 'ffed958aea': 'c',
}


def main():
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("[decode_radc_codebook] needs pypdf:  pip install pypdf")

    pdfs = sorted(f for f in os.listdir(RADC) if re.match(r"RADC_codebook.*\.pdf$", f))
    if not pdfs:
        sys.exit(f"[decode_radc_codebook] no RADC_codebook*.pdf in {RADC}")
    path = os.path.join(RADC, pdfs[-1])
    print(f"[decode_radc_codebook] reading {pdfs[-1]}")

    reader = PdfReader(path)
    out, unmapped = [], set()
    for pi, page in enumerate(reader.pages):
        # this page's own glyph-name -> bitmap-hash table
        gmap = {}
        for font in page["/Resources"]["/Font"].values():
            procs = font.get_object().get("/CharProcs")
            if procs is None:
                continue
            for gname, stream in procs.get_object().items():
                data = stream.get_object().get_data()
                gmap[gname.lstrip("/")] = hashlib.md5(data).hexdigest()[:10]

        text = page.extract_text() or ""
        buf = []
        for piece in re.split(r"(/[A-Za-z0-9]+)", text):
            if piece.startswith("/"):
                h = gmap.get(piece[1:])
                ch = GLYPH_TO_CHAR.get(h)
                if ch is None:
                    unmapped.add(h or piece)
                    buf.append("{%s}" % (h or piece[1:]))
                else:
                    buf.append(ch)
            else:
                buf.append(piece)
        out.append(f"--- page {pi} ---\n" + "".join(buf))

    os.makedirs(RADC, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(out))
    total = sum(len(p) for p in out)
    print(f"[decode_radc_codebook] {len(reader.pages)} pages, {total} chars -> {OUT}")
    print(f"[decode_radc_codebook] {len(unmapped)} glyph bitmaps still unmapped "
          f"(rendered as {{hash}}; digits/punctuation in the medication tables)")


if __name__ == "__main__":
    main()

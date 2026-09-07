#!/usr/bin/env python3
"""
typedstream_roi.py — reader for Horos/OsiriX .rois_series files.

Purpose: parse the NeXTSTEP typedstream (NSArchiver) export without resolving the
full shared-reference table, keying on two Horos invariants: arrays encode as
`92 84 93 96 <count>`, and each ROI point is an NSString like "{129.37, 66.54}".
Structure: root array = slices; each slice = ROI objects; each ROI = a points array.
RECIST crosshairs are two 2-point line ROIs per tumour.

Use (public API):
    parse_rois_series(path) -> list[list[list[(x, y)]]]   # [slice][roi][point],
    slice index aligned with the DICOM slice order.
"""

import re
import struct
import sys
from pathlib import Path

OBJ, NEW, END, FLOAT, I16, I32 = 0x92, 0x84, 0x86, 0x83, 0x81, 0x82
ARRAY_REF = 0x93
BEGIN_NEW, BEGIN_REF = 0x95, 0x96
_PT = re.compile(r"\{\s*(-?[0-9.eE]+)\s*,\s*(-?[0-9.eE]+)\s*\}\s*$")


class Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0

    def u8(self):
        b = self.d[self.p]; self.p += 1; return b

    def pk(self, o=0):
        q = self.p + o
        return self.d[q] if q < len(self.d) else -1

    def rint(self):
        b = self.u8()
        if b == I16:
            v = int.from_bytes(self.d[self.p:self.p+2], "little", signed=True); self.p += 2; return v
        if b == I32:
            v = int.from_bytes(self.d[self.p:self.p+4], "little", signed=True); self.p += 4; return v
        return b - 256 if b >= 0x80 else b

    def atom(self):
        n = self.rint()
        s = self.d[self.p:self.p+n].decode("latin-1"); self.p += n
        return s

    # ── skip a class definition chain: leading 0x84's, name, version, super ──
    def skip_class(self):
        while self.pk() == NEW:
            self.p += 1
        name = self.atom()
        _ver = self.rint()
        if self.pk() == 0x00:
            self.p += 1
        elif self.pk() == NEW:
            self.skip_class()
        return name

    def try_string(self):
        """If the bytes ahead encode an NSString payload (<len><'{'...> that is
        immediately followed by the object END marker), read and return it;
        else None and leave the cursor untouched. The trailing-END requirement
        rejects coincidental <len><'{'> byte pairs inside colour/float junk.
        Handles both the new and referenced NSString layouts (a stray enc/ref
        byte may precede the length)."""
        for off in (0, 1, 2):
            L = self.pk(off)
            if 1 <= L <= 90 and self.pk(off + 1) == 0x7b:      # '{'
                end = self.p + off + 1 + L
                if end < len(self.d) and self.d[end] == END:
                    self.p += off
                    return self.atom()
        return None

    def read_object(self):
        """0x92 already consumed. Returns dict{coords, kids}."""
        node = {"coords": [], "kids": []}
        b = self.pk()
        if b != NEW:                       # whole-object back-reference
            self.p += 1
            return node
        self.p += 1                        # object class-marker 0x84

        is_array = False
        if self.pk() == NEW:               # new class definition
            name = self.skip_class()
            if self.pk() == BEGIN_NEW:
                self.p += 1
            is_array = name in ("NSArray", "NSMutableArray")
        else:                              # class reference
            ref = self.u8()
            is_array = (ref == ARRAY_REF)
            if self.pk() == BEGIN_REF:
                self.p += 1

        if is_array:
            count = self.rint()
            for _ in range(count):
                if self.pk() == OBJ:
                    self.p += 1
                    child = self.read_object()
                    node["kids"].append(child)
                    node["coords"].extend(child["coords"])
                else:                       # defensive: unexpected token
                    self.p += 1
            if self.pk() == END:
                self.p += 1
            return node

        # non-array object: harvest strings, recurse nested objects, until END
        while True:
            b = self.pk()
            if b == END or b == -1:
                if b == END:
                    self.p += 1
                break
            if b == OBJ:
                self.p += 1
                child = self.read_object()
                node["coords"].extend(child["coords"])
                node["kids"].append(child)
                continue
            s = self.try_string()
            if s is not None:
                m = _PT.match(s.strip())
                if m:
                    x, y = float(m.group(1)), float(m.group(2))
                    if x >= 0 and y >= 0:          # drop negative junk point
                        node["coords"].append((x, y))
                continue
            if b == FLOAT:
                self.p += 5
            elif b in (I16, I32) or b < 0x80:
                self.rint()
            else:
                # opaque byte: shared reference, class/atom marker (0x84), or a
                # begin marker. The bytes of any real type-encoding atom that
                # follow are all low/safe and get consumed harmlessly below.
                self.p += 1
        return node

    def read_root(self):
        assert self.u8() == 0x04
        self.atom()                                # "streamtyped"
        self.rint()                                # system version
        # class pre-declaration + root object header, up to the root array
        assert self.u8() == NEW
        self.atom()                                # root type encoding "@"
        while self.pk() != OBJ:
            b = self.pk()
            if b == NEW:
                self.p += 1
                if self.pk() == NEW:
                    self.skip_class()
                else:
                    self.atom()
            else:
                self.p += 1
        self.p += 1                                # consume the root 0x92
        return self.read_object()


def parse_rois_series(path):
    root = Reader(Path(path).read_bytes()).read_root()
    slices = []
    for slot in root["kids"]:
        slices.append([roi["coords"] for roi in slot["kids"]])
    return slices


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "brac46551b.rois_series"
    sl = parse_rois_series(path)
    n_ann = sum(1 for s in sl if s)
    n_roi = sum(len(s) for s in sl)
    n_pts = sum(len(r) for s in sl for r in s)
    print(f"slots (slices): {len(sl)}")
    print(f"annotated slices: {n_ann}   ROIs: {n_roi}   endpoints: {n_pts}")
    for i, s in enumerate(sl):
        if s:
            print(f"  slice {i:2d}: {len(s)} ROI  " +
                  "  ".join(f"[{len(r)}pt]" for r in s))

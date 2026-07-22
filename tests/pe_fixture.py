"""Deterministic in-memory PE fixture used by feature and LLM tests."""

import struct


def build_minimal_suspicious_pe() -> bytes:
    """Return a parseable PE32+ with deterministic static-risk indicators."""
    data = bytearray(0x400)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)

    pe_offset = 0x80
    data[pe_offset : pe_offset + 4] = b"PE\x00\x00"
    coff_offset = pe_offset + 4
    struct.pack_into(
        "<HHIIIHH",
        data,
        coff_offset,
        0x8664,
        1,
        4_102_444_800,
        0,
        0,
        0xF0,
        0x0022,
    )

    optional_offset = coff_offset + 20
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        data,
        optional_offset,
        0x20B,
        14,
        0,
        0x200,
        0x200,
        0,
        0x1000,
        0x1000,
        0x140000000,
        0x1000,
        0x200,
        6,
        0,
        0,
        0,
        6,
        0,
        0,
        0x2000,
        0x200,
        0,
        3,
        0x8160,
        0x100000,
        0x1000,
        0x100000,
        0x1000,
        0,
        16,
    )

    section_offset = optional_offset + 0xF0
    struct.pack_into(
        "<8sIIIIIIHHI",
        data,
        section_offset,
        b".ex0\x00\x00\x00\x00",
        0x200,
        0x1000,
        0x200,
        0x200,
        0,
        0,
        0,
        0,
        0xE0000060,
    )
    data[0x200:0x400] = bytes(range(256)) * 2
    return bytes(data)

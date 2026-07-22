"""Runtime correctness and performance patches for thrember feature extraction.

thrember.features.AuthenticodeSignature.raw_features() does `for cert in certs[:-1]:` where
`certs` is a signify `CertificateStore`. In signify==0.8.1 (pinned for a separate import-time
compatibility reason — see CLAUDE.local.md), CertificateStore supports `len()` but not slicing,
so this raises `TypeError: 'CertificateStore' object is not subscriptable`.

This isn't a rare edge case: it fires on essentially any Authenticode-signed PE (confirmed on
Notepad++, KeePass, FileZilla, Git for Windows installers — i.e. most real signed software),
which made the live-upload path fail on the majority of legitimately-signed real-world files.

Patched by wrapping certs in list() before slicing, and adding TypeError to the same
parse-error fallback the original method already uses for other malformed-signature cases
(rather than letting it propagate as an unhandled exception).

The string extractor also calls ``re.search`` for every already-compiled pattern. Calling the
compiled pattern directly preserves counts while avoiding redundant regex dispatch.
"""

import io
from collections import OrderedDict

import lief
import numpy as np
import signify.exceptions
from signify.authenticode import SignedPEFile
from thrember.features import AuthenticodeSignature, StringExtractor

# Quiet LIEF's console logging. On packed/minimal/crafted PEs (e.g. a tiny msfvenom payload)
# LIEF prints warnings like "Unable to find the section associated with IMPORT_TABLE" to stderr;
# parsing still succeeds and our code handles genuine failures via exceptions/None, so these are
# just noise in the server log. ERROR level keeps real errors while dropping the WARN spam.
# LIEF logging is a process-global singleton, so this one call covers both the feature extractor
# and the lief-based LLM feature summary.
lief.logging.set_level(lief.logging.LEVEL.ERROR)


def _patched_raw_features(self, bytez, pe):
    if pe is None:
        return {}

    raw_obj = {
        "num_certs": 0,
        "self_signed": 0,
        "empty_program_name": 0,
        "no_countersigner": 0,
        "parse_error": 0,
        "chain_max_depth": 0,
        "latest_signing_time": 0,
        "signing_time_diff": 0,
    }
    try:
        signed_pe = SignedPEFile(io.BytesIO(bytez))
        for signed_data in signed_pe.iter_signed_datas():
            raw_obj["num_certs"] += 1
            if signed_data.signer_info.program_name is None:
                raw_obj["empty_program_name"] = 1

            signer_info = signed_data.signer_info
            countersigner = signer_info.countersigner

            if countersigner is not None:
                signing_time = countersigner.signing_time.timestamp()
                if signing_time >= raw_obj["latest_signing_time"]:
                    raw_obj["latest_signing_time"] = signing_time
                pe_timestamp = pe.FILE_HEADER.TimeDateStamp
                raw_obj["signing_time_diff"] = signing_time - pe_timestamp
            else:
                raw_obj["no_countersigner"] = 1

            certs = list(signed_data.certificates)  # <-- the actual fix: was certs[:-1]
            if len(certs) > raw_obj["chain_max_depth"]:
                raw_obj["chain_max_depth"] = len(certs)
            for cert in certs[:-1]:
                if cert.issuer == cert.subject:
                    raw_obj["self_signed"] = 1

    except signify.exceptions.SignerInfoParseError:
        raw_obj["parse_error"] = 1
    except signify.exceptions.ParseError:
        raw_obj["parse_error"] = 1
    except ValueError:
        raw_obj["parse_error"] = 1
    except KeyError:
        raw_obj["parse_error"] = 1
    except TypeError:
        raw_obj["parse_error"] = 1
    return raw_obj


def _optimized_string_raw_features(self, bytez, pe):
    """Preserve thrember string features while avoiding redundant regex dispatch."""
    allstrings = self._allstrings.findall(bytez)
    allstrings_ascii = [value.decode() for value in allstrings]
    if allstrings:
        string_lengths = [len(value) for value in allstrings]
        average_length = sum(string_lengths) / len(string_lengths)
        shifted = [value - ord(b"\x20") for value in b"".join(allstrings)]
        counts = np.bincount(shifted, minlength=96)
        printable_count = counts.sum()
        probabilities = counts.astype(np.float32) / printable_count
        populated = np.where(counts)[0]
        entropy = np.sum(
            -probabilities[populated] * np.log2(probabilities[populated])
        )
    else:
        average_length = 0
        counts = np.zeros((96,), dtype=np.float32)
        entropy = 0
        printable_count = 0

    string_counts = {}
    for value in allstrings_ascii:
        for name, pattern in self._regexes.items():
            if pattern.search(value):
                string_counts[name] = string_counts.get(name, 0) + 1

    return {
        "numstrings": len(allstrings),
        "avlength": average_length,
        "printabledist": counts.tolist(),
        "printables": int(printable_count),
        "entropy": float(entropy),
        "string_counts": OrderedDict(sorted(string_counts.items())),
    }


StringExtractor.raw_features = _optimized_string_raw_features
AuthenticodeSignature.raw_features = _patched_raw_features

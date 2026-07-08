"""Runtime patch for a bug in thrember's Authenticode signature feature extractor.

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
"""

import io

import signify.exceptions
from signify.authenticode import SignedPEFile
from thrember.features import AuthenticodeSignature


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


AuthenticodeSignature.raw_features = _patched_raw_features

#!/bin/bash
# Download 10 legitimate, well-known Windows software installers for use as benign test
# samples against the malware detector. All sources are official vendor download endpoints
# (Mozilla's official "always latest" redirect, GitHub Releases API, SourceForge's official
# "latest" redirect, and 7-zip.org's own download page) — no third-party mirrors.
#
# Run inside Kali: bash download_benign_samples.sh

set -uo pipefail
OUT_DIR="./benign_samples"
mkdir -p "$OUT_DIR"

# Retry transient network failures automatically instead of giving up after one attempt.
CURL_OPTS=(--retry 4 --retry-delay 3 --retry-max-time 120 -A "Mozilla/5.0")

# Real installers here are all tens of MB. Anything smaller is an error page, a truncated
# transfer, or (as seen once) a malformed multi-URL string that slipped past extraction —
# treat it as a failure rather than silently keeping a bogus file.
MIN_SIZE_BYTES=1000000

# A real single download URL: one line, http(s) scheme, no stray JSON/braces from a mangled
# extraction (this is what crashed curl with "too many {} sets" when the network hiccuped
# mid-response and grep picked up multiple concatenated JSON fields).
is_valid_url() {
    local url="$1"
    [[ "$url" =~ ^https?://[^[:space:]{}\"\']+$ ]] && [ "$(printf '%s' "$url" | wc -l)" -eq 0 ]
}

download() {
    local name="$1"
    local url="$2"
    local out="$3"
    echo "[*] $name -> $out"

    if ! is_valid_url "$url"; then
        echo "    FAILED: extracted URL looks malformed, skipping (${url:0:80}...)"
        return
    fi

    local err
    if err=$(curl -sSL "${CURL_OPTS[@]}" --max-time 180 -o "$OUT_DIR/$out" "$url" 2>&1); then
        local size
        size=$(stat -c%s "$OUT_DIR/$out" 2>/dev/null || echo 0)
        if [ "$size" -lt "$MIN_SIZE_BYTES" ]; then
            echo "    FAILED: downloaded file is only $size bytes (expected a real installer) — likely an error page or truncated transfer"
            rm -f "$OUT_DIR/$out"
        else
            echo "    OK ($(du -h "$OUT_DIR/$out" | cut -f1))"
        fi
    else
        echo "    FAILED: $err"
    fi
}

resolve() {
    # $1 = URL to fetch, with retries — used for the small metadata/listing lookups.
    curl -s "${CURL_OPTS[@]}" --max-time 30 "$1"
}

github_latest_asset() {
    # $1 = owner/repo, $2 = grep pattern to pick the right asset from the release JSON
    resolve "https://api.github.com/repos/$1/releases/latest" \
        | grep "browser_download_url" | grep -iE "$2" | head -1 \
        | grep -oE 'https://[^"]*'
}

echo "=== 1/10 Firefox ==="
download "Firefox" "https://download.mozilla.org/?product=firefox-latest&os=win64&lang=en-US" "Firefox_Setup.exe"

echo "=== 2/10 Thunderbird ==="
download "Thunderbird" "https://download.mozilla.org/?product=thunderbird-latest&os=win64&lang=en-US" "Thunderbird_Setup.exe"

echo "=== 3/10 VLC ==="
VLC_FILE=$(resolve "https://get.videolan.org/vlc/last/win64/" | grep -oE 'vlc-[0-9]+\.[0-9]+\.[0-9]+-win64\.exe' | head -1)
if [ -n "$VLC_FILE" ]; then
    download "VLC" "https://get.videolan.org/vlc/last/win64/$VLC_FILE" "VLC_Setup.exe"
else
    echo "    could not resolve VLC filename, skipping"
fi

echo "=== 4/10 Notepad++ ==="
NPP_URL=$(github_latest_asset "notepad-plus-plus/notepad-plus-plus" 'Installer\.x64\.exe"')
[ -n "$NPP_URL" ] && download "Notepad++" "$NPP_URL" "NotepadPP_Setup.exe" || echo "    could not resolve Notepad++ URL, skipping"

echo "=== 5/10 Git for Windows ==="
GIT_URL=$(github_latest_asset "git-for-windows/git" '64-bit\.exe"')
[ -n "$GIT_URL" ] && download "Git for Windows" "$GIT_URL" "Git_Setup.exe" || echo "    could not resolve Git URL, skipping"

echo "=== 6/10 7-Zip ==="
SEVENZIP_PATH=$(resolve "https://www.7-zip.org/download.html" | grep -oE 'href="a/7z[0-9]+-x64\.exe"' | head -1 | sed 's/href="//;s/"//')
if [ -n "$SEVENZIP_PATH" ]; then
    download "7-Zip" "https://www.7-zip.org/$SEVENZIP_PATH" "7Zip_Setup.exe"
else
    echo "    could not resolve 7-Zip URL, skipping"
fi

echo "=== 7/10 Audacity ==="
AUDACITY_URL=$(github_latest_asset "audacity/audacity" '64bit\.exe"')
[ -n "$AUDACITY_URL" ] && download "Audacity" "$AUDACITY_URL" "Audacity_Setup.exe" || echo "    could not resolve Audacity URL, skipping"

echo "=== 8/10 qBittorrent ==="
QBIT_URL=$(resolve "https://api.github.com/repos/qbittorrent/qBittorrent/releases/latest" \
    | grep "browser_download_url" | grep -iE 'x64_setup\.exe"' | grep -v "lt20_" | head -1 \
    | grep -oE 'https://[^"]*')
[ -n "$QBIT_URL" ] && download "qBittorrent" "$QBIT_URL" "qBittorrent_Setup.exe" || echo "    could not resolve qBittorrent URL, skipping"

echo "=== 9/10 KeePass ==="
download "KeePass" "https://sourceforge.net/projects/keepass/files/latest/download" "KeePass_Setup.exe"

echo "=== 10/10 FileZilla ==="
download "FileZilla" "https://sourceforge.net/projects/filezilla/files/latest/download" "FileZilla_Setup.exe"

echo
echo "=== Done. Downloaded files: ==="
ls -la "$OUT_DIR"

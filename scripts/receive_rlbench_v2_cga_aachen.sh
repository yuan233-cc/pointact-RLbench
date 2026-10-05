#!/usr/bin/env bash
set -euo pipefail

root=/local/weihangli/datasets
final="$root/RLBenchPolarNormal10TasksV2_CGAOffline_20261004"
partial="$final.partial.26195"
mode="${1:?expected init, extract, or publish}"

test "$(hostname -s)" = aachen
test "$(stat -c %U "$root")" = weihangli
test "$(findmnt -n -o FSTYPE -T "$root")" = ext4
test "$(realpath -m "$partial")" = "$partial"
test "$(realpath -m "$final")" = "$final"

case "$mode" in
  init)
    test ! -e "$final"
    test ! -e "$partial"
    mkdir -- "$partial"
    ;;
  extract)
    test -d "$partial"
    test ! -e "$final"
    tar --keep-old-files -xf - -C "$partial"
    ;;
  publish)
    test -d "$partial"
    test ! -e "$final"
    /usr/bin/python3 - "$partial" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
groups = {}
count = 0
for split, expected in (("train", 4546), ("val", 505)):
    entries = json.loads((root / f"{split}_manifest.json").read_text())["samples"]
    if len(entries) != expected:
        raise SystemExit(f"{split}: expected {expected}, got {len(entries)}")
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty record: {path}")
        old = groups.setdefault(entry["group"], split)
        if old != split:
            raise SystemExit(f"Split overlap: {entry['group']}")
    count += len(entries)
records = list((root / "records").glob("*.npz"))
if len(records) != count:
    raise SystemExit(f"Expected {count} records, found {len(records)}")
metadata = json.loads((root / "conversion.json").read_text())
if metadata["records"] != count:
    raise SystemExit("Conversion metadata count mismatch")
print(f"verified {count} records; train=4546 val=505; no episode leakage")
PY
    mv -T -- "$partial" "$final"
    echo "published $final"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

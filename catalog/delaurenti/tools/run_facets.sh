#!/bin/bash
# run facets.py for every category that has no sidecar yet, 3 in parallel
cd "$(dirname "$0")"
for u in $(cat cats.txt); do s=$(basename "$u"); [ -f "$s.facets.json" ] || echo "$s.json"; done | xargs -P 3 -I{} python3 facets.py {}

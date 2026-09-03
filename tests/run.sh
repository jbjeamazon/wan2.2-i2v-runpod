#!/usr/bin/env bash
# Run every suite; stop at the first failure.
set -e
cd "$(dirname "$0")/.."
for t in tests/test_*.py; do
    echo "── $t"
    python3 "$t" | tail -1
done
echo "── all suites passed"

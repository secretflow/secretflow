#!/usr/bin/env bash
set -euo pipefail

DEV="${DEV:-Lo}"

sudo tc qdisc del dev "${DEV}" root 2>/dev/null || true
echo "Cleared qdisc on ${DEV}"
sudo tc qdisc show dev "${DEV}"
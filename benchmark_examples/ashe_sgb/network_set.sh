#!/usr/bin/env bash
set -euo pipefail

DEV="${DEV:-lo}"
RATE="${RATE:-100mbit}"
DELAY_MS="${DELAY_MS:-5}"  # one-way delay; RTT≈2*DELAY_MS

sudo tc qdisc del dev "${DEV}" root 2>/dev/null || true

sudo tc qdisc add dev "${DEV}" root handle 1: htb default 11
sudo tc class add dev "${DEV}" parent 1: classid 1:11 htb rate "${RATE}" ceil "${RATE}"
sudo tc qdisc add dev "${DEV}" parent 1:11 handle 10: netem delay "${DELAY_MS}ms"

echo "Applied netem on ${DEV}: rate=${RATE}, delay=${DELAY_MS}ms(one-way)"
sudo tc qdisc show dev "${DEV}"
sudo tc class show dev "${DEV}"
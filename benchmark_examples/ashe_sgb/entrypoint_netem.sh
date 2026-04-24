set -e

NETEM_ENABLE="${NETEM_ENABLE:-1}"
NETEM_DEV="${NETEM_DEV:=lo}"
NETEM_RATE="${NETEM_RATE:-100mbit}"
NETEM_DELAY="${NETEM_DELAY:-5ms}"

if [ "${NETEM_ENABLE}" = "1" ]; then
  if command -v tc >/dev/null 2>&1; then
      tc qdisc del dev "${NETEM_DEV}" root >/dev/null 2>&1 || true
      tc qdisc add dev "${NETEM_DEV}" root handle 1: htb default 11
      tc class add dev "${NETEM_DEV}" parent 1: classid 1:11 htb rate "${NETEM_RATE}" ceil "${NETEM_RATE}"
      tc qdisc add dev "${NETEM_DEV}" parent 1:11 handle 10: netem delay "${NETEM_DELAY}"
  fi
fi

exec "$@"
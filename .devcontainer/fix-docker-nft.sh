#!/bin/bash
# Fix iptables backend for nftables-only hosts (e.g. Fedora 43+ with podman/nftables)
# The docker-in-docker devcontainer feature forces iptables-legacy, which fails when
# the host kernel doesn't have legacy iptable_filter/iptable_nat modules loaded.
set -euo pipefail

DOCKER_INIT="/usr/local/share/docker-init.sh"
if [ ! -x "$DOCKER_INIT" ]; then
    echo "ERROR: $DOCKER_INIT not found. Is docker-in-docker feature installed?" >&2
    exit 1
fi

# Switch to nft backend if legacy is currently selected
current=$(readlink -f /usr/sbin/iptables)
if [[ "$current" == *"legacy"* ]]; then
    echo "Switching iptables from legacy to nft backend..."
    update-alternatives --set iptables /usr/sbin/iptables-nft
    update-alternatives --set ip6tables /usr/sbin/ip6tables-nft

    # Kill the stale dockerd that failed with legacy iptables
    pkill dockerd 2>/dev/null || true
    pkill containerd 2>/dev/null || true
    # Wait for processes to actually exit (up to 5s)
    for i in $(seq 1 10); do
        pgrep -x dockerd >/dev/null 2>&1 || break
        sleep 0.5
    done

    # Re-run the docker-in-docker init script to start dockerd
    "$DOCKER_INIT"
else
    echo "iptables already using nft backend, checking docker..."
    # Make sure docker is actually running
    if ! docker info >/dev/null 2>&1; then
        pkill dockerd 2>/dev/null || true
        pkill containerd 2>/dev/null || true
        for i in $(seq 1 10); do
            pgrep -x dockerd >/dev/null 2>&1 || break
            sleep 0.5
        done
        "$DOCKER_INIT"
    fi
fi

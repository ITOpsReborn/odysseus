#!/usr/bin/env bash
# check-docker-intel-gpu.sh - read-only Intel Arc Pro B60 WSL Docker diagnostic.
#
# This script does not install packages, edit .env, or restart Docker. It checks
# the WSL GPU bridge (/dev/dxg + /usr/lib/wsl/lib), Docker access, and whether a
# small container can see the same bridge when run with the Intel WSL overlay's
# mounts/devices.

set -u

PASS=0
FAIL=0
WARN=0
TEST_IMAGE="${ODYSSEUS_INTEL_TEST_IMAGE:-alpine:3.20}"

_pass() { printf '\033[32m[PASS]\033[0m %s\n' "$*"; PASS=$((PASS + 1)); }
_fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL + 1)); }
_warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; WARN=$((WARN + 1)); }
_info() { printf '\033[34m[INFO]\033[0m %s\n' "$*"; }

_usage() {
    cat <<'USAGE'
Usage: scripts/check-docker-intel-gpu.sh

Read-only Intel Arc Pro B60 Docker-on-WSL diagnostic. Installs nothing, edits
nothing, and does not restart Docker.

Checks:
  - the shell is running inside WSL
  - host /dev/dxg exists
  - host /usr/lib/wsl/lib exists
  - optional host clinfo visibility when installed
  - Docker can pass /dev/dxg and /usr/lib/wsl/lib into a small container

Environment:
  ODYSSEUS_INTEL_TEST_IMAGE  Docker image for the passthrough smoke
                             (default: alpine:3.20)
USAGE
}

for _arg in "$@"; do
    case "${_arg}" in
        --help|-h)
            _usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "${_arg}" >&2
            _usage >&2
            exit 1
            ;;
    esac
done

_check_wsl() {
    _info "Checking for WSL..."
    if grep -Eiq '(microsoft|wsl)' /proc/version 2>/dev/null; then
        _pass "WSL environment detected."
    else
        _fail "This script is intended for Ubuntu running inside WSL2 on Windows."
    fi
    echo
}

_check_host_bridge() {
    _info "Checking host Intel WSL GPU bridge..."
    if [ -e /dev/dxg ]; then
        _pass "/dev/dxg exists"
    else
        _fail "/dev/dxg is missing - WSL GPU compute is not available in this distro."
    fi

    if [ -d /usr/lib/wsl/lib ]; then
        _pass "/usr/lib/wsl/lib exists"
        ls /usr/lib/wsl/lib 2>/dev/null | head -10 | sed 's/^/        /'
    else
        _fail "/usr/lib/wsl/lib is missing - WSL GPU userspace libraries are not available."
    fi
    echo
}

_check_host_tools() {
    _info "Checking optional host Intel userspace tools..."
    if command -v clinfo >/dev/null 2>&1; then
        if clinfo 2>/dev/null | grep -Eq 'Intel|Level-Zero|OpenCL'; then
            _pass "clinfo can see an Intel/Level Zero/OpenCL device on the host."
            clinfo 2>/dev/null \
                | grep -E 'Device Name|Platform Name|Driver Version' \
                | head -12 \
                | sed 's/^/        /'
        else
            _warn "clinfo is installed but did not report an Intel GPU."
        fi
    else
        _warn "clinfo not found. This does not block Docker passthrough, but it limits host-side verification."
    fi
    echo
}

_check_docker() {
    _info "Checking Docker..."
    if ! command -v docker >/dev/null 2>&1; then
        _fail "docker not found - install Docker first."
        echo
        return 1
    fi
    if docker info >/dev/null 2>&1; then
        _pass "Docker daemon is running."
    else
        _fail "Docker daemon is not running or this user lacks Docker permission."
        echo
        return 1
    fi
    echo
}

_check_docker_passthrough() {
    _info "Testing Intel WSL GPU bridge passthrough with ${TEST_IMAGE} (may pull on first run)..."
    if docker run --rm \
        --device=/dev/dxg \
        -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
        "${TEST_IMAGE}" \
        sh -lc 'test -e /dev/dxg && test -d /usr/lib/wsl/lib && ls /usr/lib/wsl/lib >/dev/null' \
        >/dev/null 2>&1; then
        _pass "Docker can pass /dev/dxg and /usr/lib/wsl/lib into a container."
    else
        _fail "Docker Intel WSL GPU passthrough failed."
        _info "Check Docker Desktop/Engine WSL integration, then retry."
    fi
    echo
}

_print_next_steps() {
    echo "=== Suggested .env values ==="
    echo 'COMPOSE_FILE=docker-compose.yml:docker/gpu.intel.yml'
    echo
    echo "After restarting Odysseus, verify the slim app container sees the WSL GPU bridge:"
    echo "  docker compose exec odysseus sh -lc 'test -e /dev/dxg && test -d /usr/lib/wsl/lib && echo \$LD_LIBRARY_PATH && ls -l /dev/dxg /usr/lib/wsl/lib | head'"
    echo
    echo "Note: the slim Odysseus image does not include Intel oneAPI / Level Zero /"
    echo "OpenCL userspace or Intel-tuned inference engines."
    echo "Device passthrough is necessary but not sufficient for local GPU serving;"
    echo "Cookbook-installed serve backends still need Intel-compatible userspace."
}

echo "=== Odysseus Intel Arc Pro B60 Docker-on-WSL diagnostic ==="
echo
_check_wsl
_check_host_bridge
_check_host_tools
if _check_docker; then
    _check_docker_passthrough
fi
_print_next_steps
echo
echo "=== Results: ${PASS} passed, ${WARN} warnings, ${FAIL} failed ==="
[ "${FAIL}" -eq 0 ]

#!/usr/bin/env bash
# check-docker-ipex-xpu-gpu.sh — read-only Intel IPEX-LLM XPU Docker diagnostic.
#
# This script does not install packages, edit .env, or restart Docker. It checks
# the native Linux Intel Arc GPU device nodes (/dev/dri/renderD*), Docker access,
# and whether a small container can see the same DRI devices when run with the
# ipex-xpu overlay's mounts and groups.
#
# Note: this script targets NATIVE Linux Intel Arc (bare-metal / native VM).
# For Intel Arc on Windows + WSL2, use scripts/check-docker-intel-gpu.sh instead.

set -u

PASS=0
FAIL=0
WARN=0
TEST_IMAGE="${ODYSSEUS_IPEX_XPU_TEST_IMAGE:-alpine:3.20}"

_pass() { printf '\033[32m[PASS]\033[0m %s\n' "$*"; PASS=$((PASS + 1)); }
_fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL + 1)); }
_warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; WARN=$((WARN + 1)); }
_info() { printf '\033[34m[INFO]\033[0m %s\n' "$*"; }

_usage() {
    cat <<'USAGE'
Usage: scripts/check-docker-ipex-xpu-gpu.sh

Read-only Intel IPEX-LLM XPU (native Linux) Docker diagnostic. This script
installs nothing, edits nothing, and does not restart Docker.

Checks:
  - NOT running inside WSL (this overlay is for bare-metal / native-VM Linux)
  - /dev/dri/renderD* device nodes exist on the host
  - Intel i915 DRM driver is loaded
  - Optional: intel_gpu_top / clinfo Intel GPU visibility
  - Docker can pass /dev/dri into a small container
  - render group membership for the current user

Environment:
  ODYSSEUS_IPEX_XPU_TEST_IMAGE  Docker image for the passthrough smoke
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

_check_not_wsl() {
    _info "Checking execution environment..."
    if grep -Eiq '(microsoft|wsl)' /proc/version 2>/dev/null; then
        _warn "WSL environment detected. This overlay targets native Linux Intel Arc."
        _info "For Intel Arc on Windows + WSL2, use: scripts/check-docker-intel-gpu.sh"
    else
        _pass "Running on native Linux (not WSL)."
    fi
    echo
}

_check_host_dri() {
    _info "Checking /dev/dri render nodes..."

    if [ -d /dev/dri ]; then
        _pass "/dev/dri directory exists."
    else
        _fail "/dev/dri is missing — no DRM/KMS GPU devices found."
        echo
        return
    fi

    local _render_count=0
    for _node in /dev/dri/renderD*; do
        [ -e "${_node}" ] || continue
        _render_count=$((_render_count + 1))
        _pass "${_node} found."
    done

    if [ "${_render_count}" -eq 0 ]; then
        _fail "No /dev/dri/renderD* nodes found — Intel GPU DRM render nodes are not present."
        _info "Check that the i915 kernel driver is loaded: lsmod | grep i915"
    fi
    echo
}

_check_i915_driver() {
    _info "Checking Intel i915 DRM kernel driver..."
    if lsmod 2>/dev/null | grep -q '^i915'; then
        _pass "i915 kernel driver is loaded."
    else
        _warn "i915 driver not detected in lsmod output."
        _info "On Ubuntu/Debian: sudo apt install linux-modules-extra-\$(uname -r)"
        _info "Then reboot and retry."
    fi
    echo
}

_check_render_group() {
    _info "Checking render group membership..."
    if groups 2>/dev/null | grep -qw render; then
        _pass "Current user is in the render group."
    else
        _warn "Current user is NOT in the render group."
        _info "Add yourself: sudo usermod -aG render \$(whoami)"
        _info "Then log out and back in, or run: newgrp render"
        _info "Set RENDER_GID in .env to the numeric render group id:"
        _info "  getent group render | cut -d: -f3"
    fi
    echo
}

_check_host_tools() {
    _info "Checking optional host Intel userspace tools..."
    local _found=0

    if command -v intel_gpu_top >/dev/null 2>&1; then
        _found=1
        if intel_gpu_top -l 1 2>/dev/null | grep -Eq 'Intel|Render'; then
            _pass "intel_gpu_top can see an Intel GPU on the host."
        else
            _warn "intel_gpu_top is installed but did not report an Intel GPU engine."
        fi
    fi

    if command -v clinfo >/dev/null 2>&1; then
        _found=1
        if clinfo 2>/dev/null | grep -Eq 'Intel|Level-Zero|OpenCL'; then
            _pass "clinfo can see an Intel/Level Zero/OpenCL device on the host."
            clinfo 2>/dev/null | grep -E 'Device Name|Platform Name|Driver Version' \
                | head -12 \
                | sed 's/^/        /'
        else
            _warn "clinfo is installed but did not report an Intel GPU."
        fi
    fi

    if [ "${_found}" -eq 0 ]; then
        _warn "Neither intel_gpu_top nor clinfo found. This does not block Docker passthrough."
        _info "Install intel-gpu-tools or clinfo for host-side GPU verification."
    fi
    echo
}

_check_docker() {
    _info "Checking Docker..."
    if ! command -v docker >/dev/null 2>&1; then
        _fail "docker not found — install Docker first."
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
    _info "Testing Intel DRI passthrough with ${TEST_IMAGE} (may pull on first run)..."

    # Find the first available renderD node for the test
    local _render_node=""
    for _node in /dev/dri/renderD*; do
        [ -e "${_node}" ] && _render_node="${_node}" && break
    done

    if [ -z "${_render_node}" ]; then
        _fail "No /dev/dri/renderD* to pass through — skipping container test."
        echo
        return
    fi

    if docker run --rm \
        --device=/dev/dri \
        "${TEST_IMAGE}" \
        sh -lc "test -d /dev/dri && ls /dev/dri/ >/dev/null" \
        >/dev/null 2>&1; then
        _pass "Docker can pass /dev/dri into a container."
    else
        _fail "Docker Intel DRI passthrough failed."
        _info "Check that /dev/dri permissions allow your Docker user or the render group."
    fi
    echo
}

_print_next_steps() {
    local _render_gid=""
    _render_gid="$(getent group render 2>/dev/null | cut -d: -f3)" || true

    echo "=== Suggested .env values ==="
    if [ -n "${_render_gid}" ]; then
        printf 'COMPOSE_FILE=docker-compose.yml:docker/gpu.ipex-xpu.yml\n'
        printf 'RENDER_GID=%s\n' "${_render_gid}"
        printf 'IPEX_LLM_MODEL=<HuggingFace model id, e.g. Qwen/Qwen2.5-7B-Instruct>\n'
        printf '# Optional: quantisation level (default: sym_int4)\n'
        printf '# IPEX_LLM_LOAD_IN_LOW_BIT=sym_int4\n'
    else
        printf 'COMPOSE_FILE=docker-compose.yml:docker/gpu.ipex-xpu.yml\n'
        printf 'RENDER_GID=<numeric render group id>\n'
        printf 'IPEX_LLM_MODEL=<HuggingFace model id, e.g. Qwen/Qwen2.5-7B-Instruct>\n'
    fi
    echo
    echo "After starting the stack, verify the ipex-llm sidecar is serving:"
    echo "  curl http://localhost:8000/v1/models"
    echo
    echo "Add the endpoint in Odysseus Settings -> Servers:"
    echo "  http://ipex-llm:8000/v1"
    echo
    echo "Verify Odysseus itself sees the Intel DRI devices:"
    echo "  docker compose exec odysseus sh -lc 'test -d /dev/dri && ls -l /dev/dri/renderD*'"
    echo
    echo "Note: the slim Odysseus image does not bundle Intel oneAPI / Level Zero /"
    echo "OpenCL userspace. The ipex-llm sidecar (intel/ipex-llm-inference-xpu)"
    echo "provides the full Intel XPU inference stack."
}

echo "=== Odysseus Intel IPEX-LLM XPU (native Linux) Docker diagnostic ==="
echo
_check_not_wsl
_check_host_dri
_check_i915_driver
_check_render_group
_check_host_tools
if _check_docker; then
    _check_docker_passthrough
fi
_print_next_steps
echo
echo "=== Results: ${PASS} passed, ${WARN} warnings, ${FAIL} failed ==="
[ "${FAIL}" -eq 0 ]

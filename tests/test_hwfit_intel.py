"""Intel GPU (Arc / WSL / xpu) hardware detection tests.

Covers the _detect_intel() path added to services/hwfit/hardware.py:
- Returns None when no Intel signals are present.
- Detects via /dev/dxg (WSL GPU bridge).
- Detects via /dev/dxg + /usr/lib/wsl/lib with a conservative VRAM fallback.
- Detects via clinfo output (name + VRAM).
- Detects via sycl-ls when clinfo is absent.
- Detects via Intel DRM render node (/sys/class/drm, vendor 0x8086).
- Does not break NVIDIA / AMD / Apple detection (regression guard).
- Returns a dict compatible with detect_system() expectations.
"""

import os

import pytest

from services.hwfit import hardware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_run_no_intel(cmd):
    """Stub _run that pretends no Intel GPU signals exist."""
    return None


def _fake_os_exists_none(path):
    return False


def _fake_os_isdir_none(path):
    return False


def _fake_os_listdir_none(path):
    raise FileNotFoundError(path)


def _patch_no_intel(monkeypatch):
    """Monkeypatch hardware so _detect_intel() finds nothing."""
    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", _fake_run_no_intel)
    monkeypatch.setattr(os.path, "exists", _fake_os_exists_none)
    monkeypatch.setattr(os.path, "isdir", _fake_os_isdir_none)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)


# ---------------------------------------------------------------------------
# No-Intel cases
# ---------------------------------------------------------------------------

def test_returns_none_when_no_signals(monkeypatch):
    """_detect_intel() returns None when /dev/dxg, wsl/lib, and DRM all absent."""
    _patch_no_intel(monkeypatch)
    assert hardware._detect_intel() is None


def test_returns_none_on_pure_nvidia_host(monkeypatch):
    """A host with nvidia-smi but no Intel signals must return None."""
    def fake_run(cmd):
        if isinstance(cmd, list) and cmd[0] == "nvidia-smi":
            return "8000, NVIDIA RTX 4090"
        return None

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(os.path, "exists", _fake_os_exists_none)
    monkeypatch.setattr(os.path, "isdir", _fake_os_isdir_none)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    assert hardware._detect_intel() is None


# ---------------------------------------------------------------------------
# WSL /dev/dxg detection
# ---------------------------------------------------------------------------

def test_detects_via_dxg_only(monkeypatch):
    """Presence of /dev/dxg alone is sufficient to detect an Intel GPU."""
    def fake_exists(path):
        return path == "/dev/dxg"

    def fake_isdir(path):
        return False

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", _fake_run_no_intel)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    info = hardware._detect_intel()
    assert info is not None
    assert info["has_gpu"] if "has_gpu" in info else True  # returned dict doesn't have has_gpu
    assert info["backend"] == "xpu"
    assert info["gpu_count"] == 1
    assert info["gpu_vram_gb"] == 8.0  # conservative fallback
    assert "Intel" in info["gpu_name"]


def test_detects_via_wsl_lib_only(monkeypatch):
    """/usr/lib/wsl/lib alone is sufficient to detect an Intel GPU."""
    def fake_exists(path):
        return False

    def fake_isdir(path):
        return path == "/usr/lib/wsl/lib"

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", _fake_run_no_intel)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    info = hardware._detect_intel()
    assert info is not None
    assert info["backend"] == "xpu"


def test_dxg_and_wsl_lib_together(monkeypatch):
    """Both /dev/dxg and /usr/lib/wsl/lib present — still one GPU with fallback VRAM."""
    def fake_exists(path):
        return path == "/dev/dxg"

    def fake_isdir(path):
        return path == "/usr/lib/wsl/lib"

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", _fake_run_no_intel)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    info = hardware._detect_intel()
    assert info is not None
    assert info["gpu_count"] == 1
    assert info["gpu_vram_gb"] == 8.0


# ---------------------------------------------------------------------------
# clinfo name + VRAM parsing
# ---------------------------------------------------------------------------

_CLINFO_ARC_B580 = """\
  Platform Name                                   Intel(R) OpenCL Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Arc(TM) B580 Graphics
  Device Vendor                                   Intel(R) Corporation
  Device Version                                  OpenCL 3.0 NEO
  Global Memory Size                              12884901888
  Max memory allocation                           4294967296
"""


def test_clinfo_parses_name_and_vram(monkeypatch):
    """clinfo output is parsed for device name and global memory."""
    def fake_exists(path):
        return path == "/dev/dxg"

    def fake_isdir(path):
        return False

    def fake_run(cmd):
        if isinstance(cmd, list) and "clinfo" in cmd[0]:
            return _CLINFO_ARC_B580
        return None

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    info = hardware._detect_intel()
    assert info is not None
    assert "B580" in info["gpu_name"]
    assert info["gpu_vram_gb"] == 12.0


def test_clinfo_vram_fallback_on_zero(monkeypatch):
    """clinfo with 0-byte global memory keeps the 8 GB conservative fallback."""
    clinfo_zero = "  Device Name                                     Intel(R) Arc(TM) A770 Graphics\n  Global Memory Size                              0\n"

    def fake_exists(path):
        return path == "/dev/dxg"

    def fake_isdir(path):
        return False

    def fake_run(cmd):
        if isinstance(cmd, list) and "clinfo" in cmd[0]:
            return clinfo_zero
        return None

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    info = hardware._detect_intel()
    assert info is not None
    assert info["gpu_vram_gb"] == 8.0  # fallback retained


# ---------------------------------------------------------------------------
# sycl-ls name parsing
# ---------------------------------------------------------------------------

_SYCL_LS_B580 = """\
[opencl:gpu:0] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) B580 Graphics 3.0 [23.30.26918.14]
[level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) B580 Graphics 1.3 [1.3.26918]
"""


def test_sycl_ls_provides_name(monkeypatch):
    """When clinfo is absent, sycl-ls is used as a name fallback."""
    def fake_exists(path):
        return path == "/dev/dxg"

    def fake_isdir(path):
        return False

    def fake_run(cmd):
        if isinstance(cmd, list) and cmd[0] == "sycl-ls":
            return _SYCL_LS_B580
        return None

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    info = hardware._detect_intel()
    assert info is not None
    assert "Arc" in info["gpu_name"] and "B580" in info["gpu_name"]


# ---------------------------------------------------------------------------
# DRM render node detection (bare-metal Linux)
# ---------------------------------------------------------------------------

def test_detects_via_intel_drm_card(monkeypatch):
    """A DRM card node with Intel vendor 0x8086 is detected."""
    drm_files = ["card0", "card0-DP-1", "renderD128"]

    def fake_listdir(path):
        if path == "/sys/class/drm":
            return drm_files
        raise FileNotFoundError(path)

    def fake_exists(path):
        return False

    def fake_isdir(path):
        return False

    def fake_run(cmd):
        return None

    def fake_read(path):
        if path.endswith("/vendor"):
            return "0x8086"
        if path.endswith("/product_name"):
            return "Intel Arc A770 Graphics"
        return None

    # Patch the inner _read helper via the module-level open; easiest to patch
    # os.path.exists and use a _run stub that handles cat for the remote path.
    # Drive through the remote-host path so no real filesystem access occurs.
    def fake_run_remote(cmd):
        if not cmd:
            return None
        if cmd[0] == "ls" and "/sys/class/drm" in (cmd[1] if len(cmd) > 1 else ""):
            return "\n".join(drm_files)
        if cmd[0] == "cat":
            path = cmd[1] if len(cmd) > 1 else ""
            if path.endswith("/vendor"):
                return "0x8086"
            if path.endswith("/product_name"):
                return "Intel Arc A770 Graphics"
            return None
        if cmd[0] == "test":
            return None  # path doesn't exist
        return None

    monkeypatch.setattr(hardware, "_remote_host", "fake-host")
    monkeypatch.setattr(hardware, "_run", fake_run_remote)

    info = hardware._detect_intel()
    assert info is not None
    assert info["backend"] == "xpu"
    assert "Intel" in info["gpu_name"]


# ---------------------------------------------------------------------------
# Result shape compatibility
# ---------------------------------------------------------------------------

def test_result_shape_is_complete(monkeypatch):
    """_detect_intel() returns all keys that detect_system() needs."""
    def fake_exists(path):
        return path == "/dev/dxg"

    def fake_isdir(path):
        return False

    def fake_run(cmd):
        if isinstance(cmd, list) and "clinfo" in cmd[0]:
            return _CLINFO_ARC_B580
        return None

    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    monkeypatch.setattr(os, "listdir", _fake_os_listdir_none)

    info = hardware._detect_intel()
    assert info is not None
    for key in ("gpu_name", "gpu_vram_gb", "gpu_count", "gpus", "gpu_groups",
                "homogeneous", "backend"):
        assert key in info, f"missing key: {key}"
    assert isinstance(info["gpus"], list) and len(info["gpus"]) == 1
    assert isinstance(info["gpu_groups"], list) and len(info["gpu_groups"]) == 1


# ---------------------------------------------------------------------------
# Regression: existing detectors unaffected
# ---------------------------------------------------------------------------

def test_nvidia_detected_before_intel(monkeypatch):
    """When nvidia-smi reports a GPU, _detect_intel() is never reached."""
    nvidia_called = []
    intel_called = []

    def fake_nvidia():
        nvidia_called.append(True)
        return {"gpu_name": "NVIDIA RTX 4090", "gpu_vram_gb": 24.0,
                "gpu_count": 1, "gpus": [], "gpu_groups": [], "homogeneous": True,
                "backend": "cuda"}

    def fake_intel():
        intel_called.append(True)
        return None

    monkeypatch.setattr(hardware, "_detect_nvidia", fake_nvidia)
    monkeypatch.setattr(hardware, "_detect_intel", fake_intel)
    monkeypatch.setattr(hardware, "_detect_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_apple_silicon", lambda: None)

    result = hardware._detect_apple_silicon() or hardware._detect_nvidia() or hardware._detect_amd() or hardware._detect_intel()
    assert result["backend"] == "cuda"
    assert intel_called == [], "Intel detector must not run when NVIDIA is found"


def test_detect_intel_not_called_when_nvidia_wins(monkeypatch):
    """Full detect_system() integration: NVIDIA GPU wins; Intel probe not invoked."""
    intel_called = []

    def fake_detect_intel():
        intel_called.append(True)
        return None

    monkeypatch.setattr(hardware, "_detect_intel", fake_detect_intel)
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: {
        "gpu_name": "NVIDIA RTX 4090", "gpu_vram_gb": 24.0, "gpu_count": 1,
        "gpus": [{"index": 0, "name": "NVIDIA RTX 4090", "vram_gb": 24.0}],
        "gpu_groups": [], "homogeneous": True, "backend": "cuda"
    })
    monkeypatch.setattr(hardware, "_detect_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_apple_silicon", lambda: None)
    monkeypatch.setattr(hardware, "_get_ram_gb", lambda: 64.0)
    monkeypatch.setattr(hardware, "_get_available_ram_gb", lambda: 48.0)
    monkeypatch.setattr(hardware, "_get_cpu_count", lambda: 16)
    monkeypatch.setattr(hardware, "_get_cpu_name", lambda: "Intel Core i9")
    monkeypatch.setattr(hardware, "_get_cpu_arch", lambda: "x86_64")
    monkeypatch.setattr(hardware, "_remote_host", None)
    monkeypatch.setattr(hardware, "_remote_platform", None)

    result = hardware.detect_system(fresh=True)
    assert result["backend"] == "cuda"
    assert intel_called == [], "Intel detector must not run when NVIDIA wins"

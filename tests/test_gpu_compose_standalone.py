"""Guards the standalone GPU compose files against drift.

Stack-management UIs (Portainer, Coolify, Dockhand, ...) often accept only a
single compose file and do not honor COMPOSE_FILE or multiple ``-f`` overlays,
so the repo ships standalone ``docker-compose.gpu-*.yml`` files that inline the
GPU overlay. The base ``docker-compose.yml`` plus ``docker/gpu.*.yml`` overlays
remain the source of truth; these tests assert each standalone file equals the
base compose with only the matching overlay merged into the ``odysseus``
service. No Docker / docker compose is required — everything is pure YAML.
"""

import copy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "docker-compose.yml"
NVIDIA_OVERLAY = ROOT / "docker" / "gpu.nvidia.yml"
AMD_OVERLAY = ROOT / "docker" / "gpu.amd.yml"
INTEL_OVERLAY = ROOT / "docker" / "gpu.intel.yml"
IPEX_XPU_OVERLAY = ROOT / "docker" / "gpu.ipex-xpu.yml"
IPEX_XPU_WSL_OVERLAY = ROOT / "docker" / "gpu.ipex-xpu-wsl.yml"
HOST_DOCKER_OVERLAY = ROOT / "docker" / "host-docker.yml"
NVIDIA_STANDALONE = ROOT / "docker-compose.gpu-nvidia.yml"
AMD_STANDALONE = ROOT / "docker-compose.gpu-amd.yml"
INTEL_STANDALONE = ROOT / "docker-compose.gpu-intel.yml"
IPEX_XPU_STANDALONE = ROOT / "docker-compose.gpu-ipex-xpu.yml"
IPEX_XPU_WSL_STANDALONE = ROOT / "docker-compose.gpu-ipex-xpu-wsl.yml"

SERVICE = "odysseus"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Mirror docker compose overlay semantics for the keys these files use.

    Mappings merge recursively; list-valued service fields are concatenated
    (compose appends override sequences such as ``environment`` rather than
    replacing them); scalars are overwritten. The overlays here only append to
    ``environment`` and add otherwise-absent keys (``deploy``, ``devices``,
    ``group_add``), so this keeps the expected merge explicit without invoking
    docker compose.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        elif isinstance(value, list) and isinstance(result.get(key), list):
            result[key] = copy.deepcopy(result[key]) + copy.deepcopy(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_overlay_into_base(base: dict, overlay: dict) -> dict:
    """Build the expected standalone config: base + overlay on odysseus only."""
    expected = copy.deepcopy(base)
    overlay_service = overlay["services"][SERVICE]
    expected["services"][SERVICE] = _deep_merge(
        expected["services"][SERVICE], overlay_service
    )
    return expected


def _merge_overlay_with_new_services(base: dict, overlay: dict) -> dict:
    """Merge overlay into base, adding new services introduced by the overlay."""
    expected = copy.deepcopy(base)
    for svc_name, svc_def in overlay["services"].items():
        if svc_name in expected["services"]:
            expected["services"][svc_name] = _deep_merge(
                expected["services"][svc_name], svc_def
            )
        else:
            expected["services"][svc_name] = copy.deepcopy(svc_def)
    return expected


def _merge_overlays_into_base(base: dict, *overlays: dict) -> dict:
    merged = copy.deepcopy(base)
    for overlay in overlays:
        merged = _merge_overlay_into_base(merged, overlay)
    return merged


@pytest.fixture(scope="module")
def base():
    return _load(BASE)


# --- Equivalence: standalone == base + overlay -----------------------------


def test_nvidia_standalone_equals_base_plus_overlay(base):
    overlay = _load(NVIDIA_OVERLAY)
    standalone = _load(NVIDIA_STANDALONE)
    assert standalone == _merge_overlay_into_base(base, overlay)


def test_amd_standalone_equals_base_plus_overlay(base):
    overlay = _load(AMD_OVERLAY)
    standalone = _load(AMD_STANDALONE)
    assert standalone == _merge_overlay_into_base(base, overlay)


def test_intel_standalone_equals_base_plus_overlay(base):
    overlay = _load(INTEL_OVERLAY)
    standalone = _load(INTEL_STANDALONE)
    assert standalone == _merge_overlay_into_base(base, overlay)


# --- Non-odysseus services and volumes untouched ---------------------------


@pytest.mark.parametrize(
    "standalone_path", [NVIDIA_STANDALONE, AMD_STANDALONE, INTEL_STANDALONE]
)
def test_non_odysseus_services_match_base(base, standalone_path):
    standalone = _load(standalone_path)
    for name, definition in base["services"].items():
        if name == SERVICE:
            continue
        assert standalone["services"][name] == definition
    assert set(standalone["services"]) == set(base["services"])


@pytest.mark.parametrize(
    "standalone_path", [NVIDIA_STANDALONE, AMD_STANDALONE, INTEL_STANDALONE]
)
def test_top_level_volumes_match_base(base, standalone_path):
    standalone = _load(standalone_path)
    assert standalone.get("volumes") == base.get("volumes")


# --- odysseus = base service + only the overlay additions ------------------


def test_nvidia_odysseus_adds_only_overlay(base):
    standalone = _load(NVIDIA_STANDALONE)
    svc = standalone["services"][SERVICE]
    base_svc = base["services"][SERVICE]

    # Base environment preserved, plus exactly the two NVIDIA variables.
    assert "NVIDIA_VISIBLE_DEVICES=all" in svc["environment"]
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in svc["environment"]
    added_env = set(svc["environment"]) - set(base_svc["environment"])
    assert added_env == {
        "NVIDIA_VISIBLE_DEVICES=all",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    }

    # deploy block is new and matches the overlay's GPU reservation exactly.
    assert "deploy" not in base_svc
    devices = svc["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [
        {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}
    ]

    # No Docker or AMD groups are added.
    assert "devices" not in svc
    assert "group_add" not in base_svc
    assert "group_add" not in svc


def test_amd_odysseus_adds_only_overlay(base):
    standalone = _load(AMD_STANDALONE)
    svc = standalone["services"][SERVICE]
    base_svc = base["services"][SERVICE]

    # Environment is unchanged from base for AMD.
    assert svc["environment"] == base_svc["environment"]

    # Devices and GPU-only groups are added.
    assert "devices" not in base_svc
    assert svc["devices"] == ["/dev/kfd", "/dev/dri"]
    assert "group_add" not in base_svc
    assert svc["group_add"] == ["video", "${RENDER_GID:-render}"]

    # No NVIDIA-only keys leaked in.
    assert "deploy" not in svc


def test_intel_odysseus_adds_only_overlay(base):
    standalone = _load(INTEL_STANDALONE)
    svc = standalone["services"][SERVICE]
    base_svc = base["services"][SERVICE]

    assert "devices" not in base_svc
    assert svc["devices"] == ["/dev/dxg"]

    added_volumes = set(svc["volumes"]) - set(base_svc["volumes"])
    assert added_volumes == {"/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro"}

    added_env = set(svc["environment"]) - set(base_svc["environment"])
    assert added_env == {"LD_LIBRARY_PATH=/usr/lib/wsl/lib"}

    assert "group_add" not in svc
    assert "deploy" not in svc


# --- Host Docker opt-in combinations ---------------------------------------


def test_base_has_no_host_docker_access(base):
    service = base["services"][SERVICE]

    assert "/var/run/docker.sock:/var/run/docker.sock" not in service["volumes"]
    assert "ODYSSEUS_ENABLE_HOST_DOCKER=true" not in service["environment"]
    assert "group_add" not in service


def test_base_plus_host_docker_overlay_has_explicit_access(base):
    merged = _merge_overlays_into_base(base, _load(HOST_DOCKER_OVERLAY))
    service = merged["services"][SERVICE]

    assert "/var/run/docker.sock:/var/run/docker.sock" in service["volumes"]
    assert "ODYSSEUS_ENABLE_HOST_DOCKER=true" in service["environment"]
    assert service["group_add"] == ["${DOCKER_GID:-963}"]


def test_nvidia_plus_host_docker_preserves_gpu_and_docker_access(base):
    merged = _merge_overlays_into_base(
        base,
        _load(NVIDIA_OVERLAY),
        _load(HOST_DOCKER_OVERLAY),
    )
    service = merged["services"][SERVICE]

    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [
        {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}
    ]
    assert "/var/run/docker.sock:/var/run/docker.sock" in service["volumes"]
    assert "ODYSSEUS_ENABLE_HOST_DOCKER=true" in service["environment"]
    assert service["group_add"] == ["${DOCKER_GID:-963}"]


def test_amd_plus_host_docker_preserves_gpu_and_docker_groups(base):
    merged = _merge_overlays_into_base(
        base,
        _load(AMD_OVERLAY),
        _load(HOST_DOCKER_OVERLAY),
    )
    service = merged["services"][SERVICE]

    assert service["devices"] == ["/dev/kfd", "/dev/dri"]
    assert service["group_add"] == [
        "video",
        "${RENDER_GID:-render}",
        "${DOCKER_GID:-963}",
    ]
    assert "/var/run/docker.sock:/var/run/docker.sock" in service["volumes"]
    assert "ODYSSEUS_ENABLE_HOST_DOCKER=true" in service["environment"]


def test_intel_plus_host_docker_preserves_gpu_and_docker_access(base):
    merged = _merge_overlays_into_base(
        base,
        _load(INTEL_OVERLAY),
        _load(HOST_DOCKER_OVERLAY),
    )
    service = merged["services"][SERVICE]

    assert service["devices"] == ["/dev/dxg"]
    assert "/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro" in service["volumes"]
    assert "/var/run/docker.sock:/var/run/docker.sock" in service["volumes"]
    assert "LD_LIBRARY_PATH=/usr/lib/wsl/lib" in service["environment"]
    assert "ODYSSEUS_ENABLE_HOST_DOCKER=true" in service["environment"]
    assert service["group_add"] == ["${DOCKER_GID:-963}"]


# --- Intel IPEX-LLM XPU (native Linux) ------------------------------------


def test_ipex_xpu_standalone_equals_base_plus_overlay(base):
    overlay = _load(IPEX_XPU_OVERLAY)
    standalone = _load(IPEX_XPU_STANDALONE)
    assert standalone == _merge_overlay_with_new_services(base, overlay)


def test_ipex_xpu_odysseus_adds_only_overlay(base):
    standalone = _load(IPEX_XPU_STANDALONE)
    svc = standalone["services"][SERVICE]
    base_svc = base["services"][SERVICE]

    # Intel build target and DRI GPU groups are added; environment is unchanged.
    assert svc["build"] == {"context": ".", "dockerfile": "Dockerfile.ipex"}
    assert "devices" not in base_svc
    assert svc["devices"] == ["/dev/dri"]
    assert "group_add" not in base_svc
    assert svc["group_add"] == ["video", "${RENDER_GID:-render}"]
    assert svc["environment"] == base_svc["environment"]

    # No NVIDIA-specific keys, no WSL-specific volumes.
    assert "deploy" not in svc
    added_volumes = set(svc["volumes"]) - set(base_svc["volumes"])
    assert added_volumes == set()


def test_ipex_xpu_standalone_has_ipex_llm_service():
    overlay = _load(IPEX_XPU_OVERLAY)
    standalone = _load(IPEX_XPU_STANDALONE)
    assert "ipex-llm" in standalone["services"]
    assert standalone["services"]["ipex-llm"] == overlay["services"]["ipex-llm"]


def test_ipex_xpu_ipex_llm_service_shape():
    standalone = _load(IPEX_XPU_STANDALONE)
    svc = standalone["services"]["ipex-llm"]

    assert svc["image"] == "intelanalytics/ipex-llm-inference-cpp-xpu"
    assert svc["devices"] == ["/dev/dri"]
    assert svc["group_add"] == ["video", "${RENDER_GID:-render}"]
    assert "127.0.0.1:8000:8000" in svc["ports"]
    # Model cache is shared with the Odysseus HuggingFace volume.
    hf_vol = "${APP_DATA_DIR:-./data}/huggingface:/root/.cache/huggingface:z"
    assert hf_vol in svc["volumes"]
    assert "MODEL_PATH=${IPEX_LLM_MODEL:-}" in svc["environment"]
    assert "IPEX_LLM_LOAD_IN_LOW_BIT=${IPEX_LLM_LOAD_IN_LOW_BIT:-sym_int4}" in svc["environment"]


def test_ipex_xpu_base_services_unchanged(base):
    standalone = _load(IPEX_XPU_STANDALONE)
    for name, definition in base["services"].items():
        if name == SERVICE:
            continue
        assert standalone["services"][name] == definition


def test_ipex_xpu_top_level_volumes_match_base(base):
    standalone = _load(IPEX_XPU_STANDALONE)
    assert standalone.get("volumes") == base.get("volumes")


# --- Intel IPEX-LLM XPU (WSL2) --------------------------------------------


def test_ipex_xpu_wsl_standalone_equals_base_plus_overlay(base):
    overlay = _load(IPEX_XPU_WSL_OVERLAY)
    standalone = _load(IPEX_XPU_WSL_STANDALONE)
    assert standalone == _merge_overlay_with_new_services(base, overlay)


def test_ipex_xpu_wsl_odysseus_adds_only_overlay(base):
    standalone = _load(IPEX_XPU_WSL_STANDALONE)
    svc = standalone["services"][SERVICE]
    base_svc = base["services"][SERVICE]

    # WSL build target and bridge device; no /dev/dri or group_add.
    assert svc["build"] == {"context": ".", "dockerfile": "Dockerfile.ipex"}
    assert "devices" not in base_svc
    assert svc["devices"] == ["/dev/dxg"]
    assert "group_add" not in svc

    # WSL library volume is appended.
    added_volumes = set(svc["volumes"]) - set(base_svc["volumes"])
    assert added_volumes == {"/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro"}

    # LD_LIBRARY_PATH is appended to environment.
    added_env = set(svc["environment"]) - set(base_svc["environment"])
    assert added_env == {"LD_LIBRARY_PATH=/usr/lib/wsl/lib"}

    # No NVIDIA-specific keys, no native-Linux DRI groups.
    assert "deploy" not in svc


def test_ipex_xpu_wsl_standalone_has_ipex_llm_service():
    overlay = _load(IPEX_XPU_WSL_OVERLAY)
    standalone = _load(IPEX_XPU_WSL_STANDALONE)
    assert "ipex-llm" in standalone["services"]
    assert standalone["services"]["ipex-llm"] == overlay["services"]["ipex-llm"]


def test_ipex_xpu_wsl_ipex_llm_service_shape():
    standalone = _load(IPEX_XPU_WSL_STANDALONE)
    svc = standalone["services"]["ipex-llm"]

    assert svc["image"] == "intelanalytics/ipex-llm-inference-cpp-xpu"
    # WSL bridge device; no /dev/dri or render-group entries.
    assert svc["devices"] == ["/dev/dxg"]
    assert "group_add" not in svc
    # WSL library volume plus the model cache.
    assert "/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro" in svc["volumes"]
    hf_vol = "${APP_DATA_DIR:-./data}/huggingface:/root/.cache/huggingface:z"
    assert hf_vol in svc["volumes"]
    assert "127.0.0.1:8000:8000" in svc["ports"]
    assert "MODEL_PATH=${IPEX_LLM_MODEL:-}" in svc["environment"]
    assert "IPEX_LLM_LOAD_IN_LOW_BIT=${IPEX_LLM_LOAD_IN_LOW_BIT:-sym_int4}" in svc["environment"]
    assert "LD_LIBRARY_PATH=/usr/lib/wsl/lib" in svc["environment"]


def test_ipex_xpu_wsl_base_services_unchanged(base):
    standalone = _load(IPEX_XPU_WSL_STANDALONE)
    for name, definition in base["services"].items():
        if name == SERVICE:
            continue
        assert standalone["services"][name] == definition


def test_ipex_xpu_wsl_top_level_volumes_match_base(base):
    standalone = _load(IPEX_XPU_WSL_STANDALONE)
    assert standalone.get("volumes") == base.get("volumes")

"""Where the runner client connects: explicit argument, then environment, then 127.0.0.1:9876."""

import builtins
import io

import pytest

from ui_bridge_mcp.client import DEFAULT_RUNNER_PORT, LOOPBACK_HOST, UIBridgeClient


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QONTINUI_RUNNER_HOST", raising=False)
    monkeypatch.delenv("QONTINUI_RUNNER_PORT", raising=False)


def test_default_is_the_ipv4_loopback() -> None:
    # Never a resolv.conf nameserver (a DNS resolver on native Linux/macOS, a DNS
    # stub on NAT-mode WSL2) and never ``localhost`` (``::1`` first on Windows).
    assert LOOPBACK_HOST == "127.0.0.1"
    assert DEFAULT_RUNNER_PORT == 9876
    assert UIBridgeClient().base_url == "http://127.0.0.1:9876"


def test_default_opens_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # The old default read the first /etc/resolv.conf nameserver. Record every
    # open rather than raising inside the hook, so a regression that swallows
    # the error still fails here; io.open is patched too because pathlib's
    # read_text goes through it. WSL_DISTRO_NAME is set so that a regression
    # gating the read on WSL detection is exercised as well.
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    opened: list[str] = []
    real_open = io.open

    def recording_open(file: object, *args: object, **kwargs: object) -> object:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)  # type: ignore[call-overload]

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", recording_open)
        m.setattr(io, "open", recording_open)
        client = UIBridgeClient()
    assert opened == []
    assert client.host == "127.0.0.1"


def test_environment_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QONTINUI_RUNNER_HOST", "10.0.0.5")
    monkeypatch.setenv("QONTINUI_RUNNER_PORT", "9877")
    assert UIBridgeClient().base_url == "http://10.0.0.5:9877"


def test_explicit_arguments_override_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QONTINUI_RUNNER_HOST", "10.0.0.5")
    monkeypatch.setenv("QONTINUI_RUNNER_PORT", "9877")
    assert (
        UIBridgeClient(host="runner.local", port=1).base_url == "http://runner.local:1"
    )


def test_empty_environment_values_fall_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QONTINUI_RUNNER_HOST", "")
    monkeypatch.setenv("QONTINUI_RUNNER_PORT", "")
    assert UIBridgeClient().base_url == "http://127.0.0.1:9876"

import shutil

from llm.reasoning.container import run_isolated


def test_unreachable_docker_daemon_is_reported_as_unavailable_not_as_a_failed_program(
    tmp_path, monkeypatch
):
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\necho 'failed to connect to the docker API at unix:///x.sock' >&2\nexit 1\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake))
    result = run_isolated("print(1)", "", 5, "sha256:" + "0" * 64)
    assert result.status == "unavailable"
    assert "docker api" in result.stderr.lower()

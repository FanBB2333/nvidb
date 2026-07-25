import yaml

from nvidb import config


def _use_tmp_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKING_DIR", tmp_path)
    return tmp_path / "config.yml"


def test_view_settings_default_when_config_is_missing(monkeypatch, tmp_path):
    _use_tmp_config(monkeypatch, tmp_path)

    assert config.load_view_settings() == config.DEFAULT_VIEW_SETTINGS


def test_view_settings_round_trip_keeps_servers_and_basic(monkeypatch, tmp_path):
    config_path = _use_tmp_config(monkeypatch, tmp_path)
    config.write_config(
        {
            "basic": {"compact": True, "remote": True},
            "servers": [
                {
                    "hostname": "100.64.0.42",
                    "port": 2222,
                    "username": "l1ght",
                    "nickname": "training-node",
                    "auth": "key",
                }
            ],
        }
    )

    assert config.save_view_settings(
        {
            "mode": "unified",
            "detailed": True,
            "sort": "available",
            "filter": "busy",
            "processes": True,
            "trends": False,
            "group_by_node": False,
            "hide_unsupported": False,
        }
    )

    stored = yaml.safe_load(config_path.read_text())
    assert stored["basic"] == {"compact": True, "remote": True}
    assert stored["servers"][0]["nickname"] == "training-node"
    assert stored["servers"][0]["port"] == 2222
    assert config.load_view_settings() == {
        **config.DEFAULT_VIEW_SETTINGS,
        "mode": "unified",
        "detailed": True,
        "sort": "available",
        "filter": "busy",
        "processes": True,
        "trends": False,
        "group_by_node": False,
        "hide_unsupported": False,
    }


def test_saving_view_settings_preserves_comments_and_layout(monkeypatch, tmp_path):
    config_path = _use_tmp_config(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    original = (
        "basic:\n"
        "  compact: false\n"
        "  remote: true\n"
        "\n"
        "# my GPU boxes\n"
        "servers:\n"
        '  - hostname: "100.64.0.42"\n'
        "    port: 2222              # WSL sshd, port 22 is the Windows host\n"
        '    username: "l1ght"\n'
    )
    config_path.write_text(original)

    assert config.save_view_settings(config.DEFAULT_VIEW_SETTINGS)
    once = config_path.read_text()
    assert config.save_view_settings({**config.DEFAULT_VIEW_SETTINGS, "mode": "unified"})
    twice = config_path.read_text()

    for text in (once, twice):
        assert "# my GPU boxes\nservers:" in text
        assert "# WSL sshd, port 22 is the Windows host" in text
        assert text.startswith("basic:\n  compact: false\n  remote: true\n")
    # Rewriting replaces the block instead of appending a second one.
    assert twice.count("view:") == 1
    assert once.count("\nservers:") == twice.count("\nservers:") == 1
    assert config.load_view_settings()["mode"] == "unified"


def test_invalid_view_settings_fall_back_to_defaults(monkeypatch, tmp_path):
    config_path = _use_tmp_config(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "view": {
                    "mode": "galaxy",
                    "sort": "temperature",
                    "filter": "all",
                    "detailed": 1,
                    "unknown": "ignored",
                }
            }
        )
    )

    settings = config.load_view_settings()

    assert settings["mode"] == "nodes"
    assert settings["sort"] == "node"
    assert settings["filter"] == "all"
    assert settings["detailed"] is True
    assert "unknown" not in settings

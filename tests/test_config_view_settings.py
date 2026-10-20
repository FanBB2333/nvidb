import stat

import yaml

from nvidb import config
from nvidb.data_modules import ServerListInfo


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
                    "username": "alice",
                    "nickname": "training-node",
                    "auth": "key",
                }
            ],
        }
    )

    config.save_view_settings(
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
        '    username: "alice"\n'
    )
    config_path.write_text(original)

    config.save_view_settings(config.DEFAULT_VIEW_SETTINGS)
    once = config_path.read_text()
    config.save_view_settings({**config.DEFAULT_VIEW_SETTINGS, "mode": "unified"})
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


def test_quoted_false_view_settings_stay_false():
    settings = config.normalize_view_settings({"mouse": "false", "detailed": "Yes", "trends": "off"})

    assert settings["mouse"] is False
    assert settings["detailed"] is True
    assert settings["trends"] is False
    # An unrecognised string keeps the default rather than counting as true.
    assert config.normalize_view_settings({"mouse": "sometimes"})["mouse"] is True


def test_an_empty_server_list_reads_back_as_a_list(monkeypatch, tmp_path):
    config_path = _use_tmp_config(monkeypatch, tmp_path)
    config.save_view_settings({"mode": "unified"})

    assert config.load_config()["servers"] == []
    assert len(ServerListInfo.from_yaml(config_path)) == 0
    # Files written by older versions hold a bare `servers:`.
    config_path.write_text("servers:\n")
    assert len(ServerListInfo.from_yaml(config_path)) == 0


def test_saving_keeps_the_config_file_permissions(monkeypatch, tmp_path):
    config_path = _use_tmp_config(monkeypatch, tmp_path)
    config.write_config({"servers": [{"hostname": "h", "password": "secret"}]})
    config_path.chmod(0o600)

    config.save_view_settings({"mode": "unified"})
    config.write_config(config.load_config())

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert [path.name for path in tmp_path.iterdir()] == ["config.yml"]

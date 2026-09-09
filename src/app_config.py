import copy
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "reaction_defaults.json"

AZURE_PREFIX = "MemeMorph:"


def _coerce_value(value):
    """
    Azure App Configuration values normally arrive as strings.
    Convert common primitive types so threshold values remain numeric.
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    lower = stripped.lower()

    if lower == "true":
        return True

    if lower == "false":
        return False

    if lower == "null":
        return None

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        pass

    if (
        (stripped.startswith("{") and stripped.endswith("}"))
        or
        (stripped.startswith("[") and stripped.endswith("]"))
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    return value


def _set_nested(config, key_path, value):
    """
    Convert:
        reactions:speed:squint_delta_min

    into:
        config["reactions"]["speed"]["squint_delta_min"]
    """
    parts = [
        part
        for part in key_path.split(":")
        if part
    ]

    if not parts:
        return

    cursor = config

    for part in parts[:-1]:
        current = cursor.get(part)

        if not isinstance(current, dict):
            cursor[part] = {}

        cursor = cursor[part]

    cursor[parts[-1]] = _coerce_value(value)


def load_local_config(path=DEFAULT_CONFIG_PATH):
    """
    Local configuration is always the source of truth/fallback.
    MemeMorph can run with no Azure account or network connection.
    """
    with open(path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _load_azure_overrides():
    """
    Load MemeMorph:* keys once at startup.

    No dynamic refresh is used, which keeps request volume extremely low.
    Authentication uses DefaultAzureCredential so secrets do not need to
    be committed to the repository.
    """
    endpoint = os.getenv(
        "AZURE_APPCONFIG_ENDPOINT",
        ""
    ).strip()

    if not endpoint:
        return {}, "LOCAL"

    try:
        from azure.appconfiguration.provider import (
            SettingSelector,
            load,
        )
        from azure.identity import DefaultAzureCredential

    except ImportError:
        print(
            "[CONFIG] Azure packages are not installed. "
            "Using local configuration."
        )
        return {}, "LOCAL"

    try:
        credential = DefaultAzureCredential()

        provider = load(
            endpoint=endpoint,
            credential=credential,
            selects={
                SettingSelector(
                    key_filter=f"{AZURE_PREFIX}*",
                    label_filter="\0",
                )
            },
            trim_prefixes={AZURE_PREFIX},
            startup_timeout=5,
            replica_discovery_enabled=False,
        )

        overrides = {
            key: provider[key]
            for key in provider
        }

        print(
            f"[CONFIG] Loaded {len(overrides)} "
            "Azure App Configuration override(s)."
        )

        return overrides, "AZURE"

    except Exception as error:
        print(
            "[CONFIG] Azure App Configuration unavailable. "
            "Using local configuration."
        )

        print(
            f"[CONFIG] {type(error).__name__}: {error}"
        )

        return {}, "LOCAL"


def load_mememorph_config():
    """
    Merge order:

        local JSON defaults
                ↓
        Azure overrides, if available

    This means Azure can change thresholds without making the app depend
    on Azure for startup.
    """
    local_config = load_local_config()
    config = copy.deepcopy(local_config)

    azure_overrides, source = _load_azure_overrides()

    for key_path, value in azure_overrides.items():
        _set_nested(
            config,
            key_path,
            value,
        )

    config["_meta"] = {
        "source": source,
        "azure_override_count": len(
            azure_overrides
        ),
    }

    return config


def get_config_value(config, dotted_path, default=None):
    """
    Convenience helper:

        get_config_value(
            config,
            "reactions.speed.squint_delta_min"
        )
    """
    cursor = config

    for part in dotted_path.split("."):
        if not isinstance(cursor, dict):
            return default

        if part not in cursor:
            return default

        cursor = cursor[part]

    return cursor


if __name__ == "__main__":
    config = load_mememorph_config()

    print()
    print("MemeMorph Configuration")
    print("-----------------------")
    print(
        "Source:",
        config["_meta"]["source"]
    )
    print(
        "Azure overrides:",
        config["_meta"][
            "azure_override_count"
        ]
    )
    print(
        "Speed squint delta:",
        get_config_value(
            config,
            "reactions.speed.squint_delta_min"
        )
    )

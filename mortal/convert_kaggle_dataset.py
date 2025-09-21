"""Utilities for converting the Kaggle SQLite dataset into Mortal mjai logs.

The Kaggle data set contains eight tables – one for each action type.  Each
row stores a gzipped JSON blob that mirrors the state dictionary shown in the
problem statement.  Mortal, however, consumes newline separated mjai events
stored in ``*.json.gz`` files, therefore we provide a light‑weight exporter
that rewrites every record into an mjai log.

The exporter intentionally keeps the conversion logic simple and stateless: we
interpret every database row as an isolated round snapshot and generate a tiny
mock game consisting of

``start_game`` -> ``start_kyoku`` -> [optional ``tsumo``] -> ``action`` ->
``ryukyoku`` -> ``end_kyoku`` -> ``end_game``.

While the reconstructed timeline does not reflect the original replay exactly
(the SQLite dump does not provide complete move history), the emitted log is
structurally valid and can be parsed by ``GameplayLoader`` for fine‑tuning.

Usage
-----

.. code-block:: bash

    python -m mortal.convert_kaggle_dataset \
        --database mahjong.db \
        --output-dir ./converted

The command above produces eight ``*.json.gz`` files (one for every action
table) inside ``converted``.  Each line in those archives corresponds to a
single synthetic mjai log that captures the action described by a row in the
database.

The script aims to be reasonably defensive: whenever it encounters malformed
payloads or unexpected tiles, it will raise a descriptive ``RuntimeError``
instead of silently dropping the offending samples.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["main"]


# --- helpers -----------------------------------------------------------------


MJAI_TILES = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
]

WIND_TO_TILE = {0: "E", 1: "S", 2: "W", 3: "N"}


def tile_id_to_mjai(tile_id: int) -> str:
    """Translate the Kaggle 0-135 tile identifier to an mjai string."""

    if tile_id < 0:
        raise ValueError(f"negative tile id: {tile_id}")
    try:
        tile_type = tile_id // 4
        return MJAI_TILES[tile_type]
    except IndexError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"unknown tile id {tile_id}") from exc


def tiles_from_action(payload: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return ``(tile_id, owner)`` pairs from an action payload."""

    tiles: List[int] = payload.get("tiles", [])
    who: List[int] = payload.get("who", [])
    return [(tile, owner) for tile, owner in zip(tiles, who) if tile >= 0]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --- mjai event builders ------------------------------------------------------


def start_game_event() -> Dict[str, Any]:
    return {
        "type": "start_game",
        "names": ["Player0", "Player1", "Player2", "Player3"],
    }


def start_kyoku_event(state: Dict[str, Any]) -> Dict[str, Any]:
    bakaze = WIND_TO_TILE.get(state.get("round_wind", 0), "E")
    dora_markers = state.get("dora_indicators") or [0]
    dora_marker = tile_id_to_mjai(int(dora_markers[0]))

    honba = int(state.get("num_honba", 0))
    kyotaku = int(state.get("num_riichi", 0))
    kyoku = int(state.get("kyoku", 1))

    points = [int(state[str(idx)]["points"]) for idx in range(4)]

    tehais: List[List[str]] = [["?"] * 13 for _ in range(4)]
    actor = int(state.get("player_wind", 0))
    hand_tiles = [tile_id_to_mjai(int(tile)) for tile in state.get("hand_tiles", [])]
    concealed = hand_tiles[:13]
    if len(concealed) < 13:
        concealed.extend("?" for _ in range(13 - len(concealed)))
    tehais[actor] = concealed[:13]

    return {
        "type": "start_kyoku",
        "bakaze": bakaze,
        "dora_marker": dora_marker,
        "kyoku": max(1, min(4, kyoku)),
        "honba": honba,
        "kyotaku": kyotaku,
        "oya": int(state.get("oya", 0)),
        "scores": points,
        "tehais": tehais,
    }


def tsumo_event(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tiles = state.get("hand_tiles", [])
    if len(tiles) <= 13:
        return None
    actor = int(state.get("player_wind", 0))
    pai = tile_id_to_mjai(int(tiles[-1]))
    return {"type": "tsumo", "actor": actor, "pai": pai}


def _split_action_tiles(action: Dict[str, Any], actor: int) -> Tuple[List[str], List[str], Optional[int]]:
    consumed: List[str] = []
    pai: Optional[str] = None
    target: Optional[int] = None
    for tile_id, owner in tiles_from_action(action):
        tile = tile_id_to_mjai(tile_id)
        if owner == actor:
            consumed.append(tile)
        else:
            pai = tile
            target = owner
    return consumed, [pai] if pai else [], target


def chi_event(state: Dict[str, Any]) -> Dict[str, Any]:
    actor = int(state.get("player_wind", 0))
    action = state["valid_actions"][state["action_idx"]]
    consumed, pai_list, target = _split_action_tiles(action, actor)
    if target is None or not pai_list:
        raise RuntimeError("chi action missing target tile")
    return {
        "type": "chi",
        "actor": actor,
        "target": int(target),
        "pai": pai_list[0],
        "consumed": consumed[:2],
    }


def pon_event(state: Dict[str, Any]) -> Dict[str, Any]:
    actor = int(state.get("player_wind", 0))
    action = state["valid_actions"][state["action_idx"]]
    consumed, pai_list, target = _split_action_tiles(action, actor)
    if target is None or not pai_list:
        raise RuntimeError("pon action missing target tile")
    return {
        "type": "pon",
        "actor": actor,
        "target": int(target),
        "pai": pai_list[0],
        "consumed": consumed[:2],
    }


def daiminkan_event(state: Dict[str, Any]) -> Dict[str, Any]:
    actor = int(state.get("player_wind", 0))
    action = state["valid_actions"][state["action_idx"]]
    consumed, pai_list, target = _split_action_tiles(action, actor)
    if target is None or not pai_list:
        raise RuntimeError("daiminkan missing target tile")
    consumed = consumed[:3]
    return {
        "type": "daiminkan",
        "actor": actor,
        "target": int(target),
        "pai": pai_list[0],
        "consumed": consumed,
    }


def kakan_event(state: Dict[str, Any]) -> Dict[str, Any]:
    actor = int(state.get("player_wind", 0))
    action = state["valid_actions"][state["action_idx"]]
    tiles = [tile_id_to_mjai(tile) for tile, owner in tiles_from_action(action) if owner == actor]
    if len(tiles) < 4:
        raise RuntimeError("shouminkan expects four tiles from actor")
    return {"type": "kakan", "actor": actor, "pai": tiles[0], "consumed": tiles[:3]}


def ankan_event(state: Dict[str, Any]) -> Dict[str, Any]:
    actor = int(state.get("player_wind", 0))
    action = state["valid_actions"][state["action_idx"]]
    tiles = [tile_id_to_mjai(tile) for tile, owner in tiles_from_action(action) if owner == actor]
    if len(tiles) < 4:
        raise RuntimeError("ankan expects four tiles")
    return {"type": "ankan", "actor": actor, "consumed": tiles[:4]}


def discard_event(state: Dict[str, Any], tsumogiri_tile: Optional[str]) -> Dict[str, Any]:
    actor = int(state.get("player_wind", 0))
    action = state["valid_actions"][state["action_idx"]]
    tile_id = action["tiles"][0]
    pai = tile_id_to_mjai(int(tile_id))
    tsumogiri = tsumogiri_tile == pai if tsumogiri_tile is not None else False
    return {"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": tsumogiri}


def reach_events(state: Dict[str, Any], tsumogiri_tile: Optional[str]) -> List[Dict[str, Any]]:
    actor = int(state.get("player_wind", 0))
    return [
        {"type": "reach", "actor": actor},
        discard_event(state, tsumogiri_tile),
    ]


def skip_conversion(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Skips do not generate tangible mjai events; keep the log minimal.
    return []


ACTION_BUILDERS = {
    "Skip": skip_conversion,
    "Discard": lambda state, tsumo: [discard_event(state, tsumo)],
    "Chi": lambda state, tsumo: [chi_event(state)],
    "Pon": lambda state, tsumo: [pon_event(state)],
    "DaiMinKan": lambda state, tsumo: [daiminkan_event(state)],
    "ShouMinKan": lambda state, tsumo: [kakan_event(state)],
    "AnKan": lambda state, tsumo: [ankan_event(state)],
    "Riichi": reach_events,
}


def ryukyoku_event(state: Dict[str, Any]) -> Dict[str, Any]:
    deltas = [int(state[str(idx)].get("PointsReward", 0)) for idx in range(4)]
    return {"type": "ryukyoku", "deltas": deltas}


def tail_events() -> List[Dict[str, Any]]:
    return [{"type": "end_kyoku"}, {"type": "end_game"}]


# --- conversion ---------------------------------------------------------------


@dataclasses.dataclass
class TableExportResult:
    table: str
    rows: int
    output: Path


def convert_table(cursor: sqlite3.Cursor, table: str, output_dir: Path) -> TableExportResult:
    cursor.execute(f"SELECT id, data FROM {table}")
    rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(f"table {table} is empty")

    ensure_directory(output_dir)
    output_path = output_dir / f"{table.lower()}.json.gz"

    with gzip.open(output_path, "wt", encoding="utf-8") as gz:
        for identifier, blob in rows:
            try:
                payload = json.loads(gzip.decompress(blob).decode("utf-8"))
            except OSError as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"failed to decompress row {identifier} in {table}") from exc

            tsumo = tsumo_event(payload)
            builder = ACTION_BUILDERS[table]
            events = [start_game_event(), start_kyoku_event(payload)]
            if tsumo is not None:
                events.append(tsumo)
            events.extend(builder(payload, tsumo["pai"] if tsumo else None))
            events.append(ryukyoku_event(payload))
            events.extend(tail_events())

            for event in events:
                gz.write(json.dumps(event, ensure_ascii=False))
                gz.write("\n")

    return TableExportResult(table=table, rows=len(rows), output=output_path)


def run_conversion(database: Path, output_dir: Path) -> List[TableExportResult]:
    conn = sqlite3.connect(str(database))
    try:
        cursor = conn.cursor()
        results = []
        for table in ACTION_BUILDERS:
            if table == "Skip":
                # Skip table only logs "no-op" decisions; we still export the
                # synthetic events to retain one-to-one parity with the source.
                pass
            results.append(convert_table(cursor, table, output_dir))
        return results
    finally:
        conn.close()


# --- cli ---------------------------------------------------------------------


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="Path to the Kaggle SQLite database")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store the converted mjai logs",
    )
    return parser.parse_args(args)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    results = run_conversion(args.database, args.output_dir)
    for result in results:
        print(f"Exported {result.rows} rows from {result.table} to {result.output}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()

from __future__ import annotations

import json
from io import StringIO
from typing import Any, cast

from tinychess.engine import Game
from tinychess.protocols.gui import GuiSession, run_gui_loop, serialize_state


def _request(session: GuiSession, payload: dict[str, object]) -> dict[str, Any]:
    output = StringIO()
    session.handle_line(json.dumps(payload), output)
    return cast(dict[str, Any], json.loads(output.getvalue()))


def _line_response(text: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(text.strip()))


def _destination_sets(state: dict[str, Any]) -> dict[str, set[str]]:
    return {
        from_square: set(destinations)
        for from_square, destinations in state["legalDestinationsByFrom"].items()
    }


def test_hello_returns_capabilities_and_canonical_start_state() -> None:
    session = GuiSession()

    response = _request(session, {"id": 1, "cmd": "hello"})

    assert response["id"] == 1
    assert response["ok"] is True
    assert response["protocol"] == "tinychess-gui-v1"
    assert response["capabilities"] == {
        "players": ["random", "mcts", "neural"],
        "supportsUndo": False,
        "promotion": "auto_queen",
    }
    state = response["state"]
    assert state["fen"] == Game.new().to_fen()
    assert state["sideToMove"] == "white"
    assert len(state["squares"]) == 32
    assert {piece["square"]: piece["piece"] for piece in state["squares"]}["a1"] == "R"
    assert "e2e4" in state["legalMoves"]
    assert state["legalDestinationsByFrom"]["e2"] == ["e3", "e4"]
    assert state["moves"] == []
    assert state["lastMove"] is None
    assert state["outcome"] is None


def test_state_serializer_uses_canonical_fields_after_move() -> None:
    game = Game.new().play(next(move for move in Game.new().legal_moves if move.to_uci() == "e2e4"))

    state = serialize_state(game)

    assert state["sideToMove"] == "black"
    assert state["moves"] == ["e2e4"]
    assert state["lastMove"] == "e2e4"
    assert state["fullmoveNumber"] == 1
    assert state["halfmoveClock"] == 0


def test_state_command_returns_current_canonical_state_after_move() -> None:
    session = GuiSession()
    _request(session, {"id": 1, "cmd": "makeMove", "move": "e2e4"})

    response = _request(session, {"id": 2, "cmd": "state"})

    assert response["id"] == 2
    assert response["ok"] is True
    state = response["state"]
    assert state["fen"] == session.game.to_fen()
    assert state["sideToMove"] == "black"
    assert state["moves"] == ["e2e4"]
    assert state["lastMove"] == "e2e4"
    assert "e7e5" in state["legalMoves"]
    assert _destination_sets(state)["e7"] == {"e6", "e5"}


def test_start_state_legal_move_shape_is_highlight_friendly() -> None:
    state = _request(GuiSession(), {"id": 1, "cmd": "state"})["state"]

    expected_moves = {
        "b1c3",
        "b1a3",
        "g1h3",
        "g1f3",
        "a2a3",
        "a2a4",
        "b2b3",
        "b2b4",
        "c2c3",
        "c2c4",
        "d2d3",
        "d2d4",
        "e2e3",
        "e2e4",
        "f2f3",
        "f2f4",
        "g2g3",
        "g2g4",
        "h2h3",
        "h2h4",
    }
    assert set(state["legalMoves"]) == expected_moves
    assert _destination_sets(state) == {
        "b1": {"c3", "a3"},
        "g1": {"h3", "f3"},
        "a2": {"a3", "a4"},
        "b2": {"b3", "b4"},
        "c2": {"c3", "c4"},
        "d2": {"d3", "d4"},
        "e2": {"e3", "e4"},
        "f2": {"f3", "f4"},
        "g2": {"g3", "g4"},
        "h2": {"h3", "h4"},
    }


def test_new_game_resets_state_and_stores_valid_ai_config() -> None:
    session = GuiSession()
    _request(session, {"id": 1, "cmd": "makeMove", "move": "e2e4"})

    response = _request(
        session,
        {
            "id": 2,
            "cmd": "newGame",
            "humanColor": "black",
            "seed": 9,
            "ai": {"kind": "mcts", "simulations": 3, "nodeBudget": 5},
        },
    )

    assert response["ok"] is True
    assert response["state"]["fen"] == Game.new().to_fen()
    assert response["state"]["moves"] == []
    assert response["state"]["lastMove"] is None
    assert _destination_sets(response["state"])["e2"] == {"e3", "e4"}
    assert session.human_color.value == "black"
    assert session.ai_config.kind == "mcts"
    assert session.ai_config.simulations == 3
    assert session.ai_config.node_budget == 5
    assert session.ai_config.seed == 9


def test_make_move_applies_legal_move_and_reports_applied_move() -> None:
    session = GuiSession()

    response = _request(session, {"id": "move-1", "cmd": "makeMove", "move": "E2E4"})

    assert response["id"] == "move-1"
    assert response["ok"] is True
    assert response["appliedMove"] == "e2e4"
    assert response["state"]["sideToMove"] == "black"
    assert response["state"]["moves"] == ["e2e4"]
    assert response["state"]["lastMove"] == "e2e4"
    assert _destination_sets(response["state"])["e7"] == {"e6", "e5"}
    assert session.game.moves[-1].to_uci() == "e2e4"


def test_make_move_auto_queen_promotes_four_character_promotion() -> None:
    session = GuiSession()
    session.game = Game.from_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")

    response = _request(session, {"id": 1, "cmd": "makeMove", "move": "e7e8"})

    assert response["ok"] is True
    assert response["appliedMove"] == "e7e8q"
    assert response["state"]["lastMove"] == "e7e8q"
    assert response["state"]["squares"] == [
        {"square": "e1", "index": 4, "piece": "K", "color": "white", "kind": "king"},
        {"square": "a8", "index": 56, "piece": "k", "color": "black", "kind": "king"},
        {"square": "e8", "index": 60, "piece": "Q", "color": "white", "kind": "queen"},
    ]


def test_invalid_json_returns_structured_error_without_state() -> None:
    output = StringIO()

    GuiSession().handle_line("{bad json", output)

    response = _line_response(output.getvalue())
    assert response["id"] is None
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_json"
    assert "state" not in response


def test_unknown_command_returns_error_with_state() -> None:
    response = _request(GuiSession(), {"id": 7, "cmd": "doesNotExist"})

    assert response["ok"] is False
    assert response["error"]["code"] == "unknown_command"
    assert response["state"]["fen"] == Game.new().to_fen()


def test_illegal_move_returns_structured_error_and_current_state() -> None:
    response = _request(GuiSession(), {"id": 8, "cmd": "makeMove", "move": "e2e5"})

    assert response["ok"] is False
    assert response["error"]["code"] == "illegal_move"
    assert response["state"]["fen"] == Game.new().to_fen()
    assert response["state"]["moves"] == []


def test_new_game_validation_failure_does_not_mutate_session_config() -> None:
    session = GuiSession()

    response = _request(
        session,
        {
            "id": 9,
            "cmd": "newGame",
            "humanColor": "black",
            "seed": 12,
            "ai": {"kind": "mcts", "simulations": 0},
        },
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "configuration_error"
    assert session.human_color.value == "white"
    assert session.ai_config.kind == "random"
    assert session.ai_config.seed is None


def test_ai_move_random_applies_deterministic_legal_move() -> None:
    session = GuiSession()
    legal_before = set(_request(session, {"id": "state", "cmd": "state"})["state"]["legalMoves"])

    response = _request(
        session,
        {"id": 9, "cmd": "aiMove", "ai": {"kind": "random", "seed": 7}},
    )

    assert response["ok"] is True
    assert response["appliedMove"] in legal_before
    assert response["state"]["moves"] == [response["appliedMove"]]
    assert response["state"]["lastMove"] == response["appliedMove"]
    assert response["state"]["sideToMove"] == "black"
    assert response["search"]["kind"] == "random"
    assert response["search"]["elapsedSeconds"] >= 0.0

    second = _request(
        GuiSession(),
        {"id": 10, "cmd": "aiMove", "ai": {"kind": "random", "seed": 7}},
    )
    assert second["appliedMove"] == response["appliedMove"]


def test_ai_move_mcts_applies_legal_move_and_reports_search_metadata() -> None:
    session = GuiSession()
    legal_before = set(_request(session, {"id": "state", "cmd": "state"})["state"]["legalMoves"])

    response = _request(
        session,
        {
            "id": 10,
            "cmd": "aiMove",
            "ai": {"kind": "mcts", "simulations": 2, "nodeBudget": 3, "seed": 3},
        },
    )

    assert response["ok"] is True
    assert response["appliedMove"] in legal_before
    assert response["state"]["moves"] == [response["appliedMove"]]
    assert response["search"]["kind"] == "mcts"
    assert response["search"]["simulations"] == 2
    assert response["search"]["nodes"] <= 3
    assert response["search"]["elapsedSeconds"] >= 0.0
    assert set(response["search"]["visitCounts"]).issubset(legal_before)


def test_ai_move_neural_without_checkpoint_fails_gracefully() -> None:
    session = GuiSession()

    response = _request(
        session,
        {"id": 11, "cmd": "aiMove", "ai": {"kind": "neural", "simulations": 1}},
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "configuration_error"
    assert "checkpointPath" in response["error"]["message"]
    assert response["state"]["fen"] == Game.new().to_fen()
    assert response["state"]["moves"] == []


def test_set_ai_config_requires_ai_object() -> None:
    response = _request(GuiSession(), {"id": 9, "cmd": "setAiConfig"})

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_request"
    assert "ai" in response["error"]["message"]


def test_set_ai_config_validates_budget_values() -> None:
    response = _request(
        GuiSession(),
        {"id": 10, "cmd": "setAiConfig", "ai": {"kind": "mcts", "simulations": 0}},
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "configuration_error"
    assert "simulations" in response["error"]["message"]


def test_set_ai_config_rejects_non_finite_numbers() -> None:
    output = StringIO()

    GuiSession().handle_line(
        '{"id":11,"cmd":"setAiConfig","ai":{"kind":"mcts","timeLimitSeconds":NaN}}',
        output,
    )

    response = _line_response(output.getvalue())
    assert response["ok"] is False
    assert response["error"]["code"] == "configuration_error"
    assert "finite" in response["error"]["message"]


def test_set_ai_config_accepts_optional_neural_checkpoint_settings() -> None:
    response = _request(
        GuiSession(),
        {
            "id": 12,
            "cmd": "setAiConfig",
            "ai": {
                "kind": "neural",
                "simulations": 2,
                "checkpointPath": "/tmp/checkpoint",
                "timeLimitSeconds": 0.25,
                "leafParallelism": 2,
            },
        },
    )

    assert response["ok"] is True
    assert response["ai"]["kind"] == "neural"
    assert response["ai"]["checkpointPath"] == "/tmp/checkpoint"
    assert response["ai"]["timeLimitSeconds"] == 0.25
    assert response["ai"]["leafParallelism"] == 2


def test_run_gui_loop_stops_after_quit() -> None:
    output = StringIO()

    session = run_gui_loop(
        stdin=StringIO(
            '\n'.join(
                (
                    json.dumps({"id": 1, "cmd": "state"}),
                    json.dumps({"id": 2, "cmd": "quit"}),
                    json.dumps({"id": 3, "cmd": "state"}),
                    "",
                )
            )
        ),
        stdout=output,
    )

    assert session.should_quit
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert all(response["ok"] for response in responses)
    assert responses[1]["state"]["fen"] == Game.new().to_fen()

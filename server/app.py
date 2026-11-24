"""Local API and the page that plays against it."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.game import Game

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

app = FastAPI(title="Gomoku")
game = Game()
lock = threading.Lock()


class NewGame(BaseModel):
    mode: str = "play"
    human_side: int = 1


class Move(BaseModel):
    row: int
    col: int
    player: int | None = None


class Side(BaseModel):
    side: int


class Mode(BaseModel):
    mode: str


def _run(action) -> dict:
    with lock:
        try:
            action()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return game.to_dict()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/state")
def state() -> dict:
    with lock:
        return game.to_dict()


@app.post("/api/new")
def new_game(body: NewGame) -> dict:
    return _run(lambda: game.new(body.mode, body.human_side))


@app.post("/api/move")
def move(body: Move) -> dict:
    return _run(lambda: game.human_move(body.row, body.col, body.player))


@app.post("/api/analyze")
def analyze() -> dict:
    return _run(game.analyze)


@app.post("/api/undo")
def undo() -> dict:
    return _run(game.undo)


@app.post("/api/side")
def side(body: Side) -> dict:
    return _run(lambda: game.set_side(body.side))


@app.post("/api/mode")
def mode(body: Mode) -> dict:
    return _run(lambda: game.set_mode(body.mode))


@app.post("/api/bot")
def bot() -> dict:
    return _run(game.bot_move)


app.mount("/static", StaticFiles(directory=WEB), name="static")

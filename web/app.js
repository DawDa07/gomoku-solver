const SIZE = 15;
const BLACK = 1;
const WHITE = 2;
const STARS = new Set(["3,3", "3,7", "3,11", "7,3", "7,7", "7,11", "11,3", "11,7", "11,11"]);

const boardEl = document.querySelector("#board");
const colsEl = document.querySelector("#cols");
const rowsEl = document.querySelector("#rows");
const statusEl = document.querySelector("#status");
const turnEl = document.querySelector("#turn");

let state = null;
let paint = BLACK;
let busy = false;
let error = "";

for (let col = 0; col < SIZE; col += 1) {
  const label = document.createElement("span");
  label.textContent = String.fromCharCode(65 + col);
  colsEl.appendChild(label);
}
for (let row = 0; row < SIZE; row += 1) {
  const label = document.createElement("span");
  label.textContent = String(row + 1);
  rowsEl.appendChild(label);
}

document.querySelector("#mode-play").addEventListener("click", () => {
  if (state && state.mode !== "play") call("/api/mode", { mode: "play" });
});
document.querySelector("#mode-analyze").addEventListener("click", () => {
  if (state && state.mode !== "analyze") call("/api/mode", { mode: "analyze" });
});
document.querySelector("#play-black").addEventListener("click", () => chooseColor(BLACK));
document.querySelector("#play-white").addEventListener("click", () => chooseColor(WHITE));
document.querySelector("#paint-black").addEventListener("click", () => setPaint(BLACK));
document.querySelector("#paint-white").addEventListener("click", () => setPaint(WHITE));
document.querySelector("#side-black").addEventListener("click", () => call("/api/side", { side: BLACK }));
document.querySelector("#side-white").addEventListener("click", () => call("/api/side", { side: WHITE }));
document.querySelector("#find").addEventListener("click", () => call("/api/analyze", null, "Thinking…"));
document.querySelector("#bot").addEventListener("click", () => call("/api/bot", null, "Thinking…"));
document.querySelector("#undo").addEventListener("click", () => call("/api/undo"));
document.querySelector("#reset").addEventListener("click", () => {
  if (!state) return;
  call("/api/new", { mode: state.mode, human_side: state.human_side });
});

load();

async function load() {
  const response = await fetch("/api/state");
  state = await response.json();
  render();
}

function chooseColor(side) {
  if (!state || busy) return;
  if (state.mode === "play" && state.human_side === side) return;
  call("/api/new", { mode: "play", human_side: side }, side === WHITE ? "Thinking…" : null);
}

function setPaint(color) {
  paint = color;
  render();
}

async function call(path, body, pending) {
  if (busy) return;
  busy = true;
  error = "";
  if (pending) statusEl.textContent = pending;
  renderBoard();
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json();
    if (!response.ok) {
      error = typeof data.detail === "string" ? data.detail : "Request failed.";
    } else {
      state = data;
    }
  } catch (err) {
    error = "Could not reach the server.";
  } finally {
    busy = false;
    render();
  }
}

function onCell(row, col) {
  if (!canPlace() || state.board[row][col] !== 0) return;
  const body = { row, col };
  if (state.mode === "analyze") body.player = paint;
  call("/api/move", body, state.mode === "play" ? "Thinking…" : null);
}

function canPlace() {
  if (!state || busy || state.winner || state.draw) return false;
  if (state.mode === "analyze") return true;
  return state.side_to_move === state.human_side;
}

function placeColor() {
  return state.mode === "analyze" ? paint : state.human_side;
}

function render() {
  if (!state) return;
  renderBoard();
  const play = state.mode === "play";
  document.querySelector("#play-panel").hidden = !play;
  document.querySelector("#analyze-panel").hidden = play;
  document.querySelector("#mode-play").setAttribute("aria-pressed", String(play));
  document.querySelector("#mode-analyze").setAttribute("aria-pressed", String(!play));
  document.querySelector("#play-black").setAttribute("aria-pressed", String(state.human_side === BLACK));
  document.querySelector("#play-white").setAttribute("aria-pressed", String(state.human_side === WHITE));
  document.querySelector("#paint-black").setAttribute("aria-pressed", String(paint === BLACK));
  document.querySelector("#paint-white").setAttribute("aria-pressed", String(paint === WHITE));
  document.querySelector("#side-black").setAttribute("aria-pressed", String(state.side_to_move === BLACK));
  document.querySelector("#side-white").setAttribute("aria-pressed", String(state.side_to_move === WHITE));

  const botTurn = play && !state.winner && !state.draw && state.side_to_move !== state.human_side;
  const botButton = document.querySelector("#bot");
  botButton.hidden = !botTurn;
  botButton.disabled = busy;

  document.querySelector("#find").disabled = busy || Boolean(state.winner || state.draw);
  document.querySelector("#undo").disabled = busy;
  document.querySelector("#reset").disabled = busy;

  turnEl.replaceChildren();
  const dot = document.createElement("i");
  dot.className = "mini " + (state.side_to_move === BLACK ? "black" : "white");
  turnEl.appendChild(dot);
  const label = document.createElement("span");
  if (state.winner) label.textContent = (state.winner === BLACK ? "Black" : "White") + " won";
  else if (state.draw) label.textContent = "Draw";
  else label.textContent = (state.side_to_move === BLACK ? "Black" : "White") + " to move";
  turnEl.appendChild(label);

  statusEl.textContent = error || state.status;
}

function renderBoard() {
  const suggestion = state && state.suggestion;
  const best = suggestion ? suggestion.row + "," + suggestion.col : "";
  const rest = new Set(
    ((suggestion && suggestion.line) || [])
      .slice(1)
      .map((square) => square.row + "," + square.col)
  );
  const last = state && state.last_move;
  const legal = canPlace();
  const color = state ? placeColor() : BLACK;
  boardEl.replaceChildren();
  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      const key = row + "," + col;
      const stone = state ? state.board[row][col] : 0;
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      if (row === 0) cell.classList.add("edge-t");
      if (row === SIZE - 1) cell.classList.add("edge-b");
      if (col === 0) cell.classList.add("edge-l");
      if (col === SIZE - 1) cell.classList.add("edge-r");
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", squareName(row, col));
      const open = stone === 0 && legal;
      if (open) cell.classList.add("legal");
      else cell.disabled = true;
      if (STARS.has(key)) {
        const star = document.createElement("span");
        star.className = "star-dot";
        cell.appendChild(star);
      }
      if (stone) {
        cell.appendChild(stoneEl(stone, last && last.row === row && last.col === col));
      } else if (key === best) {
        const ghost = stoneEl(state.side_to_move, false);
        ghost.classList.add("ghost");
        cell.appendChild(ghost);
        cell.setAttribute("aria-label", squareName(row, col) + ", suggested");
      } else if (rest.has(key)) {
        const mark = document.createElement("span");
        mark.className = "pv";
        cell.appendChild(mark);
      } else if (open) {
        const preview = stoneEl(color, false);
        preview.classList.add("preview");
        cell.appendChild(preview);
      }
      cell.addEventListener("click", () => onCell(row, col));
      boardEl.appendChild(cell);
    }
  }
}

function stoneEl(player, last) {
  const stone = document.createElement("span");
  stone.className = "stone " + (player === BLACK ? "black" : "white");
  if (last) stone.classList.add("last");
  return stone;
}

function squareName(row, col) {
  return String.fromCharCode(65 + col) + (row + 1);
}

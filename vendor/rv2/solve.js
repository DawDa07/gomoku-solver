// Stdin/stdout front for the rv2 threat proof.
// One JSON object per line: {"moves":[[row,col],...]} with row 0 at the top.
// Answers {"row":7,"col":7,"state":2} or {"move":null}.

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const vm = require("vm");

const dir = __dirname;
const context = vm.createContext({
  console: { log() {}, error() {}, warn() {} },
  BLACK: 1,
  WHITE: 0,
});
for (const name of ["pattern.js", "line.js", "figure.js", "edge.js", "layout.js", "vertex.js"]) {
  vm.runInContext(fs.readFileSync(path.join(dir, name), "utf8"), context, { filename: name });
}

function reply(moves) {
  const squares = moves.map((pair) => (pair[0] << 4) | pair[1]);
  const layout = new context.Layout(squares);
  const attacker = squares.length % 2 ? context.Layout.WHITE : context.Layout.BLACK;
  const other = attacker === context.Layout.BLACK ? context.Layout.WHITE : context.Layout.BLACK;
  let vertex = context.Vertex.estimate(layout, attacker, true, 35);
  if (vertex.state === 0) {
    vertex = context.Vertex.estimate(layout, other, true, 35);
  }
  if (vertex.state === 0) {
    return { move: null };
  }
  const found = vertex.moves();
  if (!found || found.length === 0) {
    return { move: null };
  }
  const square = found[0];
  return { row: square >> 4, col: square & 0xf, state: vertex.state };
}

const input = readline.createInterface({ input: process.stdin, terminal: false });
input.on("line", (line) => {
  let response;
  try {
    const request = JSON.parse(line);
    response = reply(request.moves || []);
  } catch (error) {
    response = { move: null, error: String(error && error.stack ? error.stack : error) };
  }
  process.stdout.write(JSON.stringify(response) + "\n");
});

#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const args = { target: path.resolve(process.cwd(), "webgame") };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--target" && i + 1 < argv.length) {
      args.target = path.resolve(process.cwd(), argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg.startsWith("--target=")) {
      args.target = path.resolve(process.cwd(), arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      console.log("Usage: node patch_webgame_for_rl.js [--target ./webgame]");
      process.exit(0);
    }
  }
  return args;
}

function writeTextFile(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content, "utf8");
  console.log(`[patch] wrote ${filePath}`);
}

const INDEX_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Web Flappy RL</title>
  <link rel="stylesheet" href="./style.css">
</head>
<body>
  <main class="container">
    <h1>Web Flappy RL</h1>
    <p class="hint">Space / Click to flap. Enter to reset after game over.</p>
    <canvas id="game" width="400" height="600" aria-label="Flappy game canvas"></canvas>
    <p class="hint">RL API is exposed as <code>window.__rl</code>.</p>
  </main>
  <script src="./game.js"></script>
</body>
</html>
`;

const STYLE_CSS = `* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: #0d1724;
  color: #f5f7fb;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
}

.container {
  text-align: center;
}

#game {
  border: 3px solid #1e2f45;
  border-radius: 8px;
  image-rendering: pixelated;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
}

.hint {
  margin: 8px 0;
  opacity: 0.85;
}
`;

let GAME_JS = `"use strict";

(function () {
  var canvas = document.getElementById("game");
  var ctx = canvas.getContext("2d");

  var W = canvas.width;
  var H = canvas.height;

  var BIRD_X = 100;
  var BIRD_R = 14;
  var GRAVITY = 900;
  var FLAP_VELOCITY = -300;
  var PIPE_WIDTH = 70;
  var PIPE_GAP = 170;
  var PIPE_SPEED = 150;
  var PIPE_SPAWN_INTERVAL = 1.35;

  var rngState = 1;
  var birdY = H * 0.5;
  var birdVy = 0;
  var pipes = [];
  var spawnTimer = 0;
  var score = 0;
  var alive = true;
  var lastFrameTs = performance.now();

  function setSeed(seed) {
    var v = Number(seed);
    if (!Number.isFinite(v)) {
      v = Date.now() ^ Math.floor(Math.random() * 1000000);
    }
    v = (v >>> 0);
    if (v === 0) {
      v = 1;
    }
    rngState = v;
  }

  function randUnit() {
    rngState ^= rngState << 13;
    rngState ^= rngState >>> 17;
    rngState ^= rngState << 5;
    return (rngState >>> 0) / 4294967296;
  }

  function randomGapY() {
    var minY = 120;
    var maxY = H - 120;
    return minY + randUnit() * (maxY - minY);
  }

  function addPipe(x) {
    pipes.push({
      x: x,
      gapY: randomGapY(),
      passed: false
    });
  }

  function resetGame(seed) {
    setSeed(seed);
    birdY = H * 0.5;
    birdVy = 0;
    pipes = [];
    spawnTimer = 0;
    score = 0;
    alive = true;
    addPipe(W + 120);
    addPipe(W + 320);
    lastFrameTs = performance.now();
  }

  function flap() {
    if (!alive) {
      return;
    }
    birdVy = FLAP_VELOCITY;
  }

  function getNextPipe() {
    for (var i = 0; i < pipes.length; i += 1) {
      if (pipes[i].x + PIPE_WIDTH >= BIRD_X - BIRD_R) {
        return pipes[i];
      }
    }
    return pipes.length > 0 ? pipes[0] : null;
  }

  function getState() {
    var next = getNextPipe();
    var nextPipeDx = next ? (next.x + PIPE_WIDTH - BIRD_X) : W;
    var gapDy = next ? (birdY - next.gapY) : 0;
    return {
      y: birdY,
      vy: birdVy,
      nextPipeDx: nextPipeDx,
      gapDy: gapDy,
      score: score,
      alive: alive
    };
  }

  function collidePipe(pipe) {
    var withinX = (BIRD_X + BIRD_R > pipe.x) && (BIRD_X - BIRD_R < pipe.x + PIPE_WIDTH);
    if (!withinX) {
      return false;
    }

    var gapTop = pipe.gapY - (PIPE_GAP / 2);
    var gapBottom = pipe.gapY + (PIPE_GAP / 2);
    var hitUpper = birdY - BIRD_R < gapTop;
    var hitLower = birdY + BIRD_R > gapBottom;
    return hitUpper || hitLower;
  }

  function update(dt) {
    if (!alive) {
      return;
    }

    birdVy += GRAVITY * dt;
    birdY += birdVy * dt;

    if (birdY - BIRD_R <= 0 || birdY + BIRD_R >= H) {
      alive = false;
    }

    spawnTimer += dt;
    if (spawnTimer >= PIPE_SPAWN_INTERVAL) {
      spawnTimer -= PIPE_SPAWN_INTERVAL;
      addPipe(W + 50);
    }

    for (var i = 0; i < pipes.length; i += 1) {
      var pipe = pipes[i];
      pipe.x -= PIPE_SPEED * dt;
      if (!pipe.passed && pipe.x + PIPE_WIDTH < BIRD_X) {
        pipe.passed = true;
        score += 1;
      }
      if (collidePipe(pipe)) {
        alive = false;
      }
    }

    pipes = pipes.filter(function (pipe) {
      return pipe.x + PIPE_WIDTH > -40;
    });
  }

  function drawBackground() {
    ctx.fillStyle = "#71c5cf";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#4ec0ca";
    ctx.fillRect(0, H - 80, W, 80);
  }

  function drawPipes() {
    ctx.fillStyle = "#2f9e44";
    for (var i = 0; i < pipes.length; i += 1) {
      var pipe = pipes[i];
      var gapTop = pipe.gapY - PIPE_GAP / 2;
      var gapBottom = pipe.gapY + PIPE_GAP / 2;

      ctx.fillRect(pipe.x, 0, PIPE_WIDTH, gapTop);
      ctx.fillRect(pipe.x, gapBottom, PIPE_WIDTH, H - gapBottom);
    }
  }

  function drawBird() {
    ctx.beginPath();
    ctx.fillStyle = "#ffd43b";
    ctx.arc(BIRD_X, birdY, BIRD_R, 0, Math.PI * 2);
    ctx.fill();
    ctx.closePath();
  }

  function drawText() {
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 28px Segoe UI";
    ctx.fillText("Score: " + String(score), 14, 36);

    if (!alive) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 36px Segoe UI";
      ctx.fillText("Game Over", 108, 275);
      ctx.font = "20px Segoe UI";
      ctx.fillText("Press Enter to reset", 112, 315);
    }
  }

  function draw() {
    drawBackground();
    drawPipes();
    drawBird();
    drawText();
  }

  function frame(ts) {
    var dt = (ts - lastFrameTs) / 1000;
    if (dt < 0) {
      dt = 0;
    }
    if (dt > 0.05) {
      dt = 0.05;
    }
    lastFrameTs = ts;
    update(dt);
    draw();
    requestAnimationFrame(frame);
  }

  function onUserFlap() {
    flap();
  }

  window.addEventListener("keydown", function (event) {
    if (event.code === "Space" || event.code === "ArrowUp" || event.code === "KeyW") {
      event.preventDefault();
      onUserFlap();
    }
    if (!alive && event.code === "Enter") {
      resetGame();
    }
  });

  window.addEventListener("pointerdown", function () {
    if (!alive) {
      resetGame();
      return;
    }
    onUserFlap();
  });

  window.__rl = {
    reset: function (seed) {
      resetGame(seed);
    },
    flap: function () {
      onUserFlap();
    },
    getState: function () {
      return getState();
    },
    isDone: function () {
      return !alive;
    }
  };

  resetGame();
  requestAnimationFrame(frame);
}());
`;

const GAME_TEMPLATE_PATH = path.join(__dirname, "webgame_game_template.js");
if (fs.existsSync(GAME_TEMPLATE_PATH)) {
  GAME_JS = fs.readFileSync(GAME_TEMPLATE_PATH, "utf8");
}

const SERVER_JS = `"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");

function resolvePort() {
  var argPort = null;
  var idx = process.argv.indexOf("--port");
  if (idx >= 0 && process.argv[idx + 1]) {
    argPort = Number(process.argv[idx + 1]);
  }
  if (!argPort) {
    var inline = process.argv.find(function (arg) {
      return arg.indexOf("--port=") === 0;
    });
    if (inline) {
      argPort = Number(inline.split("=", 2)[1]);
    }
  }
  if (!argPort && process.env.PORT) {
    argPort = Number(process.env.PORT);
  }
  if (!argPort || !Number.isFinite(argPort)) {
    argPort = 5173;
  }
  return argPort;
}

const PORT = resolvePort();
const HOST = "127.0.0.1";
const ROOT = path.resolve(__dirname);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

function safePath(urlPath) {
  var cleaned = decodeURIComponent(urlPath.split("?")[0]);
  if (cleaned === "/") {
    cleaned = "/index.html";
  }
  var filePath = path.resolve(ROOT, "." + cleaned);
  if (!filePath.startsWith(ROOT)) {
    return null;
  }
  return filePath;
}

const server = http.createServer(function (req, res) {
  var filePath = safePath(req.url || "/");
  if (!filePath) {
    res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, function (err, data) {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
      return;
    }

    var ext = path.extname(filePath).toLowerCase();
    var type = MIME[ext] || "application/octet-stream";
    res.writeHead(200, { "Content-Type": type });
    res.end(data);
  });
});

server.listen(PORT, HOST, function () {
  console.log("[webgame] running at http://" + HOST + ":" + PORT);
});
`;

const PACKAGE_JSON = `{
  "name": "webgame-flappy-rl",
  "version": "1.0.0",
  "private": true,
  "description": "Minimal Flappy Bird web game for RL with window.__rl API",
  "scripts": {
    "start": "node server.js",
    "dev": "node server.js"
  },
  "dependencies": {}
}
`;

function main() {
  const args = parseArgs(process.argv.slice(2));
  const target = args.target;

  writeTextFile(path.join(target, "index.html"), INDEX_HTML);
  writeTextFile(path.join(target, "style.css"), STYLE_CSS);
  writeTextFile(path.join(target, "game.js"), GAME_JS);
  writeTextFile(path.join(target, "server.js"), SERVER_JS);
  writeTextFile(path.join(target, "package.json"), PACKAGE_JSON);

  console.log("");
  console.log(`[patch] webgame is ready at ${target}`);
  console.log("[patch] next commands:");
  console.log(`  cd ${target}`);
  console.log("  npm install");
  console.log("  npm run start -- --port 5173");
}

main();

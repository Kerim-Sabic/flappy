"use strict";

(function () {
  var canvas = document.getElementById("game");
  var ctx = canvas.getContext("2d");

  var W = canvas.width;
  var H = canvas.height;

  var BIRD_X = 100;
  var BIRD_R = 14;
  var BASE_GRAVITY = 900;
  var BASE_FLAP_VELOCITY = -300;
  var BASE_PIPE_WIDTH = 70;
  var BASE_PIPE_GAP = 170;
  var BASE_PIPE_SPEED = 150;
  var BASE_PIPE_SPAWN_INTERVAL = 1.35;
  var BASE_GAP_MARGIN = 120;

  var rngState = 1;
  var birdY = H * 0.5;
  var birdVy = 0;
  var pipes = [];
  var spawnTimer = 0;
  var score = 0;
  var alive = true;
  var lastFrameTs = performance.now();

  var currentGravity = BASE_GRAVITY;
  var currentFlapVelocity = BASE_FLAP_VELOCITY;
  var currentPipeWidth = BASE_PIPE_WIDTH;
  var currentPipeGap = BASE_PIPE_GAP;
  var currentPipeSpeed = BASE_PIPE_SPEED;
  var currentPipeSpawnInterval = BASE_PIPE_SPAWN_INTERVAL;
  var currentGapMargin = BASE_GAP_MARGIN;

  var drConfig = {
    enabled: true,
    gravityScaleMin: 0.96,
    gravityScaleMax: 1.08,
    flapScaleMin: 0.94,
    flapScaleMax: 1.06,
    pipeSpeedScaleMin: 0.92,
    pipeSpeedScaleMax: 1.14,
    pipeGapScaleMin: 0.90,
    pipeGapScaleMax: 1.10,
    spawnScaleMin: 0.88,
    spawnScaleMax: 1.10,
    gapMarginScaleMin: 0.90,
    gapMarginScaleMax: 1.12
  };

  function clamp(v, minV, maxV) {
    if (v < minV) {
      return minV;
    }
    if (v > maxV) {
      return maxV;
    }
    return v;
  }

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

  function mixRange(minBase, maxBase, progress) {
    var p = clamp(progress, 0, 1);
    var lerpAmount = 0.35 + 0.65 * p;
    var center = (minBase + maxBase) * 0.5;
    var half = (maxBase - minBase) * 0.5 * lerpAmount;
    return [center - half, center + half];
  }

  function resolveLevelPreset(level) {
    var key = String(level || "moderate").toLowerCase();
    if (key === "low") {
      return {
        gravity: [0.98, 1.03],
        flap: [0.97, 1.03],
        speed: [0.95, 1.05],
        gap: [0.95, 1.05],
        spawn: [0.95, 1.05],
        margin: [0.95, 1.05]
      };
    }
    if (key === "high") {
      return {
        gravity: [0.92, 1.12],
        flap: [0.90, 1.10],
        speed: [0.88, 1.18],
        gap: [0.86, 1.14],
        spawn: [0.84, 1.14],
        margin: [0.86, 1.18]
      };
    }
    return {
      gravity: [drConfig.gravityScaleMin, drConfig.gravityScaleMax],
      flap: [drConfig.flapScaleMin, drConfig.flapScaleMax],
      speed: [drConfig.pipeSpeedScaleMin, drConfig.pipeSpeedScaleMax],
      gap: [drConfig.pipeGapScaleMin, drConfig.pipeGapScaleMax],
      spawn: [drConfig.spawnScaleMin, drConfig.spawnScaleMax],
      margin: [drConfig.gapMarginScaleMin, drConfig.gapMarginScaleMax]
    };
  }

  function sampleInRange(minV, maxV) {
    return minV + randUnit() * (maxV - minV);
  }

  function applyRandomization(options) {
    var cfg = options || {};
    var progress = clamp(Number(cfg.curriculumProgress || 0), 0, 1);
    var levelPreset = resolveLevelPreset(cfg.drLevel || cfg.level || "moderate");

    var gravityRange = mixRange(levelPreset.gravity[0], levelPreset.gravity[1], progress);
    var flapRange = mixRange(levelPreset.flap[0], levelPreset.flap[1], progress);
    var speedRange = mixRange(levelPreset.speed[0], levelPreset.speed[1], progress);
    var gapRange = mixRange(levelPreset.gap[0], levelPreset.gap[1], progress);
    var spawnRange = mixRange(levelPreset.spawn[0], levelPreset.spawn[1], progress);
    var marginRange = mixRange(levelPreset.margin[0], levelPreset.margin[1], progress);

    if (cfg.enabled === false || drConfig.enabled === false) {
      gravityRange = [1, 1];
      flapRange = [1, 1];
      speedRange = [1, 1];
      gapRange = [1, 1];
      spawnRange = [1, 1];
      marginRange = [1, 1];
    }

    currentGravity = BASE_GRAVITY * sampleInRange(gravityRange[0], gravityRange[1]);
    currentFlapVelocity = BASE_FLAP_VELOCITY * sampleInRange(flapRange[0], flapRange[1]);
    currentPipeSpeed = BASE_PIPE_SPEED * sampleInRange(speedRange[0], speedRange[1]);
    currentPipeGap = BASE_PIPE_GAP * sampleInRange(gapRange[0], gapRange[1]);
    currentPipeSpawnInterval = BASE_PIPE_SPAWN_INTERVAL * sampleInRange(spawnRange[0], spawnRange[1]);
    currentGapMargin = BASE_GAP_MARGIN * sampleInRange(marginRange[0], marginRange[1]);
    currentPipeWidth = BASE_PIPE_WIDTH;
  }

  function randomGapY() {
    var minY = currentGapMargin;
    var maxY = H - currentGapMargin;
    if (maxY <= minY) {
      minY = 110;
      maxY = H - 110;
    }
    return minY + randUnit() * (maxY - minY);
  }

  function addPipe(x) {
    pipes.push({
      x: x,
      gapY: randomGapY(),
      passed: false
    });
  }

  function resetGame(config) {
    var cfg = config;
    if (typeof config !== "object" || config === null) {
      cfg = { seed: config };
    }
    setSeed(cfg.seed);
    applyRandomization(cfg);
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
    birdVy = currentFlapVelocity;
  }

  function getNextPipe() {
    for (var i = 0; i < pipes.length; i += 1) {
      if (pipes[i].x + currentPipeWidth >= BIRD_X - BIRD_R) {
        return pipes[i];
      }
    }
    return pipes.length > 0 ? pipes[0] : null;
  }

  function getState() {
    var next = getNextPipe();
    var nextPipeDx = next ? (next.x + currentPipeWidth - BIRD_X) : W;
    var gapDy = next ? (birdY - next.gapY) : 0;
    return {
      y: birdY,
      vy: birdVy,
      nextPipeDx: nextPipeDx,
      gapDy: gapDy,
      score: score,
      alive: alive,
      pipeSpeed: currentPipeSpeed,
      pipeGap: currentPipeGap,
      gravity: currentGravity
    };
  }

  function collidePipe(pipe) {
    var withinX = (BIRD_X + BIRD_R > pipe.x) && (BIRD_X - BIRD_R < pipe.x + currentPipeWidth);
    if (!withinX) {
      return false;
    }

    var gapTop = pipe.gapY - (currentPipeGap / 2);
    var gapBottom = pipe.gapY + (currentPipeGap / 2);
    var hitUpper = birdY - BIRD_R < gapTop;
    var hitLower = birdY + BIRD_R > gapBottom;
    return hitUpper || hitLower;
  }

  function update(dt) {
    if (!alive) {
      return;
    }

    birdVy += currentGravity * dt;
    birdY += birdVy * dt;

    if (birdY - BIRD_R <= 0 || birdY + BIRD_R >= H) {
      alive = false;
    }

    spawnTimer += dt;
    if (spawnTimer >= currentPipeSpawnInterval) {
      spawnTimer -= currentPipeSpawnInterval;
      addPipe(W + 50);
    }

    for (var i = 0; i < pipes.length; i += 1) {
      var pipe = pipes[i];
      pipe.x -= currentPipeSpeed * dt;
      if (!pipe.passed && pipe.x + currentPipeWidth < BIRD_X) {
        pipe.passed = true;
        score += 1;
      }
      if (collidePipe(pipe)) {
        alive = false;
      }
    }

    pipes = pipes.filter(function (pipe) {
      return pipe.x + currentPipeWidth > -40;
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
      var gapTop = pipe.gapY - currentPipeGap / 2;
      var gapBottom = pipe.gapY + currentPipeGap / 2;

      ctx.fillRect(pipe.x, 0, currentPipeWidth, gapTop);
      ctx.fillRect(pipe.x, gapBottom, currentPipeWidth, H - gapBottom);
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
    reset: function (config) {
      resetGame(config);
    },
    flap: function () {
      onUserFlap();
    },
    getState: function () {
      return getState();
    },
    isDone: function () {
      return !alive;
    },
    configureRandomization: function (config) {
      if (!config || typeof config !== "object") {
        return;
      }
      var entries = Object.keys(config);
      for (var i = 0; i < entries.length; i += 1) {
        var key = entries[i];
        if (Object.prototype.hasOwnProperty.call(drConfig, key)) {
          if (key === "enabled") {
            drConfig[key] = Boolean(config[key]);
          } else {
            drConfig[key] = Number(config[key]);
          }
        }
      }
    },
    getRandomization: function () {
      return {
        gravity: currentGravity,
        flapVelocity: currentFlapVelocity,
        pipeSpeed: currentPipeSpeed,
        pipeGap: currentPipeGap,
        pipeSpawnInterval: currentPipeSpawnInterval,
        gapMargin: currentGapMargin,
        drLevel: drConfig.enabled ? 1 : 0
      };
    }
  };

  resetGame();
  requestAnimationFrame(frame);
}());

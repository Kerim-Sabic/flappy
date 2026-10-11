"use strict";

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

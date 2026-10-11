# Webgame Package Setup (Windows)

This project uses a reproducible, minimal Canvas Flappy game generated locally in `./webgame` so training does not depend on third-party web repo compatibility.

## 1) Generate or patch the webgame

From the project root:

```powershell
node .\patch_webgame_for_rl.js --target .\webgame
```

This writes/updates:
- `webgame/index.html`
- `webgame/style.css`
- `webgame/game.js`
- `webgame/server.js`
- `webgame/package.json`

The RL instrumentation is in `webgame/game.js` and exposes:

```js
window.__rl = {
  reset({ seed, drLevel, curriculumProgress }),
  flap(),
  getState(), // { y, vy, nextPipeDx, gapDy, score, alive }
  isDone(),
  configureRandomization(config),
  getRandomization()
}
```

## 2) Install and run the webgame

```powershell
cd .\webgame
npm install
npm run start -- --port 5173
```

Game URL:

`http://127.0.0.1:5173`

## 3) Optional quick API check in browser devtools

```js
window.__rl.getState()
window.__rl.flap()
window.__rl.reset({ seed: 123, drLevel: "moderate", curriculumProgress: 0.4 })
window.__rl.getRandomization()
```

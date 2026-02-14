const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");

let mainWindow;
let pythonProcess = null;
const SERVER_PORT = 8000;
const SERVER_URL = `http://127.0.0.1:${SERVER_PORT}`;

// ---------------------------------------------------------------------------
// Python server management
// ---------------------------------------------------------------------------

function findPython() {
  // Use absolute paths based on the project root (one level up from gui/)
  const projectRoot = path.join(__dirname, "..");
  const candidates =
    process.platform === "win32"
      ? [
          path.join(projectRoot, ".venv", "Scripts", "python.exe"),
          "python",
          "python3",
        ]
      : [
          path.join(projectRoot, ".venv", "bin", "python"),
          "python3",
          "python",
        ];
  return candidates;
}

function startPythonServer() {
  const projectRoot = path.join(__dirname, "..");
  const serverScript = path.join(projectRoot, "server.py");
  const pythonCandidates = findPython();

  function tryStart(index) {
    if (index >= pythonCandidates.length) {
      console.error("Could not find a working Python executable.");
      if (mainWindow) {
        mainWindow.webContents.send(
          "server-status",
          "error",
          "Python not found. Please ensure Python is installed and a virtual environment exists."
        );
      }
      return;
    }

    const pythonExe = pythonCandidates[index];
    console.log(`Trying Python: ${pythonExe}`);

    pythonProcess = spawn(pythonExe, [serverScript], {
      cwd: projectRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let started = false;

    pythonProcess.stdout.on("data", (data) => {
      const msg = data.toString();
      console.log(`[server] ${msg}`);
      if (msg.includes("Uvicorn running") || msg.includes("Application startup complete")) {
        started = true;
        waitForServer();
      }
    });

    pythonProcess.stderr.on("data", (data) => {
      const msg = data.toString();
      console.error(`[server] ${msg}`);
      // Uvicorn logs to stderr by default
      if (msg.includes("Uvicorn running") || msg.includes("Application startup complete")) {
        started = true;
        waitForServer();
      }
    });

    pythonProcess.on("error", () => {
      console.log(`Python candidate '${pythonExe}' failed, trying next...`);
      tryStart(index + 1);
    });

    pythonProcess.on("exit", (code) => {
      console.log(`Python server exited with code ${code}`);
      if (!started) {
        tryStart(index + 1);
      }
    });
  }

  tryStart(0);
}

function waitForServer(retries = 60, delay = 2000) {
  const check = (remaining) => {
    if (remaining <= 0) {
      if (mainWindow) {
        mainWindow.webContents.send(
          "server-status",
          "error",
          "Server failed to start within timeout."
        );
      }
      return;
    }

    http
      .get(`${SERVER_URL}/health`, (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          try {
            const data = JSON.parse(body);
            if (data.status === "ok") {
              console.log("Server is ready.");
              if (mainWindow) {
                mainWindow.webContents.send("server-status", "ready", data);
              }
            } else {
              // Still loading models
              setTimeout(() => check(remaining - 1), delay);
            }
          } catch {
            setTimeout(() => check(remaining - 1), delay);
          }
        });
      })
      .on("error", () => {
        setTimeout(() => check(remaining - 1), delay);
      });
  };

  if (mainWindow) {
    mainWindow.webContents.send(
      "server-status",
      "loading",
      "Starting inference server and loading models..."
    );
  }
  check(retries);
}

function stopPythonServer() {
  if (pythonProcess) {
    console.log("Stopping Python server...");
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// ---------------------------------------------------------------------------
// Electron window
// ---------------------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 850,
    minWidth: 900,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    title: "Ovarian Cancer Multimodal Classifier",
    icon: undefined, // Add icon path here if desired
  });

  mainWindow.loadFile(path.join(__dirname, "index.html"));

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Start the Python server after the window is ready
  mainWindow.webContents.on("did-finish-load", () => {
    startPythonServer();
  });
}

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------

ipcMain.handle("select-image", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Ultrasound Image",
    filters: [
      { name: "All Supported", extensions: ["dcm", "png", "jpg", "jpeg", "bmp", "tif", "tiff"] },
      { name: "DICOM", extensions: ["dcm"] },
      { name: "Images", extensions: ["png", "jpg", "jpeg", "bmp", "tif", "tiff"] },
    ],
    properties: ["openFile"],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

ipcMain.handle("get-server-url", () => {
  return SERVER_URL;
});

ipcMain.handle("read-file", async (_event, filePath) => {
  const fs = require("fs");
  try {
    const buffer = fs.readFileSync(filePath);
    return { data: buffer.toString("base64"), size: buffer.length };
  } catch (err) {
    console.error(`Failed to read file: ${filePath}`, err);
    return null;
  }
});

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  stopPythonServer();
  app.quit();
});

app.on("before-quit", () => {
  stopPythonServer();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

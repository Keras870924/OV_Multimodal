const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  selectImage: () => ipcRenderer.invoke("select-image"),
  getServerUrl: () => ipcRenderer.invoke("get-server-url"),
  readFile: (filePath) => ipcRenderer.invoke("read-file", filePath),
  onServerStatus: (callback) =>
    ipcRenderer.on("server-status", (_event, status, data) =>
      callback(status, data)
    ),
});

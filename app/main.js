const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
    const win = new BrowserWindow({
        width: 500,
        height: 586,
        maximizable: false,
        resizable: false,
        fullscreenable: false,
        webPreferences: {
            preload: path.join(__dirname, 'render.js'),
            contextIsolation: false,
            nodeIntegration: true
        }
    });
    win.loadFile('index.html');
}
app.whenReady().then(createWindow);
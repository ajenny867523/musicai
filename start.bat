@echo off
title AI Music Studio Local Server
echo 正在啟動 AI Music Studio 本地伺服器...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_server.ps1"
pause

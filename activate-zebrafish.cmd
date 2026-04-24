@echo off
REM 打开新的 PowerShell 窗口，并激活 zebrafish，停留在 doing 目录
powershell -NoExit -ExecutionPolicy Bypass -Command ^
  "& 'D:\anaconda3\shell\condabin\conda-hook.ps1'; conda activate zebrafish; Set-Location 'E:\graduation\doing'"

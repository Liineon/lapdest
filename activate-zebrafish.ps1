# 激活 conda 环境 zebrafish（Anaconda 路径：D:\anaconda3）
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# 重要：必须在「当前」PowerShell 里用点源方式执行，激活才会留在本窗口：
#   cd E:\graduation\doing
#   . .\activate-zebrafish.ps1
#
# 若写成 .\activate-zebrafish.ps1（无前面的点），脚本在子作用域结束后，环境不会保留。

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "D:\anaconda3\shell\condabin\conda-hook.ps1"
conda activate zebrafish

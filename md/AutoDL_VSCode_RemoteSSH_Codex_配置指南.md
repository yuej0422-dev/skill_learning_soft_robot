# AutoDL 中通过 VS Code Remote-SSH 使用 Codex：完整配置记录

> 适用环境  
> - 本地系统：Ubuntu  
> - 本地 VS Code：通过 Remote-SSH 连接 AutoDL  
> - 本地代理：Clash / Mihomo 等，HTTP 或 Mixed 端口为 `127.0.0.1:7890`  
> - AutoDL SSH 地址：`connect.cqa1.seetacloud.com`  
> - AutoDL SSH 端口：`23699`  
> - AutoDL 用户：`root`  
> - Codex：安装在 VS Code 的 `SSH: AutoDL` 远程环境中  
>
> 当前验证成功的网络模式：**规则代理模式**。  
> TUN 模式下 SSH 容易被代理接管，表现为连接慢、断连、Resolver Error；切回规则代理后，SSH 和 Codex 都明显稳定。

---

## 1. 最终网络结构

当前正常工作的网络链路如下：

```text
本地 VS Code 界面
    ↓ Remote-SSH
AutoDL 上的 VS Code Extension Host
    ↓
AutoDL 上的 Codex 扩展
    ↓ http://127.0.0.1:17890
SSH RemoteForward 反向隧道
    ↓
本地电脑 http://127.0.0.1:7890
    ↓
本地规则代理 / 代理节点
    ↓
OpenAI Codex 服务
```

返回数据沿相反方向返回：

```text
OpenAI
  ↓
本地代理节点
  ↓
本地 7890
  ↓
SSH RemoteForward
  ↓
AutoDL 17890
  ↓
远程 Codex 扩展
  ↓
本地 VS Code 界面
```

功能分工：

```text
代码读取、修改、终端命令、GPU 调试：AutoDL
Codex 网络出口：本地代理
Codex 登录凭证：AutoDL 的 /root/.codex/auth.json
编辑器界面：本地 VS Code
```

---

## 2. 前置检查

### 2.1 确认本地代理端口

在本地 Ubuntu 执行：

```bash
ss -lntp | grep ':7890'
```

预期：

```text
LISTEN ... 127.0.0.1:7890 ...
```

测试本地代理：

```bash
curl -x http://127.0.0.1:7890 \
  -I https://api.openai.com/v1/models \
  --connect-timeout 10
```

未携带 API Key 时，快速返回 `401 Unauthorized` 是正常结果，说明代理链路可用。

也可以测试：

```bash
curl -x http://127.0.0.1:7890 \
  -I https://chatgpt.com \
  --connect-timeout 10
```

`chatgpt.com` 可能因为 Cloudflare 浏览器挑战返回 `403`，这不一定说明代理失效。

---

## 3. 本地 SSH 配置

编辑本地 SSH 配置：

```bash
nano ~/.ssh/config
```

推荐使用独立别名，不要直接把域名作为 Host 名：

```sshconfig
Host autodl
    HostName connect.cqa1.seetacloud.com
    User root
    Port 23699

    RemoteForward 17890 127.0.0.1:7890
    ExitOnForwardFailure yes

    ServerAliveInterval 15
    ServerAliveCountMax 8
    TCPKeepAlive yes
```

含义：

```text
AutoDL 的 127.0.0.1:17890
        ↓ SSH 反向转发
本地电脑的 127.0.0.1:7890
```

保存后测试：

```bash
ssh autodl
```

也可以查看实际生效配置：

```bash
ssh -G autodl | grep -Ei \
'^(hostname|user|port|remoteforward|serveralive|tcpkeepalive)'
```

应该包含：

```text
hostname connect.cqa1.seetacloud.com
user root
port 23699
remoteforward 17890 127.0.0.1:7890
```

---

## 4. 为什么最终使用规则代理，而不是 TUN

在 TUN 模式下，SSH 日志曾显示：

```text
Authenticated to connect.cqa1.seetacloud.com ([198.18.x.x]:23699)
```

`198.18.0.0/15` 常被 Clash/Mihomo 的 fake-IP 使用。这说明：

```text
connect.cqa1.seetacloud.com:23699
```

本身也被 TUN 接管，导致：

- SSH 连接变慢；
- VS Code Remote-SSH 偶发 Resolver Error；
- SSH 反向隧道不稳定；
- Codex 流式连接频繁 `Reconnecting`；
- TLS 偶发握手超时。

切换到**规则代理模式**后：

- SSH 直接连接 AutoDL；
- Codex 仍通过 `17890 → 本地 7890` 使用代理；
- SSH 和 Codex 均恢复稳定。

因此推荐：

```text
SSH → 直连 AutoDL
Codex → 通过 SSH RemoteForward 使用本地代理
```

不要让 SSH 本身再经过代理节点。

---

## 5. VS Code Remote-SSH 连接 AutoDL

在 VS Code 中：

```text
Ctrl + Shift + P
Remote-SSH: Connect to Host...
autodl
```

连接成功后，左下角应显示：

```text
SSH: autodl
```

然后打开服务器项目，例如：

```text
/root/autodl-tmp/smolvla_project
```

---

## 6. 在远程环境安装 Codex

打开 VS Code 扩展页面，搜索：

```text
Codex – OpenAI's coding agent
```

确认安装目标为：

```text
SSH: autodl
```

点击：

```text
Install in SSH: autodl
```

不要只安装在：

```text
LOCAL
```

安装后可在 AutoDL 终端检查：

```bash
ls ~/.vscode-server/extensions | grep -Ei 'openai|codex|chatgpt'
```

---

## 7. Codex 黑屏问题

安装后若 Codex 面板黑屏、没有登录界面：

1. 先执行：

```text
Ctrl + Shift + P
Developer: Reload Window
```

2. 如果仍黑屏，检查远程 Codex 扩展是否确实安装在：

```text
SSH: autodl
```

3. 再检查 Remote-SSH 和远程 VS Code Server 是否正常。

本次实际使用中，黑屏问题解决后，登录阶段又出现了 `403`，后续通过复制本地凭证解决。

---

## 8. 解决远程 Codex 登录 403

本地 Codex 已经正常登录，因此本地存在：

```text
/home/yuej/.codex/auth.json
```

本地检查：

```bash
ls -lah ~/.codex/auth.json
```

当前文件示例：

```text
-rw------- 1 yuej yuej 4.5K ... /home/yuej/.codex/auth.json
```

不要执行：

```bash
cat ~/.codex/auth.json
```

该文件包含登录凭证。

### 8.1 在 AutoDL 创建目录

因为已有 VS Code 连接可能占用 `17890`，通过额外 SSH/SCP 连接时使用：

```text
-o ClearAllForwardings=yes
```

本地执行：

```bash
ssh \
  -o ClearAllForwardings=yes \
  -p 23699 \
  root@connect.cqa1.seetacloud.com '
mkdir -p /root/.codex
chmod 700 /root/.codex
'
```

### 8.2 上传凭证

本地执行：

```bash
scp \
  -o ClearAllForwardings=yes \
  -P 23699 \
  /home/yuej/.codex/auth.json \
  root@connect.cqa1.seetacloud.com:/root/.codex/auth.json
```

注意：

```text
ssh 使用小写 -p
scp 使用大写 -P
```

### 8.3 设置权限

在 AutoDL 远程终端执行：

```bash
chmod 700 /root/.codex
chmod 600 /root/.codex/auth.json
ls -lah /root/.codex/auth.json
```

预期：

```text
-rw------- 1 root root ... /root/.codex/auth.json
```

重新加载 VS Code：

```text
Ctrl + Shift + P
Developer: Reload Window
```

Codex 应直接读取：

```text
/root/.codex/auth.json
```

无需再次在 AutoDL 上点击登录。

---

## 9. 配置 VS Code Remote Settings

这是让远程 Codex 插件稳定使用代理的关键步骤。

在已经连接 AutoDL 的 VS Code 窗口中：

```text
Ctrl + Shift + P
Preferences: Open Remote Settings (JSON)
```

必须打开 **Remote Settings**，不是本地 User Settings。

写入或合并：

```json
{
    "http.proxy": "http://127.0.0.1:17890",
    "http.proxySupport": "override",
    "terminal.integrated.env.linux": {
        "HTTP_PROXY": "http://127.0.0.1:17890",
        "HTTPS_PROXY": "http://127.0.0.1:17890",
        "http_proxy": "http://127.0.0.1:17890",
        "https_proxy": "http://127.0.0.1:17890",
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1"
    }
}
```

AutoDL 端必须写：

```text
127.0.0.1:17890
```

不能写：

```text
127.0.0.1:7890
```

因为 `7890` 是本地电脑的代理端口，而 AutoDL 能访问的是 SSH 反向映射后的 `17890`。

当前没有证书问题，因此不需要设置：

```json
"http.proxyStrictSSL": false
```

---

## 10. 给远程 Shell 配置代理环境变量

在 AutoDL 创建统一代理配置：

```bash
cat > /root/.codex_proxy_env <<'EOF'
export HTTP_PROXY="http://127.0.0.1:17890"
export HTTPS_PROXY="http://127.0.0.1:17890"
export http_proxy="http://127.0.0.1:17890"
export https_proxy="http://127.0.0.1:17890"

export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"
EOF

chmod 600 /root/.codex_proxy_env
```

让 Bash 登录环境加载它：

```bash
touch /root/.bash_profile
```

```bash
grep -qF '/root/.codex_proxy_env' /root/.bash_profile || \
echo '[ -f /root/.codex_proxy_env ] && . /root/.codex_proxy_env' \
  >> /root/.bash_profile
```

也可以加入 `.profile`：

```bash
grep -qF '/root/.codex_proxy_env' /root/.profile 2>/dev/null || \
echo '[ -f /root/.codex_proxy_env ] && . /root/.codex_proxy_env' \
  >> /root/.profile
```

验证：

```bash
bash -lc 'env | grep -iE "^(HTTP_PROXY|HTTPS_PROXY|http_proxy|https_proxy)="'
```

预期：

```text
https_proxy=http://127.0.0.1:17890
HTTPS_PROXY=http://127.0.0.1:17890
HTTP_PROXY=http://127.0.0.1:17890
http_proxy=http://127.0.0.1:17890
```

---

## 11. 验证 AutoDL 到本地代理的链路

在 AutoDL 执行：

```bash
curl \
  -x http://127.0.0.1:17890 \
  --connect-timeout 10 \
  --max-time 30 \
  -sS \
  -o /dev/null \
  -w 'HTTP=%{http_code} connect=%{time_connect}s total=%{time_total}s\n' \
  https://api.openai.com/v1/models
```

正常结果：

```text
HTTP=401 connect=... total=...
```

解释：

- `401`：网络正常，只是请求没有 API Key；
- `000`：连接或 TLS 未完成；
- `Connection refused`：`17890` 没有建立；
- `SSL connection timeout`：隧道或本地代理节点不稳定。

连续测试：

```bash
for i in $(seq 1 10); do
  printf '%02d  ' "$i"

  curl \
    -x http://127.0.0.1:17890 \
    --connect-timeout 10 \
    --max-time 30 \
    -sS \
    -o /dev/null \
    -w 'HTTP=%{http_code} connect=%{time_connect}s total=%{time_total}s\n' \
    https://api.openai.com/v1/models \
  || echo FAILED

  sleep 2
done
```

理想状态：

```text
10/10 均快速返回 401
```

---

## 12. 检查远程 Extension Host 是否继承代理

在 AutoDL 执行：

```bash
pid="$(pgrep -n -f 'extensionHost' || true)"

echo "Extension Host PID: $pid"

if [ -n "$pid" ]; then
  tr '\0' '\n' < "/proc/${pid}/environ" \
    | grep -iE \
      '^(HTTP_PROXY|HTTPS_PROXY|http_proxy|https_proxy)='
fi
```

如果没有输出，说明当前 Extension Host 是在代理配置写入之前启动的。

此时执行：

```text
Ctrl + Shift + P
Remote-SSH: Kill VS Code Server on Host...
```

然后重新连接：

```text
autodl
```

再次检查 Extension Host 环境变量。

需要注意：本次实际稳定使用的关键是 Remote Settings 的：

```json
"http.proxy": "http://127.0.0.1:17890"
```

即使 Extension Host 环境变量检查不理想，Remote Settings 也可能已经让 Codex 正常走代理。

---

## 13. SSH 连接不稳定的排查

### 13.1 查看 VS Code Remote-SSH 日志

在 VS Code：

```text
View → Output
```

选择：

```text
Remote - SSH
```

正常日志应包括：

```text
Authenticated to connect.cqa1.seetacloud.com
Found existing installation at /root/.vscode-server
Remote server is listening on port ...
Exec server created and cached
```

### 13.2 绕过 SSH 转发进行测试

本地执行：

```bash
ssh \
  -vvv \
  -o ClearAllForwardings=yes \
  -p 23699 \
  root@connect.cqa1.seetacloud.com
```

如果仍然很慢，说明问题不只是 `RemoteForward`。

本次日志中出现：

```text
Authenticated to connect.cqa1.seetacloud.com ([198.18.0.47]:23699)
```

说明 SSH 被 TUN/fake-IP 接管。

### 13.3 最终解决

将本地代理从：

```text
TUN 模式
```

切换为：

```text
规则代理模式
```

结果：

- SSH 连接明显稳定；
- VS Code Remote-SSH 不再频繁失败；
- Codex 流式响应恢复正常；
- `Reconnecting` 大幅减少或消失。

---

## 14. `remote port forwarding failed` 的原因

曾出现：

```text
Error: remote port forwarding failed for listen port 17890
```

原因通常是：

- 当前 VS Code Remote-SSH 连接已占用 AutoDL 的 `17890`；
- 新开的 SSH 或 SCP 连接再次读取同一个 `RemoteForward`；
- 新连接试图重复监听 `17890`；
- `ExitOnForwardFailure yes` 导致整个 SSH 命令退出。

临时 SSH/SCP 命令应加：

```text
-o ClearAllForwardings=yes
```

例如：

```bash
ssh \
  -o ClearAllForwardings=yes \
  -p 23699 \
  root@connect.cqa1.seetacloud.com
```

```bash
scp \
  -o ClearAllForwardings=yes \
  -P 23699 \
  local_file \
  root@connect.cqa1.seetacloud.com:/remote/path/
```

这样不会影响 VS Code 当前已经建立的 `17890` 隧道。

---

## 15. `ss: command not found`

AutoDL 社区镜像可能较精简，没有安装 `ss`。

安装：

```bash
apt-get update
apt-get install -y iproute2
```

然后检查：

```bash
ss -lntp | grep ':17890'
```

安装 `iproute2` 不会影响 PyTorch、CUDA 或 Python 环境。

不过，最直接的代理验证仍然是：

```bash
curl -x http://127.0.0.1:17890 \
  -I https://api.openai.com/v1/models
```

---

## 16. 日常使用方式

### 16.1 启动前

1. 本地启动代理软件；
2. 使用规则代理模式；
3. 确认本地 `7890` 正常：

```bash
ss -lntp | grep ':7890'
```

4. VS Code 通过：

```text
Remote-SSH: Connect to Host...
autodl
```

连接 AutoDL。

### 16.2 连接后验证

AutoDL 终端：

```bash
curl \
  -x http://127.0.0.1:17890 \
  --connect-timeout 10 \
  --max-time 30 \
  -sS \
  -o /dev/null \
  -w 'HTTP=%{http_code} total=%{time_total}s\n' \
  https://api.openai.com/v1/models
```

返回 `401` 后，再打开 Codex。

### 16.3 Codex 能做什么

因为 Codex 安装在远程环境中，它可以直接：

```text
读取 /root/autodl-tmp/smolvla_project
修改服务器代码
读取训练日志
执行 Python
运行 GPU smoke test
修改 YAML
查看 Git diff
排查 CUDA 和 PyTorch 环境
```

执行命令的位置是 AutoDL，不是本地电脑。

### 16.4 断开连接的影响

本地电脑关机、睡眠、关闭代理或断开 SSH 后：

- Codex 会断网；
- VS Code Remote-SSH 会断开；
- `17890` 隧道消失。

但使用 `tmux` 启动的训练不会因此停止。

训练建议：

```bash
tmux new -s smolvla
```

断开：

```text
Ctrl+B
D
```

恢复：

```bash
tmux attach -t smolvla
```

---

## 17. 安全注意事项

`auth.json` 等同于账号登录凭证。

禁止：

```text
提交到 Git
放入项目压缩包
上传 Hugging Face
上传 GitHub
通过聊天发送内容
与他人共享
```

检查 Git：

```bash
git status --short
```

建议在 `.gitignore` 中加入：

```gitignore
.codex/
auth.json
*.env
.env
```

远程权限保持：

```bash
chmod 700 /root/.codex
chmod 600 /root/.codex/auth.json
```

---

## 18. 最终成功配置摘要

### 本地代理

```text
模式：规则代理
HTTP/Mixed 端口：127.0.0.1:7890
```

### 本地 SSH

```sshconfig
Host autodl
    HostName connect.cqa1.seetacloud.com
    User root
    Port 23699
    RemoteForward 17890 127.0.0.1:7890
    ExitOnForwardFailure yes
    ServerAliveInterval 15
    ServerAliveCountMax 8
    TCPKeepAlive yes
```

### AutoDL Codex 凭证

```text
/root/.codex/auth.json
```

### VS Code Remote Settings

```json
{
    "http.proxy": "http://127.0.0.1:17890",
    "http.proxySupport": "override",
    "terminal.integrated.env.linux": {
        "HTTP_PROXY": "http://127.0.0.1:17890",
        "HTTPS_PROXY": "http://127.0.0.1:17890",
        "http_proxy": "http://127.0.0.1:17890",
        "https_proxy": "http://127.0.0.1:17890"
    }
}
```

### AutoDL Shell 代理

```bash
export HTTP_PROXY="http://127.0.0.1:17890"
export HTTPS_PROXY="http://127.0.0.1:17890"
export http_proxy="http://127.0.0.1:17890"
export https_proxy="http://127.0.0.1:17890"
```

### 最终工作状态

```text
VS Code Remote-SSH：稳定
Codex 登录：成功
Codex 读取远程项目：正常
Codex 执行服务器命令：正常
Codex 流式交互：正常
SSH：在规则代理模式下明显稳定
```

---

## 19. 最短恢复流程

以后 Codex 突然不可用时，按以下顺序检查：

```text
1. 本地代理是否启动，7890 是否监听
2. 当前是否误切回 TUN 模式
3. VS Code 是否通过 autodl Host 连接
4. AutoDL 的 17890 是否能返回 API 401
5. Remote Settings 中 http.proxy 是否仍为 127.0.0.1:17890
6. /root/.codex/auth.json 是否存在
7. Developer: Reload Window
8. 必要时 Kill VS Code Server 后重新连接
```

快速诊断命令：

本地：

```bash
ss -lntp | grep ':7890'
```

AutoDL：

```bash
curl \
  -x http://127.0.0.1:17890 \
  --connect-timeout 10 \
  --max-time 30 \
  -sS \
  -o /dev/null \
  -w 'HTTP=%{http_code} total=%{time_total}s\n' \
  https://api.openai.com/v1/models
```

只要本地 `7890` 正常，AutoDL 测试稳定返回 `401`，远程 Codex 通常就可以正常工作。

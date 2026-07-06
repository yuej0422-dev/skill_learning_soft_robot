# AutoDL 中通过 VS Code Remote-SSH 使用 Codex：关键配置

## 1. 本地代理

确认本地代理监听 `7890`：

```bash
ss -lntp | grep ':7890'
```

测试：

```bash
curl -x http://127.0.0.1:7890   -I https://api.openai.com/v1/models   --connect-timeout 10
```

快速返回 `401` 即说明代理正常。

> 实际使用中，**规则代理模式比 TUN 模式稳定**。  
> TUN 模式可能接管 SSH，导致 VS Code Remote-SSH 连接慢、断线或 Resolver Error。

---

## 2. 配置 SSH 反向代理

编辑本地：

```bash
nano ~/.ssh/config
```

加入：

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
AutoDL 127.0.0.1:17890
        ↓ SSH 反向转发
本地 127.0.0.1:7890
```

VS Code 通过：

```text
Remote-SSH: Connect to Host...
autodl
```

连接服务器。

---

## 3. 在远程环境安装 Codex

在 VS Code 扩展页面搜索 Codex，确认安装到：

```text
SSH: autodl
```

而不是只安装到 `LOCAL`。

---

## 4. 解决远程登录 403

本地已经登录 Codex 时，凭证一般位于：

```text
~/.codex/auth.json
```

将其复制到 AutoDL：

```bash
ssh -o ClearAllForwardings=yes   -p 23699   root@connect.cqa1.seetacloud.com   'mkdir -p /root/.codex && chmod 700 /root/.codex'
```

```bash
scp -o ClearAllForwardings=yes   -P 23699   ~/.codex/auth.json   root@connect.cqa1.seetacloud.com:/root/.codex/auth.json
```

远程设置权限：

```bash
chmod 600 /root/.codex/auth.json
```

然后执行：

```text
Developer: Reload Window
```

> `auth.json` 等同于登录凭证，不要提交 Git 或上传到项目。

---

## 5. 配置 VS Code Remote Settings

连接 AutoDL 后打开：

```text
Preferences: Open Remote Settings (JSON)
```

加入：

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

注意：

```text
AutoDL 端写 17890
本地代理端口是 7890
```

---

## 6. 验证代理链路

在 AutoDL 终端执行：

```bash
curl   -x http://127.0.0.1:17890   --connect-timeout 10   --max-time 30   -sS   -o /dev/null   -w 'HTTP=%{http_code} total=%{time_total}s
'   https://api.openai.com/v1/models
```

正常结果：

```text
HTTP=401
```

如果出现：

```text
HTTP=000
SSL connection timeout
Connection refused
```

检查：

1. 本地代理是否启动；
2. 是否误切回 TUN 模式；
3. SSH 是否重新连接并建立了 `RemoteForward`；
4. Remote Settings 是否仍为 `127.0.0.1:17890`。

---

## 7. 最终稳定配置

```text
本地代理模式：规则代理
本地代理端口：127.0.0.1:7890
AutoDL 代理入口：127.0.0.1:17890
SSH：直接连接 AutoDL
Codex：通过 SSH RemoteForward 使用本地代理
Codex 凭证：/root/.codex/auth.json
```

最终网络链路：

```text
本地 VS Code
    ↓ Remote-SSH
AutoDL 上的 Codex
    ↓ 127.0.0.1:17890
SSH RemoteForward
    ↓ 本地 127.0.0.1:7890
本地规则代理
    ↓
OpenAI
```

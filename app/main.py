# -*- coding: utf-8 -*-
"""LoRA 训练器入口：python app/main.py [--port 28100] [--host 127.0.0.1] [--no-browser]"""
import argparse
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import paths  # noqa: E402
import scanner  # noqa: E402
import server  # noqa: E402
import settings  # noqa: E402
import tasks  # noqa: E402


def _port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _is_us(host, port):
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/index.html", timeout=2) as r:
            return b"LoRA" in r.read(4000)
    except Exception:
        return False


def _open_browser(url):
    """没有桌面的机器（云主机 / 容器）上别去调浏览器。"""
    if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        return webbrowser.open(url)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--host", help="监听地址；云主机上用 0.0.0.0（或设环境变量 LORA_TRAINER_HOST）")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--wait-port", action="store_true", help="界面里点「重启训练器」时用：等旧的训练器让出端口再启动")
    args = ap.parse_args()
    s = settings.load()
    host = args.host or os.environ.get("LORA_TRAINER_HOST") or s.get("host") or "127.0.0.1"
    port = args.port or int(s.get("port") or 28100)
    url = f"http://127.0.0.1:{port}/"
    shown = f"http://{host}:{port}/" if host not in ("0.0.0.0", "::") else f"http://<本机IP>:{port}/"
    if args.wait_port:
        for _ in range(60):
            if _port_free("127.0.0.1", port):
                break
            time.sleep(0.5)

    if not _port_free("127.0.0.1", port):
        if _is_us("127.0.0.1", port):
            print(f"训练器已经在运行：{url}")
            if not args.no_browser and s.get("open_browser", True):
                _open_browser(url)
            return
        for p in range(port + 1, port + 50):
            if _port_free("127.0.0.1", p):
                print(f"端口 {port} 被其他程序占用，改用 {p}（可在「训练器设置」里修改端口）")
                port, url = p, f"http://127.0.0.1:{p}/"
                shown = f"http://{host}:{p}/" if host not in ("0.0.0.0", "::") else f"http://<本机IP>:{p}/"
                break

    try:   # 整合包搬过位置：先把训练环境里的旧路径改过来，再做别的
        import envsetup
        if envsetup.relocate_env():
            print(f"训练器换了位置，训练环境的路径已改到：{paths.ROOT}", flush=True)
    except Exception as e:
        print(f"训练环境路径修正失败：{e}", flush=True)
    tasks.load()
    if not os.path.isfile(scanner.INSTALLS_FILE):
        scanner.start_scan_async()
    if s.get("network") == "auto":
        threading.Thread(target=settings.detect_network, daemon=True).start()

    def _sync():   # 训练器更新过（任何方式）而引擎 / 依赖还是旧的：自动同步
        try:
            import updater
            if updater.auto_sync():
                print("训练引擎 / 环境和这个版本的训练器对不上，正在后台自动同步（进度在左下角）", flush=True)
        except Exception as e:
            print(f"检查训练环境版本失败：{e}", flush=True)
    threading.Thread(target=_sync, daemon=True).start()

    server.serve(host, port)
    for p in (os.environ.get("LORA_TRAINER_EXTRA_PORTS") or "").replace(",", " ").split():
        try:
            if int(p) != port:
                server.serve(host, int(p), extra=True)
                print(f"同时监听端口 {p}", flush=True)
        except (ValueError, OSError) as e:
            print(f"端口 {p} 监听失败：{e}", flush=True)

    def _bye(_sig=None, _frame=None):
        tasks.stop_current()
        os._exit(0)

    for _sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
        if _sig:
            try:
                signal.signal(_sig, _bye)
            except (ValueError, OSError):
                pass

    print("=" * 56)
    print(f"  LoRA 训练器已启动：{shown}")
    if host in ("0.0.0.0", "::"):
        print(f"  本机访问：{url}")
    if os.name == "nt":
        print("  关闭这个窗口即退出训练器（正在训练的任务会被中断）")
    else:
        print("  Ctrl+C 或 ./stop.sh 退出训练器（正在训练的任务会被中断）")
    print("=" * 56, flush=True)
    if not args.no_browser and s.get("open_browser", True):
        _open_browser(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        _bye()


if __name__ == "__main__":
    main()

"""Authoritative test: does headroom.compress() make ANY network call?

We monkeypatch socket.socket.connect to record every outbound
connection attempt, then run a real compression. The relevance model
is already cached on disk, so a clean run should record ZERO network
connections (filesystem model load is not a socket op).
"""
import socket

_attempts = []
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _spy_connect(self, address):
    _attempts.append(("connect", address))
    host = address[0] if isinstance(address, tuple) else str(address)
    if host not in ("127.0.0.1", "::1", "localhost"):
        import traceback
        with open("/tmp/net_stack.txt", "a") as f:
            f.write(f"\n=== REMOTE connect to {address} ===\n")
            f.write("".join(traceback.format_stack()))
    return _orig_connect(self, address)


def _spy_connect_ex(self, address):
    _attempts.append(("connect_ex", address))
    return _orig_connect_ex(self, address)


socket.socket.connect = _spy_connect
socket.socket.connect_ex = _spy_connect_ex

# Also force HF offline so a cache-miss can't sneak a download.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import headroom

ctx = "\n\n".join(
    f"[{i+1}] (doc-{i}.pdf p.{i}) The Ohio Geological Survey was founded in "
    "1837 under William Williams Mather. Limestone and dolomite are the "
    "leading mineral commodities by value. " * 2
    for i in range(8)
)
messages = [
    {"role": "system", "content": "Answer only from context. Cite [n]."},
    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: When was the "
                                "Ohio Geological Survey founded?"},
]

res = headroom.compress(messages, model="gpt-4o", model_limit=32768,
                        compress_user_messages=True)
print(f"compression ok: {res.tokens_before} -> {res.tokens_after} "
      f"({res.compression_ratio*100:.0f}%), transforms={res.transforms_applied}")

# Filter to genuinely remote endpoints (ignore loopback / unix sockets).
remote = []
for kind, addr in _attempts:
    host = addr[0] if isinstance(addr, tuple) else str(addr)
    if host in ("127.0.0.1", "::1", "localhost"):
        continue
    remote.append((kind, addr))

print(f"\ntotal socket.connect attempts: {len(_attempts)}")
print(f"REMOTE (non-loopback) connection attempts: {len(remote)}")
for kind, addr in remote:
    print(f"  {kind} -> {addr}")
if not remote:
    print("RESULT: compress() made NO remote network connections. Local-only. OK")
else:
    print("RESULT: compress() attempted remote connections (see above).")

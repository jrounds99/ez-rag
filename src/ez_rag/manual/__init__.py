"""In-tool offline manual. Renders bundled markdown via Rich."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown


_HERE = Path(__file__).parent


def list_topics() -> list[str]:
    return sorted(p.stem for p in _HERE.glob("*.md"))


def show(topic: str | None, console: Console) -> None:
    topics = list_topics()
    if topic is None or topic in ("topics", "list"):
        console.print("[bold]Manual topics[/]")
        for t in topics:
            console.print(f"  [cyan]ez-rag help {t}[/]")
        return
    page = _HERE / f"{topic}.md"
    if not page.exists():
        console.print(f"[red]No manual page for[/] '{topic}'.\nAvailable: {', '.join(topics)}")
        return
    console.print(Markdown(page.read_text(encoding="utf-8")))

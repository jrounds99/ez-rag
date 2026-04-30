"""ez-rag command-line interface."""
from __future__ import annotations

import json as _json
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

from .config import Config
from .embed import make_embedder
from .generate import answer as gen_answer, detect_backend
from .index import Index
from .ingest import ingest
from .retrieve import hybrid_search, smart_retrieve
from .workspace import Workspace, find_workspace, require_workspace


app = typer.Typer(
    no_args_is_help=True,
    help="ez-rag — drop documents in a folder, chat with them.",
    rich_markup_mode="rich",
)
console = Console()


# ----- init / status ---------------------------------------------------------

@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Where to create the workspace"),
):
    """Create a workspace at PATH."""
    path = path.resolve()
    ws = Workspace(path)
    if ws.is_initialized():
        console.print(f"Workspace already initialized at [bold]{path}[/]")
        return
    ws.initialize()
    console.print(Panel.fit(
        f"[green]ez-rag workspace created.[/]\n\n"
        f"  drop docs in:  [bold]{ws.docs_dir}[/]\n"
        f"  config:        {ws.config_path}\n\n"
        f"Next:  [cyan]ez-rag ingest[/]",
        title="ez-rag init",
        border_style="green",
    ))


@app.command()
def status():
    """Show workspace state."""
    from .index import read_stats
    ws = require_workspace()
    cfg = ws.load_config()
    s = read_stats(ws.meta_db_path)
    backend = detect_backend(cfg)
    table = Table(show_header=False, box=None)
    table.add_row("workspace", str(ws.root))
    table.add_row("docs/", str(ws.docs_dir))
    if s is None:
        table.add_row("files indexed", "[dim]nothing ingested yet — `ez-rag ingest`[/]")
    else:
        table.add_row("files indexed", str(s["files"]))
        table.add_row("chunks", str(s["chunks"]))
        table.add_row("doc bytes", f"{s['doc_bytes']:,}")
        if s.get("last_embedder"):
            table.add_row("last embedder", s["last_embedder"])
    table.add_row("LLM backend", backend)
    table.add_row("LLM model", cfg.llm_model)
    table.add_row(
        "embedder",
        cfg.embedder_model + (" (ollama)" if backend == "ollama" else " (fastembed)"),
    )
    console.print(Panel(table, title="ez-rag status", border_style="cyan"))


# ----- ingest ----------------------------------------------------------------

@app.command("ingest")
def ingest_cmd(
    force: bool = typer.Option(False, "--force", help="Reingest even if file unchanged"),
    watch: bool = typer.Option(False, "--watch", help="Watch docs/ and reingest on change"),
    flavor_seconds: int = typer.Option(
        15, "--flavor-seconds",
        help="Print a sample of the chunk being processed every N seconds (0 = off)",
    ),
):
    """Parse, chunk, embed everything in ./docs/."""
    import time as _time

    ws = require_workspace()
    cfg = ws.load_config()

    def run_once():
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
            TextColumn, TimeElapsedColumn, TimeRemainingColumn,
        )

        def fmt_bytes(n: int) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if n < 1024 or unit == "GB":
                    return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
                n /= 1024
            return f"{n:.1f} GB"

        last_flavor_t = [0.0]
        last_path = [""]

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description:<48}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[bytes_str]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as bar:
            task = bar.add_task(
                "ingesting", total=1, completed=0,
                bytes_str="0 B / 0 B",
            )

            def cb(prog):
                # Update bar
                if prog.files_total > 0:
                    bar.update(
                        task, total=prog.files_total,
                        completed=prog.files_done,
                        description=(
                            f"{Path(prog.current_path).name[:40] if prog.current_path else '…':<40} "
                            f"{prog.status[:8]:<8}"
                        ),
                        bytes_str=(
                            f"{fmt_bytes(prog.bytes_done)} / "
                            f"{fmt_bytes(prog.bytes_total)}  "
                            f"db {fmt_bytes(prog.db_bytes)}"
                        ),
                    )
                # Periodic flavor blurb
                now = _time.monotonic()
                if (flavor_seconds > 0 and prog.snippet and
                        (now - last_flavor_t[0]) >= flavor_seconds and
                        prog.current_path != last_path[0]):
                    page = f" p.{prog.page}" if prog.page else ""
                    bar.console.print(
                        f"\n[dim]── {prog.current_path}{page} ──[/]"
                    )
                    snip = prog.snippet[:200].replace("\n", " ").strip()
                    bar.console.print(f"[italic dim]   {snip}…[/]\n")
                    last_flavor_t[0] = now
                    last_path[0] = prog.current_path

            stats = ingest(ws, cfg=cfg, force=force, progress=cb)
        _print_ingest_stats(stats)

    run_once()
    if not watch:
        return

    last_mtimes: dict[str, float] = {}
    console.print("[dim]watching docs/ — Ctrl-C to stop[/]")
    try:
        while True:
            changed = False
            for p in ws.docs_dir.rglob("*"):
                if p.is_file():
                    m = p.stat().st_mtime
                    if last_mtimes.get(str(p)) != m:
                        last_mtimes[str(p)] = m
                        changed = True
            if changed:
                run_once()
            _time.sleep(2.0)
    except KeyboardInterrupt:
        console.print("\nstopped.")


def _print_ingest_stats(stats):
    t = Table(show_header=False, box=None)
    t.add_row("files seen", str(stats.files_seen))
    t.add_row("new",        str(stats.files_new))
    t.add_row("changed",    str(stats.files_changed))
    t.add_row("unchanged",  str(stats.files_skipped_unchanged))
    t.add_row("removed",    str(stats.files_removed))
    t.add_row("unsupported",str(stats.files_unsupported))
    t.add_row("errored",    str(stats.files_errored))
    t.add_row("chunks added", str(stats.chunks_added))
    t.add_row("seconds",    f"{stats.seconds:.2f}")
    console.print(Panel(t, title="ingest done", border_style="green"))
    if stats.errors:
        for path, err in stats.errors[:10]:
            console.print(f"[red]ERROR[/] {path}: {err}")
        if len(stats.errors) > 10:
            console.print(f"[red]…and {len(stats.errors) - 10} more[/]")


# ----- ask / chat ------------------------------------------------------------

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the corpus"),
    top_k: int = typer.Option(8, "--top-k", "-k"),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    citations: bool = typer.Option(True, "--citations/--no-citations"),
    no_hybrid: bool = typer.Option(False, "--no-hybrid"),
    no_rag: bool = typer.Option(
        False, "--no-rag",
        help="Skip retrieval entirely; ask the LLM directly.",
    ),
):
    """One-shot Q&A."""
    ws = require_workspace()
    cfg = ws.load_config()
    use_rag = cfg.use_rag and not no_rag
    if use_rag:
        embedder = make_embedder(cfg)
        idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
        # smart_retrieve honors cfg.hybrid / rerank / use_hyde / multi_query.
        # The --top-k flag overrides cfg.top_k for this single call.
        saved_k, saved_hybrid = cfg.top_k, cfg.hybrid
        cfg.top_k = top_k
        if no_hybrid:
            cfg.hybrid = False
        try:
            hits = smart_retrieve(query=question, embedder=embedder,
                                  index=idx, cfg=cfg)
        finally:
            cfg.top_k, cfg.hybrid = saved_k, saved_hybrid
    else:
        hits = []
    if json:
        out = {
            "question": question,
            "retrieved": [
                {"doc_id": h.path, "source": h.path, "page": h.page,
                 "section": h.section, "score": h.score, "text": h.text}
                for h in hits
            ],
        }
        if hits:
            try:
                ans = gen_answer(question=question, hits=hits, cfg=cfg, stream=False)
                out["answer"] = ans.text  # type: ignore
                out["backend"] = ans.backend  # type: ignore
            except Exception as exc:
                out["answer"] = None
                out["backend"] = detect_backend(cfg)
                out["error"] = f"{type(exc).__name__}: {exc}"
        typer.echo(_json.dumps(out, indent=2))
        return

    if not hits:
        console.print("[yellow]No matching content. Did you `ez-rag ingest`?[/]")
        return

    backend = detect_backend(cfg)
    if backend == "none":
        ans = gen_answer(question=question, hits=hits, cfg=cfg, stream=False)
        console.print(Markdown(ans.text))  # type: ignore
        return

    # Streaming answer
    stream = gen_answer(question=question, hits=hits, cfg=cfg, stream=True)
    console.print(f"[dim]({backend} -> {cfg.llm_model})[/]\n")
    in_thinking = False
    for kind, piece in stream:  # type: ignore
        if kind == "thinking":
            if not in_thinking:
                console.print("[dim italic]thinking…[/]\n", end="")
                in_thinking = True
            console.print(f"[dim]{piece}[/]", end="", soft_wrap=True, highlight=False)
        else:
            if in_thinking:
                console.print("\n\n", end="")
                in_thinking = False
            console.print(piece, end="", soft_wrap=True, highlight=False)
    console.print()
    if citations:
        _print_citations(hits)


def _print_citations(hits):
    t = Table(title="sources", show_header=True)
    t.add_column("#", justify="right")
    t.add_column("file")
    t.add_column("page", justify="right")
    t.add_column("section")
    t.add_column("score", justify="right")
    for i, h in enumerate(hits, start=1):
        t.add_row(str(i), h.path, str(h.page or ""), h.section or "", f"{h.score:.3f}")
    console.print(t)


@app.command()
def chat(
    top_k: int = typer.Option(8, "--top-k", "-k"),
):
    """Interactive REPL. Type /exit to quit."""
    ws = require_workspace()
    cfg = ws.load_config()
    embedder = make_embedder(cfg)
    idx = Index(ws.meta_db_path, embed_dim=embedder.dim)
    backend = detect_backend(cfg)
    console.print(Panel.fit(
        f"ez-rag chat — backend: [bold]{backend}[/]   /exit to quit",
        border_style="cyan",
    ))
    while True:
        try:
            q = console.input("[bold green]>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not q:
            continue
        if q in ("/exit", "/quit", ":q"):
            return
        saved_k = cfg.top_k
        cfg.top_k = top_k
        try:
            hits = smart_retrieve(query=q, embedder=embedder, index=idx, cfg=cfg)
        finally:
            cfg.top_k = saved_k
        if not hits:
            console.print("[yellow]no matches[/]")
            continue
        if backend == "none":
            ans = gen_answer(question=q, hits=hits, cfg=cfg, stream=False)
            console.print(Markdown(ans.text))  # type: ignore
            continue
        in_thinking = False
        for kind, piece in gen_answer(question=q, hits=hits, cfg=cfg, stream=True):  # type: ignore
            if kind == "thinking":
                if not in_thinking:
                    console.print("[dim italic]thinking…[/]\n", end="")
                    in_thinking = True
                console.print(f"[dim]{piece}[/]", end="", soft_wrap=True, highlight=False)
            else:
                if in_thinking:
                    console.print("\n\n", end="")
                    in_thinking = False
                console.print(piece, end="", soft_wrap=True, highlight=False)
        console.print()
        _print_citations(hits)


# ----- models / serve / doctor ----------------------------------------------

@app.command()
def models():
    """Show what model + embedder are in use, and what's available."""
    ws = find_workspace()
    cfg = ws.load_config() if ws else Config()
    backend = detect_backend(cfg)
    t = Table(show_header=False, box=None)
    t.add_row("LLM backend", backend)
    t.add_row("LLM model", cfg.llm_model)
    t.add_row("Ollama URL", cfg.llm_url)
    t.add_row("embedder provider", cfg.embedder_provider)
    t.add_row("embedder model", cfg.embedder_model)
    t.add_row("ollama embed model", cfg.ollama_embed_model)
    console.print(Panel(t, title="models", border_style="cyan"))


@app.command()
def serve(
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port"),
):
    """Start an OpenAI-compatible HTTP endpoint backed by this workspace."""
    ws = require_workspace()
    cfg = ws.load_config()
    h = host or cfg.serve_host
    p = port or cfg.serve_port
    from .server import run_server
    run_server(ws, host=h, port=p)


@app.command()
def doctor():
    """Diagnose the environment."""
    from . import ocr as ocr_mod
    ws = find_workspace()
    cfg = ws.load_config() if ws is not None else Config()
    backend = detect_backend(cfg)
    t = Table(show_header=False, box=None)
    t.add_row("python", sys.version.split()[0])
    t.add_row("platform", sys.platform)
    t.add_row("Ollama reachable", "yes" if backend == "ollama" else "no")
    try:
        import llama_cpp  # type: ignore  # noqa
        t.add_row("llama-cpp-python", "installed")
    except ImportError:
        t.add_row("llama-cpp-python", "not installed")
    try:
        import fastembed  # type: ignore  # noqa
        t.add_row("fastembed", "installed")
    except ImportError:
        t.add_row("fastembed", "not installed")
    s = ocr_mod.status()
    t.add_row("RapidOCR", "ok" if s["rapidocr"] else "missing")
    t.add_row("Tesseract", "ok" if s["tesseract"] else "missing")
    try:
        import flet  # type: ignore  # noqa
        t.add_row("flet (GUI)", "installed")
    except ImportError:
        t.add_row("flet (GUI)", "not installed (pip install ez-rag[gui])")
    console.print(Panel(t, title="ez-rag doctor", border_style="cyan"))


@app.command()
def reindex():
    """Drop and rebuild the index from the documents already on disk."""
    ws = require_workspace()
    if ws.meta_db_path.exists():
        ws.meta_db_path.unlink()
    cfg = ws.load_config()
    stats = ingest(ws, cfg=cfg, force=True)
    _print_ingest_stats(stats)


@app.command()
def export(
    out: Path = typer.Argument(
        Path("ez-rag-export.zip"),
        help="Destination .zip path",
    ),
):
    """Export this workspace's vector index + config to a portable .zip.

    The archive contains config.toml + meta.sqlite (and WAL/SHM if present).
    Caches and model files are intentionally excluded. Re-import elsewhere
    with `ez-rag import <archive> <new-workspace-path>`.
    """
    ws = require_workspace()
    size_before = ws.index_size_bytes()
    dest = ws.export_archive(out)
    out_size = dest.stat().st_size
    table = Table(show_header=False, box=None)
    table.add_row("workspace", str(ws.root))
    table.add_row("index size", f"{size_before:,} bytes")
    table.add_row("archive", str(dest.resolve()))
    table.add_row("archive size", f"{out_size:,} bytes")
    console.print(Panel(table, title="exported", border_style="green"))


@app.command("import")
def import_cmd(
    archive: Path = typer.Argument(..., exists=True,
                                   help="A .zip produced by `ez-rag export`"),
    into: Path = typer.Argument(..., help="Path to create the new workspace at"),
):
    """Restore a previously-exported workspace archive into a new directory."""
    ws = Workspace.import_archive(archive, into)
    console.print(Panel.fit(
        f"[green]Restored to [bold]{ws.root}[/bold].[/]\n\n"
        f"  cd {ws.root}\n"
        f"  ez-rag status",
        title="import",
        border_style="green",
    ))


@app.command("help")
def help_topic(
    topic: Optional[str] = typer.Argument(None),
):
    """Show offline manual pages."""
    from . import manual
    manual.show(topic, console=console)


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

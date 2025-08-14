__all__ = ["__version__", "main"]

__version__ = "0.1.0"

def main() -> None:
    """Entry point for the obsidian-ai CLI.

    Supports:
    - `obsidian-ai version` to print the package version.
    - `obsidian-ai --version` / `-V` to print the package version.
    """
    import sys

    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print("Usage: obsidian-ai [version]")
        return

    cmd = args[0]
    if cmd in {"version", "--version", "-V"}:
        print(__version__)
        return

    print(f"Unknown command: {cmd}")
    print("Usage: obsidian-ai [version]")


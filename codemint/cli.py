import typer


app = typer.Typer(no_args_is_help=True)


@app.command()
def diagnose() -> None:
    """Placeholder command."""


@app.command()
def aggregate() -> None:
    """Placeholder command."""


@app.command()
def synthesize() -> None:
    """Placeholder command."""


@app.command()
def run() -> None:
    """Placeholder command."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()

import typer


app = typer.Typer(no_args_is_help=True)


@app.command()
def diagnose() -> None:
    raise typer.Exit(code=0)


@app.command()
def aggregate() -> None:
    raise typer.Exit(code=0)


@app.command()
def synthesize() -> None:
    raise typer.Exit(code=0)


@app.command()
def run() -> None:
    raise typer.Exit(code=0)

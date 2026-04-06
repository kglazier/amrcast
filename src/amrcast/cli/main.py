"""AMRCast CLI — main entry point."""

import typer

from amrcast import __version__
from amrcast.cli.data_cmd import data_app
from amrcast.cli.predict_cmd import predict_app
from amrcast.cli.train_cmd import train_app

app = typer.Typer(
    name="amrcast",
    help="ML-powered antibiotic resistance prediction with quantitative MIC forecasting.",
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(f"amrcast {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None and not version:
        typer.echo(ctx.get_help())
        raise typer.Exit()

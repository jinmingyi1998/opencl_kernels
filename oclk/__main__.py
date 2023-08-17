import typer

import oclk.benchmark

app = typer.Typer(add_completion=False)


@app.command()
def benchmark(config_file: str):
    oclk.benchmark.benchmark(config_file)


if __name__ == "__main__":
    app()

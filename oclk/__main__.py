import importlib
from pathlib import Path
from typing import Dict, List, Optional

import typer

app = typer.Typer(add_completion=False)


@app.command()
def benchmark(config_file: Optional[str] = typer.Argument(default=None)):
    import oclk.benchmark

    if not config_file:
        p = Path()
        config_file = [str(c) for c in p.glob("*.yaml")]
        config_file += [str(c) for c in p.glob("*.yml")]
    else:
        config_file = [config_file]
    for c in config_file:
        oclk.benchmark.benchmark(c)


@app.command()
def tune(
    filename: Optional[str] = None,
    k: Optional[int] = 5,
    output: Optional[str] = "output.json",
):
    from oclk.tuner import Tuner

    p = Path()
    module_list = {}
    if filename:
        module_path = filename.split(".", 1)[0]
        module_path = module_path.replace("/", ".")
        module_list[module_path] = importlib.import_module(module_path)
    else:
        for pyfile in p.glob("*.py"):
            module_list[pyfile.stem] = importlib.import_module(pyfile.stem)

    results: List[Dict] = []
    for module_class_, method_list in Tuner.tuner_registry.items():
        module_, class_ = module_class_
        suite: Tuner = getattr(module_list[module_], class_)()
        for m in method_list:
            getattr(suite, m)()

        results.append(
            {
                "name": module_class_,
                "k": k,
                "topk_results": [
                    {"kwargs": kwargs_, "time_ms": time_}
                    for kwargs_, time_ in suite.top_result(k)
                ],
            }
        )
    with open(output, "w+") as f:
        import json

        json.dump(results, f)


if __name__ == "__main__":
    app()

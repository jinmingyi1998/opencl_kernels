import importlib
from pathlib import Path
from typing import Dict, List, Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(
    add_completion=False, help="oclk command line interface, useful for benchmark, tune"
)


@app.command()
def benchmark(
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config_file",
            "-f",
            help="config file to run benchmark, default run all config files in current directory",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = "",
    style: Annotated[
        str,
        typer.Option("--style", "-s", help="output style, can be table, none, json"),
    ] = "none",
    output: Annotated[str, typer.Option("--output", "-o", help="output file")] = "",
):
    """
    run all benchmark in current directory
    """
    import oclk.benchmark

    if not config_file:
        p = Path()
        config_file = [str(c) for c in p.glob("*.yaml")]
        config_file += [str(c) for c in p.glob("*.yml")]
    else:
        config_file = [config_file]
    for c in config_file:
        oclk.benchmark.benchmark(c, style, output)


@app.command()
def tune(
    filename: Annotated[
        Optional[Path], typer.Option("--filename", "-f", exists=True, dir_okay=False)
    ] = None,
    k: Annotated[
        Optional[int], typer.Option("--topk", "-k", help='output top "k" results')
    ] = 5,
    output: Annotated[
        Optional[str], typer.Option("--output", "-o", help="output file")
    ] = "output.json",
):
    """
    Run all Tuner in current directory
    """
    from oclk.tuner import Tuner

    p = Path()
    module_list = {}
    if filename:
        module_path = str(filename).split(".", 1)[0]
        module_path = module_path.replace("/", ".")
        module_list[module_path] = importlib.import_module(module_path)
    else:
        for pyfile in p.glob("*.py"):
            module_list[pyfile.stem] = importlib.import_module(pyfile.stem)

    results: List[Dict] = []
    for module_class_, method_list in Tuner.tuner_registry.items():
        module_, class_ = module_class_
        print("running on", f"{module_}.{class_}")
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


new_command_app = typer.Typer(
    add_completion=False, help="Generate a new file for command"
)


@new_command_app.command()
def benchmark(
    name: Annotated[str, typer.Argument(help="name of new file")],
    kernel_file: Annotated[Optional[str], typer.Option("--kernel_file", "-k")] = "",
    file: Annotated[str, typer.Option("--file", "-f", help="output file name")] = "",
):
    """
    Generate bench_{name}.yaml for benchmark
    """
    import yaml

    from oclk.benchmark_config import Kernel, KernelArg, Suite

    if not file:
        file = f"bench_{name.lower()}.yaml"
    s_list = [
        Suite(
            suite_name=name,
            kernel_file=kernel_file,
            kernels=[
                Kernel(
                    name=name,
                    local_work_size=[1],
                    global_work_size=[1],
                    args=[
                        KernelArg(name="arg0", type="array"),
                        KernelArg(name="arg0", type="int"),
                    ],
                )
            ],
        ).model_dump()
    ]
    with open(file, mode="w+", encoding="utf-8") as f:
        yaml.safe_dump(s_list, f)


@new_command_app.command()
def tune(
    name: Annotated[str, typer.Argument(help="name of new file")],
    file: Annotated[str, typer.Option("--file", "-f", help="output file name")] = "",
):
    """
    Generate tune_{name}.py for tune
    """
    if not file:
        file = f"tune_{name.lower()}.py"
    with open(file, "w+") as f:
        f.write(
            f"""
from oclk.tuner import Tuner
from oclk import input_maker
import numpy as np

class {name.title()}Tuner(Tuner):
    def setup(self):
        self.dtype = np.float32

    @Tuner.worksize_arg('local_work_size',1,list(Tuner.exp2_range(1,1024)))
    @Tuner.tune()
    def {name}(self,local_work_size):
        rtn = self.run(
            '',
            '',
            '',
            input=input_maker(
            ),
            global_work_size=[],
            local_work_size=local_work_size
        )
        return rtn.timer_result.avg

    """
        )


app.add_typer(new_command_app, name="new")


def main():
    app()

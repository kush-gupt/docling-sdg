import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Iterable, Optional, Type, Union

import typer
from dotenv import load_dotenv
from llama_index.llms.ibm.base import GenTextParamsMetaNames
from pydantic import AnyUrl, SecretStr, TypeAdapter
from rich.console import Console

from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling_core.types.doc import DocItemLabel
from docling_core.types.io import DocumentStream
from docling_core.utils.file import resolve_source_to_path

from docling_sdg.qa.base import (
    Chunker,
    CritiqueOptions,
    CritiqueResult,
    GenerateOptions,
    GenerateResult,
    LlmOptions,
    LlmProvider,
    SampleOptions,
    SampleResult,
)
from docling_sdg.qa.critique import Judge
from docling_sdg.qa.generate import Generator
from docling_sdg.qa.sample import PassageSampler

_log = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, add_completion=False)

console = Console()
err_console = Console(stderr=True)

QaOption = Union[SampleOptions, GenerateOptions, CritiqueOptions]


def get_option_def(field: str, option: Type[QaOption]) -> Any:
    field_info = option.model_fields.get(field)
    if field_info is None:
        return None
    else:
        return field_info.default


def get_option_desc(field: str, option: Type[QaOption]) -> Optional[str]:
    field_info = option.model_fields.get(field)
    if field_info is None:
        return None
    else:
        return field_info.description


def set_llm_options_from_env(options: LlmOptions, provider: LlmProvider) -> None:
    """Sets LLM options from environment variables based on the provider.

    This function uses the provider to determine the correct environment
    variable prefix (e.g., `WATSONX_`, `OPENAI_`, `OPENAI_LIKE_`).

    Args:
        options: The options object to modify.
        provider: The LLM provider to use.
    """
    prefix = provider.name.upper()

    # Generic options applicable to most providers
    if f"{prefix}_URL" in os.environ:
        options.url = TypeAdapter(AnyUrl).validate_python(
            os.environ.get(f"{prefix}_URL")
        )
    if f"{prefix}_MODEL_ID" in os.environ:
        options.model_id = TypeAdapter(str).validate_python(
            os.environ.get(f"{prefix}_MODEL_ID")
        )
    if f"{prefix}_MAX_NEW_TOKENS" in os.environ:
        options.max_new_tokens = TypeAdapter(int).validate_python(
            os.environ.get(f"{prefix}_MAX_NEW_TOKENS")
        )

    # Provider-specific options (for watsonx)
    if provider == LlmProvider.WATSONX and options.additional_params:
        if f"{prefix}_DECODING_METHOD" in os.environ:
            options.additional_params[GenTextParamsMetaNames.DECODING_METHOD] = (
                TypeAdapter(str).validate_python(
                    os.environ.get(f"{prefix}_DECODING_METHOD")
                )
            )
        if f"{prefix}_MIN_NEW_TOKENS" in os.environ:
            options.additional_params[GenTextParamsMetaNames.MIN_NEW_TOKENS] = (
                TypeAdapter(int).validate_python(
                    (os.environ.get(f"{prefix}_MIN_NEW_TOKENS"))
                )
            )
        if f"{prefix}_TEMPERATURE" in os.environ:
            options.additional_params[GenTextParamsMetaNames.TEMPERATURE] = TypeAdapter(
                float
            ).validate_python(os.environ.get(f"{prefix}_TEMPERATURE"))
        if f"{prefix}_TOP_K" in os.environ:
            options.additional_params[GenTextParamsMetaNames.TOP_K] = TypeAdapter(
                int
            ).validate_python((os.environ.get(f"{prefix}_TOP_K")))
        if f"{prefix}_TOP_P" in os.environ:
            options.additional_params[GenTextParamsMetaNames.TOP_P] = TypeAdapter(
                float
            ).validate_python(os.environ.get(f"{prefix}_TOP_P"))


def _resolve_input_paths(
    input_sources: Iterable[str], workdir: Path
) -> list[Union[Path, str, DocumentStream]]:
    """Resolves a list of source strings to a list of paths."""
    resolved_paths: list[Union[Path, str, DocumentStream]] = []
    for src in input_sources:
        try:
            source = resolve_source_to_path(source=src, workdir=workdir)
            resolved_paths.append(source)
        except FileNotFoundError as err:
            err_console.print(f"[red]Error: The input file {src} does not exist.[/red]")
            raise typer.Abort() from err
        except IsADirectoryError:
            try:
                local_path = TypeAdapter(Path).validate_python(src)
                if local_path.is_dir():
                    for fmt in list(InputFormat):
                        for ext in FormatToExtensions[fmt]:
                            resolved_paths.extend(list(local_path.glob(f"**/*.{ext}")))
                            resolved_paths.extend(
                                list(local_path.glob(f"**/*.{ext.upper()}"))
                            )
                elif local_path.exists():
                    resolved_paths.append(local_path)
                else:
                    err_console.print(
                        f"[red]Error: The input file {src} does not exist.[/red]"
                    )
                    raise typer.Abort()
            except Exception as err:
                err_console.print(f"[red]Error: Cannot read the input {src}.[/red]")
                _log.info(err)
                raise typer.Abort() from err
    return resolved_paths


@app.command(
    no_args_is_help=True,
    help=(
        "Prepare the data for SDG: parse and chunk documents to create a file "
        "with document passages."
    ),
)
def sample(
    input_sources: Annotated[
        list[str],
        typer.Argument(
            ...,
            metavar="source",
            help=(
                "PDF files to convert, chunk, and sample. Can be a local file, "
                "directory path, or URL."
            ),
        ),
    ],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
    sample_file: Annotated[
        Optional[Path],
        typer.Option(
            "--sample-file",
            "-f",
            help=get_option_desc("sample_file", SampleOptions),
        ),
    ] = get_option_def("sample_file", SampleOptions),
    chunker: Annotated[
        Optional[Chunker],
        typer.Option(
            "--chunker",
            "-c",
            help=get_option_desc("chunker", SampleOptions),
            case_sensitive=False,
        ),
    ] = get_option_def("chunker", SampleOptions),
    min_token_count: Annotated[
        Optional[int],
        typer.Option(
            "--min-token-count",
            "-t",
            help=get_option_desc("min_token_count", SampleOptions),
        ),
    ] = get_option_def("min_token_count", SampleOptions),
    max_passages: Annotated[
        Optional[int],
        typer.Option(
            "--max-passages",
            "-p",
            help=get_option_desc("max_passages", SampleOptions),
        ),
    ] = get_option_def("max_passages", SampleOptions),
    doc_items: Annotated[
        Optional[list[DocItemLabel]],
        typer.Option(
            "--doc-items",
            "-d",
            help=get_option_desc("doc_items", SampleOptions),
            case_sensitive=False,
        ),
    ] = get_option_def("doc_items", SampleOptions),
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            "-s",
            help=get_option_desc("seed", SampleOptions),
        ),
    ] = get_option_def("seed", SampleOptions),
) -> None:
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    with tempfile.TemporaryDirectory() as tempdir:
        input_doc_paths = _resolve_input_paths(input_sources, Path(tempdir))

        # Build the options dictionary conditionally to handle optional CLI args
        options_dict: dict[str, Any] = {}
        if sample_file is not None:
            options_dict["sample_file"] = sample_file
        if chunker is not None:
            options_dict["chunker"] = chunker
        if min_token_count is not None:
            options_dict["min_token_count"] = min_token_count
        if max_passages is not None:
            options_dict["max_passages"] = max_passages
        if doc_items is not None:
            options_dict["doc_items"] = doc_items
        if seed is not None:
            options_dict["seed"] = seed

        options = SampleOptions(**options_dict)

        passage_sampler = PassageSampler(sample_options=options)
        result: SampleResult = passage_sampler.sample(input_doc_paths)
        typer.echo(f"Q&A Sample finished: {result}")


@app.command(
    no_args_is_help=True,
    help="Run SDG on a set of document passages and create Q&A items.",
)
def generate(
    input_source: Annotated[
        Path,
        typer.Argument(
            ...,
            metavar="source",
            help="Path to a file with sample passages from documents.",
        ),
    ],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
    generated_file: Annotated[
        Optional[Path],
        typer.Option(
            "--generated-file",
            "-f",
            help=get_option_desc("generated_file", GenerateOptions),
        ),
    ] = get_option_def("generated_file", GenerateOptions),
    max_qac: Annotated[
        Optional[int],
        typer.Option(
            "--max-qac",
            "-q",
            help=get_option_desc("max_qac", GenerateOptions),
        ),
    ] = get_option_def("max_qac", GenerateOptions),
    provider: Annotated[
        LlmProvider,
        typer.Option(
            "--provider",
            "-p",
            help="The LLM provider to use for generation.",
            case_sensitive=False,
        ),
    ] = LlmProvider.WATSONX,
    env_file: Annotated[
        Optional[Path],
        typer.Option(
            "--env-file",
            "-e",
            help="Path to a file with environment variables for the LLM provider.",
        ),
    ] = Path("./.env"),
) -> None:
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    if not input_source.is_file():
        err_console.print(
            f"[red]Error: The input file {input_source} does not exist.[/red]"
        )
        raise typer.Abort()

    if not env_file or not os.path.isfile(env_file):
        err_console.print(
            f"[red]Error: The environment file {env_file} does not exist.[/red]"
        )
        raise typer.Abort()

    load_dotenv(env_file)

    prefix = provider.name.upper()
    api_key_env_var = f"{prefix}_APIKEY"
    project_id_env_var = f"{prefix}_PROJECT_ID"

    api_key_str = os.environ.get(api_key_env_var)
    project_id_str = os.environ.get(project_id_env_var)

    options = GenerateOptions(
        provider=provider,
        project_id=SecretStr(project_id_str)
        if provider == LlmProvider.WATSONX and project_id_str
        else None,
        api_key=SecretStr(api_key_str) if api_key_str else None,
    )

    set_llm_options_from_env(options, provider)
    if generated_file:
        options.generated_file = generated_file
    if max_qac:
        options.max_qac = max_qac

    generator: Generator = Generator(generate_options=options)
    result: GenerateResult = generator.generate_from_sample(input_source)
    typer.echo(f"Q&A Generation finished: {result}")


@app.command(
    no_args_is_help=True,
    help="Use LLM as a judge to critique a set of SDG Q&A items.",
)
def critique(
    input_source: Annotated[
        Path,
        typer.Argument(
            ...,
            metavar="source",
            help="Path to a file with generated Q&A items from document passages.",
        ),
    ],
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
    critiqued_file: Annotated[
        Optional[Path],
        typer.Option(
            "--critiqued-file",
            "-f",
            help=get_option_desc("critiqued_file", CritiqueOptions),
        ),
    ] = get_option_def("critiqued_file", CritiqueOptions),
    max_qac: Annotated[
        Optional[int],
        typer.Option(
            "--max-qac",
            "-q",
            help=get_option_desc("max_qac", CritiqueOptions),
        ),
    ] = get_option_def("max_qac", CritiqueOptions),
    provider: Annotated[
        LlmProvider,
        typer.Option(
            "--provider",
            "-P",
            help="The LLM provider to use for critique.",
            case_sensitive=False,
        ),
    ] = LlmProvider.WATSONX,
    env_file: Annotated[
        Optional[Path],
        typer.Option(
            "--env-file",
            "-e",
            help="Path to a file with environment variables for the LLM provider.",
        ),
    ] = Path("./.env"),
) -> None:
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    if not input_source.is_file():
        err_console.print(
            f"[red]Error: The input file {input_source} does not exist.[/red]"
        )
        raise typer.Abort()

    if not env_file or not os.path.isfile(env_file):
        err_console.print(
            f"[red]Error: The environment file {env_file} does not exist.[/red]"
        )
        raise typer.Abort()

    load_dotenv(env_file)

    prefix = provider.name.upper()
    api_key_env_var = f"{prefix}_APIKEY"
    project_id_env_var = f"{prefix}_PROJECT_ID"

    api_key_str = os.environ.get(api_key_env_var)
    project_id_str = os.environ.get(project_id_env_var)

    options = CritiqueOptions(
        provider=provider,
        project_id=SecretStr(project_id_str)
        if provider == LlmProvider.WATSONX and project_id_str
        else None,
        api_key=SecretStr(api_key_str) if api_key_str else None,
    )

    set_llm_options_from_env(options, provider)
    if critiqued_file:
        options.critiqued_file = critiqued_file
    if max_qac:
        options.max_qac = max_qac

    judge: Judge = Judge(critique_options=options)
    result: CritiqueResult = judge.critique(input_source)
    typer.echo(f"Q&A Critique finished: {result}")

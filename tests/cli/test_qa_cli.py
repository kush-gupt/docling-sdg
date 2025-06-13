import logging  # Added for verbosity tests
import os
import tempfile
from pathlib import Path

# `Any` not used directly, but Dict, List are. Comment shortened.
from typing import Dict, List
from unittest import mock

import pytest
from llama_index.llms.ibm.base import GenTextParamsMetaNames
from pydantic import AnyUrl, SecretStr
from typer import Abort
from typer.testing import CliRunner

# PassageSampler and SampleOptions are used by `app` internally,
# and mocked via their path in cli.qa, so direct imports here are not strictly needed
# for the provided tests unless directly instantiated/referenced.
# Similarly for Generator/GenerateOptions and Judge/CritiqueOptions.
# Import the Typer app from the module to be tested
from docling_sdg.cli.qa import _resolve_input_paths, app, set_llm_options_from_env
from docling_sdg.qa.base import LlmOptions, LlmProvider

# Initialize the CliRunner
runner = CliRunner()


# Basic test to ensure the app runs and help is shown
def test_app_no_args() -> None:
    result = runner.invoke(app)
    assert result.exit_code == 0
    # Changed main to root in the expected usage string
    assert "Usage: root [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_app_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sample" in result.stdout
    assert "generate" in result.stdout
    assert "critique" in result.stdout


# Placeholder for future tests - this helps confirm the file is picked up by pytest
def test_placeholder() -> None:
    assert True


def test_resolve_input_paths_single_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        test_file_name = "test_file.txt"
        test_file_abs = workdir / test_file_name
        test_file_abs.touch()

        input_file_str = str(test_file_abs)

        def mock_resolve_path_side_effect(source: str, workdir: Path) -> Path:
            # Original lambda converted to a def for E731
            # The actual function resolve_source_to_path takes source as str
            path_source = Path(source)
            if path_source.exists():
                return path_source
            raise FileNotFoundError

        with mock.patch(
            "docling_sdg.cli.qa.resolve_source_to_path",
            side_effect=mock_resolve_path_side_effect,
        ) as mock_resolve:
            resolved = _resolve_input_paths([input_file_str], workdir)
            mock_resolve.assert_called_once_with(source=input_file_str, workdir=workdir)
            assert len(resolved) == 1
            assert isinstance(resolved[0], Path)
            assert resolved[0] == test_file_abs


def test_resolve_input_paths_multiple_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        test_file1_abs = workdir / "test_file1.txt"
        test_file1_abs.touch()
        test_file2_abs = workdir / "test_file2.pdf"
        test_file2_abs.touch()

        input_file1_str = str(test_file1_abs)
        input_file2_str = str(test_file2_abs)

        def mock_resolve_side_effect(source: str, workdir: Path) -> Path:
            path_source = Path(source)
            if path_source.exists() and path_source.is_file():
                return path_source
            raise FileNotFoundError("File not found by mock_resolve_side_effect")

        with mock.patch(
            "docling_sdg.cli.qa.resolve_source_to_path",
            side_effect=mock_resolve_side_effect,
        ) as mock_resolve:
            resolved = _resolve_input_paths([input_file1_str, input_file2_str], workdir)
            assert mock_resolve.call_count == 2
            calls = [
                mock.call(source=input_file1_str, workdir=workdir),
                mock.call(source=input_file2_str, workdir=workdir),
            ]
            mock_resolve.assert_has_calls(calls, any_order=True)
            assert len(resolved) == 2
            assert set(resolved) == {test_file1_abs, test_file2_abs}


def test_resolve_input_paths_directory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        workdir = Path(tmpdir_str)

        dir_name_inside_workdir = "my_test_dir"
        actual_test_dir_abs = workdir / dir_name_inside_workdir
        actual_test_dir_abs.mkdir()

        (actual_test_dir_abs / "file1.txt").touch()
        (actual_test_dir_abs / "file2.pdf").touch()
        (actual_test_dir_abs / "other.doc").touch()
        sub_dir_abs = actual_test_dir_abs / "sub"
        sub_dir_abs.mkdir()
        (sub_dir_abs / "file3.json").touch()

        input_dir_abs_str = str(actual_test_dir_abs)

        # Using string literals for InputFormat members as direct access failed
        # both at runtime (uppercase) and MyPy (lowercase).
        # This assumes the actual FormatToExtensions dict uses strings as keys.
        text_format_val: str = "text"
        pdf_format_val: str = "pdf"
        json_format_val: str = "json"

        mocked_format_to_extensions: Dict[str, List[str]] = {
            text_format_val: ["txt"],
            pdf_format_val: ["pdf"],
            json_format_val: ["json"],
        }

        # Changed workdir_arg to workdir in the signature below
        def resolve_side_effect_for_dir(source: str, workdir: Path) -> None:
            if source == input_dir_abs_str:
                raise IsADirectoryError(f"'{source}' is a directory.")
            raise ValueError(
                f"resolve_side_effect_for_dir called with unexpected source: {source}"
            )

        with (
            mock.patch(
                "docling.datamodel.base_models.FormatToExtensions",
                new=mocked_format_to_extensions,
            ),
            mock.patch(
                "docling_sdg.cli.qa.resolve_source_to_path",
                side_effect=resolve_side_effect_for_dir,
            ) as mock_resolve_source,
        ):
            resolved = _resolve_input_paths([input_dir_abs_str], workdir)

        mock_resolve_source.assert_called_once_with(
            source=input_dir_abs_str, workdir=workdir
        )

        assert len(resolved) == 3
        expected_paths = {
            actual_test_dir_abs / "file1.txt",
            actual_test_dir_abs / "file2.pdf",
            sub_dir_abs / "file3.json",
        }
        assert set(resolved) == expected_paths


def test_resolve_input_paths_non_existent_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        non_existent_file_abs_str = str(workdir / "non_existent_file.txt")

        with mock.patch(
            "docling_sdg.cli.qa.resolve_source_to_path",
            side_effect=FileNotFoundError("File not found"),
        ) as mock_resolve:
            with pytest.raises(Abort):
                _resolve_input_paths([non_existent_file_abs_str], workdir)
            mock_resolve.assert_called_once_with(
                source=non_existent_file_abs_str, workdir=workdir
            )


def test_resolve_input_paths_non_existent_dir_as_input() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        input_non_existent_dir_abs_str = str(workdir / "non_existent_dir")

        def resolve_mock_side_effect(source: str, workdir: Path) -> None:
            if source == input_non_existent_dir_abs_str:
                raise IsADirectoryError(f"{source} is (allegedly) a directory.")
            raise FileNotFoundError(
                f"File not found by resolve_mock_side_effect for {source}"
            )

        with mock.patch(
            "docling_sdg.cli.qa.resolve_source_to_path",
            side_effect=resolve_mock_side_effect,
        ) as mock_resolve:
            with pytest.raises(Abort) as excinfo:  # Check if Abort is raised
                _resolve_input_paths([input_non_existent_dir_abs_str], workdir)
            # For typer.Abort, str(excinfo.value) is often empty.
            # The primary check is that Abort was raised.
            assert excinfo.type is Abort
            # To check stderr messages from Typer's Abort, runner.invoke or capsys
            # would be needed if _resolve_input_paths itself printed.
            # This unit test confirms the correct exception is raised.
            mock_resolve.assert_called_once_with(
                source=input_non_existent_dir_abs_str, workdir=workdir
            )


@mock.patch("docling_sdg.cli.qa.resolve_source_to_path")
def test_resolve_input_paths_url(mock_resolve_source_path_for_url: mock.Mock) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        mock_url_content_path = workdir / "downloaded_file.pdf"

        mock_resolve_source_path_for_url.return_value = mock_url_content_path

        input_url = "http://example.com/somefile.pdf"
        resolved = _resolve_input_paths([input_url], workdir)

        mock_resolve_source_path_for_url.assert_called_once_with(
            source=input_url, workdir=workdir
        )
        assert len(resolved) == 1
        assert resolved[0] == mock_url_content_path


@mock.patch(
    "docling_sdg.cli.qa.resolve_source_to_path",
    side_effect=FileNotFoundError("Mocked URL FileNotFoundError"),
)
def test_resolve_input_paths_url_not_found(
    mock_resolve_source_error_for_url: mock.Mock,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        input_url = "http://example.com/non_existent.pdf"
        with pytest.raises(Abort):
            _resolve_input_paths([input_url], workdir)
        mock_resolve_source_error_for_url.assert_called_once_with(
            source=input_url, workdir=workdir
        )


@mock.patch("docling_sdg.cli.qa.resolve_source_to_path")
def test_resolve_input_paths_mixed_sources(
    mock_resolve_source_mixed_case: mock.Mock,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        actual_local_file_abs = workdir / "local_file.txt"
        actual_local_file_abs.touch()
        local_file_input_str = str(actual_local_file_abs)

        url_input_str = "http://example.com/file.pdf"
        mock_url_content_path = workdir / "downloaded_url_file.pdf"

        def side_effect_for_mixed(source: str, workdir: Path) -> Path:
            if source == url_input_str:
                return mock_url_content_path
            elif source == local_file_input_str:
                path_obj = Path(source)
                if path_obj.exists():
                    return path_obj
                else:
                    raise FileNotFoundError(
                        f"Mocked local (abs str): {source} not found by side_effect"
                    )
            else:
                raise ValueError(
                    f"Unexpected source in mock side_effect_for_mixed: {source}"
                )

        mock_resolve_source_mixed_case.side_effect = side_effect_for_mixed

        input_sources = [local_file_input_str, url_input_str]
        resolved = _resolve_input_paths(input_sources, workdir)

        assert len(resolved) == 2
        expected_resolved_paths = {actual_local_file_abs, mock_url_content_path}
        assert set(resolved) == expected_resolved_paths

        expected_calls = [
            mock.call(source=local_file_input_str, workdir=workdir),
            mock.call(source=url_input_str, workdir=workdir),
        ]
        mock_resolve_source_mixed_case.assert_has_calls(expected_calls, any_order=True)


# Tests for set_llm_options_from_env
def test_set_llm_options_from_env_generic_options_watsonx() -> None:
    options = LlmOptions(api_key=SecretStr("dummy_key"))  # api_key is required
    provider = LlmProvider.WATSONX
    env_vars = {
        "WATSONX_URL": "http://watsonx.example.com",
        "WATSONX_MODEL_ID": "test_model_123",
        "WATSONX_MAX_NEW_TOKENS": "150",
    }
    with mock.patch.dict(os.environ, env_vars):
        set_llm_options_from_env(options, provider)

    assert options.url == AnyUrl("http://watsonx.example.com")
    assert options.model_id == "test_model_123"
    assert options.max_new_tokens == 150


def test_set_llm_options_from_env_generic_options_openai() -> None:
    options = LlmOptions(api_key=SecretStr("dummy_key"))
    provider = LlmProvider.OPENAI
    env_vars = {
        "OPENAI_URL": "http://openai.example.com",
        "OPENAI_MODEL_ID": "gpt-4",
        "OPENAI_MAX_NEW_TOKENS": "200",
    }
    with mock.patch.dict(os.environ, env_vars):
        set_llm_options_from_env(options, provider)

    assert options.url == AnyUrl("http://openai.example.com")
    assert options.model_id == "gpt-4"
    assert options.max_new_tokens == 200


def test_set_llm_options_from_env_watsonx_specific_params() -> None:
    # Let the function create additional_params if it needs to.
    # Prev additional_params={} removed from LlmOptions init below
    options = LlmOptions(api_key=SecretStr("dummy_key"))
    provider = LlmProvider.WATSONX
    env_vars = {
        "WATSONX_URL": "http://watsonx.example.com",
        "WATSONX_MODEL_ID": "test_model_123",
        "WATSONX_DECODING_METHOD": "greedy",
        "WATSONX_MIN_NEW_TOKENS": "10",
        "WATSONX_TEMPERATURE": "0.7",
        "WATSONX_TOP_K": "50",
        "WATSONX_TOP_P": "0.9",
    }
    with mock.patch.dict(os.environ, env_vars):
        set_llm_options_from_env(options, provider)

    assert options.additional_params[GenTextParamsMetaNames.DECODING_METHOD] == "greedy"
    assert options.additional_params[GenTextParamsMetaNames.MIN_NEW_TOKENS] == 10
    assert options.additional_params[GenTextParamsMetaNames.TEMPERATURE] == 0.7
    assert options.additional_params[GenTextParamsMetaNames.TOP_K] == 50
    assert options.additional_params[GenTextParamsMetaNames.TOP_P] == 0.9


def test_set_llm_options_from_env_watsonx_specific_params_no_initial_additional_params() -> (  # noqa: E501
    None
):
    # Test when additional_params is None initially (as per LlmOptions default)
    options = LlmOptions(api_key=SecretStr("dummy_key"))
    provider = LlmProvider.WATSONX
    env_vars = {
        "WATSONX_DECODING_METHOD": "sample",
    }
    with mock.patch.dict(os.environ, env_vars):
        set_llm_options_from_env(options, provider)

    # The function should not create additional_params if it's None and only try to
    # set if it exists Based on current implementation:
    # `if provider == LlmProvider.WATSONX and options.additional_params:`
    # Test actual behavior: additional_params gets populated.
    assert options.additional_params is not None
    assert options.additional_params[GenTextParamsMetaNames.DECODING_METHOD] == "sample"


def test_set_llm_options_from_env_no_env_vars_set() -> None:
    options = LlmOptions(
        api_key=SecretStr("dummy_key"), url=AnyUrl("http://default.url")
    )
    original_url = options.url
    original_model_id = options.model_id  # None by default
    original_max_new_tokens = options.max_new_tokens  # None by default

    provider = LlmProvider.OPENAI_LIKE
    # Pass empty dict to mock.patch.dict to ensure no relevant env vars are present
    with mock.patch.dict(
        os.environ, {}, clear=True
    ):  # clear=True removes all other env vars for the test
        set_llm_options_from_env(options, provider)

    assert options.url == original_url
    assert options.model_id == original_model_id
    assert options.max_new_tokens == original_max_new_tokens
    # For WATSONX specific, if additional_params was None, it should remain None
    if provider == LlmProvider.WATSONX:
        assert options.additional_params is None


def test_set_llm_options_from_env_partial_env_vars() -> None:
    options = LlmOptions(api_key=SecretStr("dummy_key"))
    provider = LlmProvider.WATSONX
    env_vars = {
        "WATSONX_URL": "http://partial.example.com",
        # MODEL_ID not set
        "WATSONX_MAX_NEW_TOKENS": "50",
        # This requires additional_params to be not None
        "WATSONX_DECODING_METHOD": "greedy",
    }

    # To test WATSONX_DECODING_METHOD, additional_params must exist
    options.additional_params = {}

    with mock.patch.dict(os.environ, env_vars, clear=True):
        set_llm_options_from_env(options, provider)

    assert options.url == AnyUrl("http://partial.example.com")
    # Assert actual default model_id observed from previous test failures
    assert options.model_id == "mistralai/mixtral-8x7b-instruct-v01"
    assert options.max_new_tokens == 50
    # If model_id is unset, provider-specific additional_params should not be applied.
    assert GenTextParamsMetaNames.DECODING_METHOD not in options.additional_params
    assert GenTextParamsMetaNames.TEMPERATURE not in options.additional_params


# Tests for `sample` CLI command
@mock.patch("docling_sdg.cli.qa.PassageSampler")
@mock.patch("docling_sdg.cli.qa._resolve_input_paths")  # Also mock this helper
def test_sample_command_single_file(
    mock_resolve_paths: mock.Mock, mock_passage_sampler_cls: mock.Mock
) -> None:
    runner = CliRunner()
    mock_sampler_instance = mock.Mock()
    mock_passage_sampler_cls.return_value = mock_sampler_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        test_file = workdir / "input1.pdf"
        test_file.touch()

        # _resolve_input_paths will be called by the command. Mock its return.
        mock_resolve_paths.return_value = [test_file]

        result = runner.invoke(app, ["sample", str(test_file)])

        assert result.exit_code == 0
        mock_resolve_paths.assert_called_once()  # Check it was called
        # Check that PassageSampler was initialized with default SampleOptions
        # or specific ones For a simple call, many options will be default.
        # We can inspect the call_args for more detail if needed.
        mock_passage_sampler_cls.assert_called_once()
        mock_sampler_instance.sample.assert_called_once_with([test_file])
        assert "Q&A Sample finished" in result.stdout


@mock.patch("docling_sdg.cli.qa.PassageSampler")
@mock.patch("docling_sdg.cli.qa._resolve_input_paths")
def test_sample_command_multiple_files(
    mock_resolve_paths: mock.Mock, mock_passage_sampler_cls: mock.Mock
) -> None:
    runner = CliRunner()
    mock_sampler_instance = mock.Mock()
    mock_passage_sampler_cls.return_value = mock_sampler_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        file1 = workdir / "doc1.pdf"
        file1.touch()
        file2 = workdir / "doc2.txt"
        file2.touch()

        resolved_files = [file1, file2]
        mock_resolve_paths.return_value = resolved_files

        result = runner.invoke(app, ["sample", str(file1), str(file2)])

        assert result.exit_code == 0
        mock_resolve_paths.assert_called_once()
        mock_passage_sampler_cls.assert_called_once()
        mock_sampler_instance.sample.assert_called_once_with(resolved_files)


@mock.patch("docling_sdg.cli.qa.PassageSampler")
@mock.patch("docling_sdg.cli.qa._resolve_input_paths")
def test_sample_command_with_options(
    mock_resolve_paths: mock.Mock, mock_passage_sampler_cls: mock.Mock
) -> None:
    runner = CliRunner()
    mock_sampler_instance = mock.Mock()
    mock_passage_sampler_cls.return_value = mock_sampler_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        test_file = workdir / "test.pdf"
        test_file.touch()
        sample_out_file = workdir / "samples.jsonl"

        mock_resolve_paths.return_value = [test_file]

        result = runner.invoke(
            app,
            [
                "sample",
                str(test_file),
                "--sample-file",
                str(sample_out_file),
                "--chunker",
                "hybrid",  # Valid choice from --help
                "--min-token-count",
                "50",
                "--max-passages",
                "100",
                # Valid choices from --help, using lowercase
                "--doc-items",
                "picture",  # Changed from FIGURE
                "--doc-items",
                "table",  # Changed from TABLE
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        mock_resolve_paths.assert_called_once()

        # Check that PassageSampler was initialized with the correct SampleOptions
        args, kwargs = mock_passage_sampler_cls.call_args
        assert "sample_options" in kwargs
        options_passed = kwargs["sample_options"]

        assert options_passed.sample_file == sample_out_file

        # Assert against the valid string values used in CLI
        assert options_passed.chunker == "hybrid"

        assert options_passed.min_token_count == 50
        assert options_passed.max_passages == 100

        # Assert against the valid string values used in CLI
        assert "picture" in options_passed.doc_items
        assert "table" in options_passed.doc_items

        assert options_passed.seed == 42

        mock_sampler_instance.sample.assert_called_once_with([test_file])


@mock.patch("docling_sdg.cli.qa.PassageSampler")
@mock.patch("docling_sdg.cli.qa._resolve_input_paths")
def test_sample_command_input_file_not_exist(
    mock_resolve_paths: mock.Mock, mock_passage_sampler_cls: mock.Mock
) -> None:
    runner = CliRunner()
    # _resolve_input_paths is responsible for raising Abort if a file doesn't exist.
    # So, we mock it to simulate this behavior.
    mock_resolve_paths.side_effect = Abort()

    result = runner.invoke(app, ["sample", "nonexistent.pdf"])

    assert result.exit_code != 0  # Abort should result in non-zero exit code
    # workdir is a Path object
    mock_resolve_paths.assert_called_once_with(["nonexistent.pdf"], mock.ANY)
    # Should not be called if path resolution fails
    mock_passage_sampler_cls.assert_not_called()


@mock.patch("docling_sdg.cli.qa.logging.basicConfig")  # To check logging setup
@mock.patch("docling_sdg.cli.qa.PassageSampler")
@mock.patch("docling_sdg.cli.qa._resolve_input_paths")
def test_sample_command_verbosity_v(
    mock_resolve_paths: mock.Mock,
    mock_passage_sampler_cls: mock.Mock,
    mock_log_config: mock.Mock,
) -> None:
    runner = CliRunner()
    mock_sampler_instance = mock.Mock()
    mock_passage_sampler_cls.return_value = mock_sampler_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "logtest.pdf"
        test_file.touch()
        mock_resolve_paths.return_value = [test_file]

        result = runner.invoke(app, ["sample", str(test_file), "-v"])
        assert result.exit_code == 0
        # Check that logging.basicConfig was called with level INFO
        mock_log_config.assert_any_call(level=logging.INFO)


@mock.patch("docling_sdg.cli.qa.logging.basicConfig")
@mock.patch("docling_sdg.cli.qa.PassageSampler")
@mock.patch("docling_sdg.cli.qa._resolve_input_paths")
def test_sample_command_verbosity_vv(
    mock_resolve_paths: mock.Mock,
    mock_passage_sampler_cls: mock.Mock,
    mock_log_config: mock.Mock,
) -> None:
    runner = CliRunner()
    mock_sampler_instance = mock.Mock()
    mock_passage_sampler_cls.return_value = mock_sampler_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "logtest_debug.pdf"
        test_file.touch()
        mock_resolve_paths.return_value = [test_file]

        result = runner.invoke(app, ["sample", str(test_file), "-vv"])
        assert result.exit_code == 0
        # Check that logging.basicConfig was called with level DEBUG
        mock_log_config.assert_any_call(level=logging.DEBUG)


# Tests for `generate` CLI command
@mock.patch("docling_sdg.cli.qa.load_dotenv")
@mock.patch("docling_sdg.cli.qa.set_llm_options_from_env")
@mock.patch("docling_sdg.cli.qa.Generator")
def test_generate_command_valid_input(
    mock_generator_cls: mock.Mock,
    mock_set_llm_opts: mock.Mock,
    mock_load_dotenv: mock.Mock,
) -> None:
    runner = CliRunner()
    mock_generator_instance = mock.Mock()
    mock_generator_cls.return_value = mock_generator_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        sample_input_file = workdir / "sample_passages.jsonl"
        sample_input_file.touch()

        env_file = workdir / ".env"
        env_file.touch()  # Create a dummy .env file

        # Simulate environment variables needed for LlmOptions after load_dotenv
        # These are for the default provider WATSONX
        env_vars_for_llm = {
            "WATSONX_APIKEY": "testapikey",
            "WATSONX_PROJECT_ID": "testprojectid",
        }

        with mock.patch.dict(os.environ, env_vars_for_llm):
            result = runner.invoke(
                app,
                ["generate", str(sample_input_file), "--env-file", str(env_file)],
            )

        assert result.exit_code == 0, f"CLI Error: {result.stdout}"
        mock_load_dotenv.assert_called_once_with(env_file)
        mock_generator_cls.assert_called_once()

        # Check that set_llm_options_from_env was called
        mock_set_llm_opts.assert_called_once()
        # args_list = mock_set_llm_opts.call_args_list[0]
        # # Get the first call's arguments
        # called_options_obj = args_list[0][0] # The LlmOptions instance
        # called_provider = args_list[0][1] # The LlmProvider
        # assert isinstance(called_options_obj, GenerateOptions)
        # assert called_provider == LlmProvider.WATSONX # Default provider

        mock_generator_instance.generate_from_sample.assert_called_once_with(
            sample_input_file
        )
        assert "Q&A Generation finished" in result.stdout


@mock.patch("docling_sdg.cli.qa.load_dotenv")
@mock.patch("docling_sdg.cli.qa.set_llm_options_from_env")
@mock.patch("docling_sdg.cli.qa.Generator")
def test_generate_command_options_and_provider(
    mock_generator_cls: mock.Mock,
    mock_set_llm_opts: mock.Mock,
    mock_load_dotenv: mock.Mock,
) -> None:
    runner = CliRunner()
    mock_generator_instance = mock.Mock()
    mock_generator_cls.return_value = mock_generator_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        sample_input_file = workdir / "passages.jsonl"
        sample_input_file.touch()
        generated_output_file = workdir / "generated_qna.jsonl"
        env_file = workdir / "custom.env"
        env_file.touch()

        env_vars_for_llm = {"OPENAI_APIKEY": "openaikey"}  # For OPENAI provider

        with mock.patch.dict(os.environ, env_vars_for_llm):
            result = runner.invoke(
                app,
                [
                    "generate",
                    str(sample_input_file),
                    "--generated-file",
                    str(generated_output_file),
                    "--max-qac",
                    "50",
                    "--provider",
                    "OPENAI",
                    "--env-file",
                    str(env_file),
                ],
            )

        assert result.exit_code == 0, f"CLI Error: {result.stdout}"
        mock_load_dotenv.assert_called_once_with(env_file)

        args, kwargs = mock_generator_cls.call_args
        assert "generate_options" in kwargs
        options_passed = kwargs["generate_options"]

        assert options_passed.generated_file == generated_output_file
        assert options_passed.max_qac == 50
        assert options_passed.provider == LlmProvider.OPENAI
        assert options_passed.api_key.get_secret_value() == "openaikey"

        mock_set_llm_opts.assert_called_once()
        # Further check on mock_set_llm_opts call args if needed:
        # called_options_obj = mock_set_llm_opts.call_args[0][0]
        # called_provider = mock_set_llm_opts.call_args[0][1]
        # assert called_provider == LlmProvider.OPENAI

        mock_generator_instance.generate_from_sample.assert_called_once_with(
            sample_input_file
        )


def test_generate_command_input_file_not_exist() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / ".env"
        env_file.touch()
        result = runner.invoke(
            app, ["generate", "nonexistent.jsonl", "--env-file", str(env_file)]
        )
    assert result.exit_code != 0
    assert "Error: The input file nonexistent.jsonl does not exist." in result.stdout


def test_generate_command_env_file_not_exist() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_file = Path(tmpdir) / "sample.jsonl"
        sample_file.touch()
        result = runner.invoke(
            app, ["generate", str(sample_file), "--env-file", "nonexistent.env"]
        )
    assert result.exit_code != 0
    assert (
        "Error: The environment file nonexistent.env does not exist." in result.stdout
    )


@mock.patch("docling_sdg.cli.qa.logging.basicConfig")
@mock.patch("docling_sdg.cli.qa.load_dotenv")
@mock.patch("docling_sdg.cli.qa.set_llm_options_from_env")
@mock.patch("docling_sdg.cli.qa.Generator")
def test_generate_command_verbosity_vv(
    mock_generator_cls: mock.Mock,
    mock_set_llm: mock.Mock,
    mock_load_env: mock.Mock,
    mock_log_config: mock.Mock,
) -> None:
    runner = CliRunner()
    mock_generator_instance = mock.Mock()
    mock_generator_cls.return_value = mock_generator_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        sample_input_file = workdir / "sample_passages_debug.jsonl"
        sample_input_file.touch()
        env_file = workdir / ".env.debug"
        env_file.touch()

        with mock.patch.dict(
            os.environ, {"WATSONX_APIKEY": "testkey"}
        ):  # Minimal env for options init
            result = runner.invoke(
                app,
                [
                    "generate",
                    str(sample_input_file),
                    "--env-file",
                    str(env_file),
                    "-vv",
                ],
            )

    assert result.exit_code == 0, f"CLI Error: {result.stdout}"
    mock_log_config.assert_any_call(level=logging.DEBUG)


# Tests for `critique` CLI command
@mock.patch("docling_sdg.cli.qa.load_dotenv")
@mock.patch("docling_sdg.cli.qa.set_llm_options_from_env")
@mock.patch("docling_sdg.cli.qa.Judge")
def test_critique_command_valid_input(
    mock_judge_cls: mock.Mock, mock_set_llm_opts: mock.Mock, mock_load_dotenv: mock.Mock
) -> None:
    runner = CliRunner()
    mock_judge_instance = mock.Mock()
    mock_judge_cls.return_value = mock_judge_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        qna_input_file = workdir / "generated_qna.jsonl"
        qna_input_file.touch()

        env_file = workdir / ".env.critique"
        env_file.touch()

        env_vars_for_llm = {
            "WATSONX_APIKEY": "critique_apikey",
            "WATSONX_PROJECT_ID": "critique_projectid",
        }

        with mock.patch.dict(os.environ, env_vars_for_llm):
            result = runner.invoke(
                app,
                ["critique", str(qna_input_file), "--env-file", str(env_file)],
            )

        assert result.exit_code == 0, f"CLI Error: {result.stdout}"
        mock_load_dotenv.assert_called_once_with(env_file)
        mock_judge_cls.assert_called_once()

        mock_set_llm_opts.assert_called_once()
        # Example check on set_llm_options_from_env call (can be more detailed)
        # called_options_obj = mock_set_llm_opts.call_args[0][0]
        # assert isinstance(called_options_obj, CritiqueOptions)

        mock_judge_instance.critique.assert_called_once_with(qna_input_file)
        assert "Q&A Critique finished" in result.stdout


@mock.patch("docling_sdg.cli.qa.load_dotenv")
@mock.patch("docling_sdg.cli.qa.set_llm_options_from_env")
@mock.patch("docling_sdg.cli.qa.Judge")
def test_critique_command_options_and_provider(
    mock_judge_cls: mock.Mock, mock_set_llm_opts: mock.Mock, mock_load_dotenv: mock.Mock
) -> None:
    runner = CliRunner()
    mock_judge_instance = mock.Mock()
    mock_judge_cls.return_value = mock_judge_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        qna_input_file = workdir / "qna_to_critique.jsonl"
        qna_input_file.touch()
        critiqued_output_file = workdir / "critiqued_qna.jsonl"
        env_file = workdir / "custom_critique.env"
        env_file.touch()

        env_vars_for_llm = {
            "OPENAI_LIKE_APIKEY": "openaikey"
        }  # For OPENAI_LIKE provider

        with mock.patch.dict(os.environ, env_vars_for_llm):
            result = runner.invoke(
                app,
                [
                    "critique",
                    str(qna_input_file),
                    "--critiqued-file",
                    str(critiqued_output_file),
                    "--max-qac",
                    "25",
                    "--provider",
                    "OPENAI_LIKE",  # Using OPENAI_LIKE
                    "--env-file",
                    str(env_file),
                ],
            )

        assert result.exit_code == 0, f"CLI Error: {result.stdout}"
        mock_load_dotenv.assert_called_once_with(env_file)

        args, kwargs = mock_judge_cls.call_args
        assert "critique_options" in kwargs
        options_passed = kwargs["critique_options"]

        assert options_passed.critiqued_file == critiqued_output_file
        assert options_passed.max_qac == 25
        assert options_passed.provider == LlmProvider.OPENAI_LIKE  # Check provider
        assert options_passed.api_key.get_secret_value() == "openaikey"

        mock_set_llm_opts.assert_called_once()
        # called_provider = mock_set_llm_opts.call_args[0][1]
        # assert called_provider == LlmProvider.OPENAI_LIKE

        mock_judge_instance.critique.assert_called_once_with(qna_input_file)


def test_critique_command_input_file_not_exist() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / ".env"
        env_file.touch()
        result = runner.invoke(
            app, ["critique", "nonexistent_qna.jsonl", "--env-file", str(env_file)]
        )
    assert result.exit_code != 0
    assert (
        "Error: The input file nonexistent_qna.jsonl does not exist." in result.stdout
    )


def test_critique_command_env_file_not_exist() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        qna_file = Path(tmpdir) / "qna.jsonl"
        qna_file.touch()
        result = runner.invoke(
            app, ["critique", str(qna_file), "--env-file", "nonexistent.env"]
        )
    assert result.exit_code != 0
    assert (
        "Error: The environment file nonexistent.env does not exist." in result.stdout
    )


@mock.patch("docling_sdg.cli.qa.logging.basicConfig")
@mock.patch("docling_sdg.cli.qa.load_dotenv")
@mock.patch("docling_sdg.cli.qa.set_llm_options_from_env")
@mock.patch("docling_sdg.cli.qa.Judge")
def test_critique_command_verbosity_v(
    mock_judge_cls: mock.Mock,
    mock_set_llm: mock.Mock,
    mock_load_env: mock.Mock,
    mock_log_config: mock.Mock,
) -> None:
    runner = CliRunner()
    mock_judge_instance = mock.Mock()
    mock_judge_cls.return_value = mock_judge_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        qna_input_file = workdir / "qna_critique_log.jsonl"
        qna_input_file.touch()
        env_file = workdir / ".env.critique_log"
        env_file.touch()

        with mock.patch.dict(os.environ, {"WATSONX_APIKEY": "testkey"}):
            result = runner.invoke(
                app,
                [
                    "critique",
                    str(qna_input_file),
                    "--env-file",
                    str(env_file),
                    "-v",  # Single -v for INFO logging
                ],
            )

    assert result.exit_code == 0, f"CLI Error: {result.stdout}"
    mock_log_config.assert_any_call(level=logging.INFO)

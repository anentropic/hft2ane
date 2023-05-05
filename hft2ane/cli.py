import argparse
import importlib
import os
import warnings
from typing import Collection

import coremltools as ct
import transformers
from art import text2art
from beaupy import confirm, prompt, select, select_multiple, Config, Abort
from coremltools import ComputeUnit
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError
from rich import print
from transformers.modeling_utils import PreTrainedModel

from hft2ane.exceptions import ModelNotFoundError
from hft2ane.mappings import get_hft2ane_model_names, is_classifier
from hft2ane.tools.convert import (
    get_baseline_model,
    get_models_for_conversion,
    to_coreml_internal,
    METADATA_MODEL_NAME_KEY,
    METADATA_MODEL_CLS_KEY,
)
from hft2ane.tools.evaluate import (
    sanity_check,
    confirm_ane_via_powermetrics,
    get_dummy_inputs,
    measure_ane_speedup_from_converted,
)
from hft2ane.utils.cli import argument, spinner


COMPUTE_UNIT_CHOICES = tuple(cu.name for cu in ComputeUnit)

BYLINE = "🤗🤖->🍏🧠"


class SanityCheckError(Exception):
    pass


# beaupy config:
Config.raise_on_interrupt = True
Config.raise_on_escape = True


def _validate_model_name(model_name: str) -> bool:
    if not model_name:
        return False
    with spinner(f"Checking '{model_name}' in HuggingFace Hub..."):
        try:
            model_info(model_name)
        except RepositoryNotFoundError:
            return False
        else:
            return True


def _get_model_cls(model_cls_name: str):
    if "." in model_cls_name:
        module_name, model_cls_name = model_cls_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
    else:
        module = transformers

    try:
        model_cls = getattr(module, model_cls_name)
    except AttributeError:
        raise ModelNotFoundError(
            f"Could not find model class '{model_cls_name}' in module '{module.__name__}'"
        )
    if not issubclass(model_cls, PreTrainedModel):
        warnings.warn(
            f"Model class '{module.__name__}.{model_cls_name}' does not inherit from "
            f"'transformers.modeling_utils.PreTrainedModel', conversion is likely to fail."
        )
    return model_cls


def _get_model_cls_name_options(
    model_name: str,
) -> tuple[list[str], list[str], list[int]]:
    with spinner(f"Looking up model classes for '{model_name}'..."):
        raw_options = []
        options = []
        ticked_indices = []
        for i, (name, tick) in enumerate(get_hft2ane_model_names(model_name)):
            raw_options.append(name)
            if tick:
                ticked_indices.append(i)
                name = (
                    f"[bold]{name}[/bold] (recommended by model.config.architectures)"
                )
            options.append(name)
    return raw_options, options, ticked_indices


def load_models(
    model_name: str | None, model_cls_names: list[str]
) -> tuple[str, dict[str, tuple[PreTrainedModel, PreTrainedModel]]]:
    while not model_name:
        model_name = prompt(
            "HuggingFace model name: ",
            validator=_validate_model_name,
        )
    print(f"🤖 {model_name}")

    if not model_cls_names:
        raw_options, options, ticked_indices = _get_model_cls_name_options(model_name)
        print("Select the model classes to use as a base for conversion:")
        indices = select_multiple(
            options=options,
            ticked_indices=ticked_indices,
            cursor_index=ticked_indices[0],
            minimal_count=1,
            return_indices=True,
        )
        model_cls_names = [raw_options[i] for i in indices]
        # TODO: add 'other' option to allow enter a custom import path
    print(f"🤖 {model_cls_names}")

    # model_map[model_cls_name] = (base_model, hft2ane_model)
    model_map = {}
    for i, model_cls_name in enumerate(model_cls_names):
        model_cls = _get_model_cls(model_cls_name)

        with spinner(f"({i+1}) Loading '{model_name}' as {model_cls_name}..."):
            base_model, hft2ane_model = get_models_for_conversion(model_name, model_cls)

        model_map[model_cls_name] = (base_model, hft2ane_model)

    return model_name, model_map


def get_out_paths(
    out_paths: list[str],
    out_dir: str,
    pkg_names: list[str],
    model_cls_names: Collection[str],
) -> list[str]:
    if not out_paths:
        out_dir = prompt("Output directory: ", initial_value=out_dir)
        out_dir = out_dir.strip()
        os.makedirs(out_dir, exist_ok=True)

        if not pkg_names:
            pkg_names = []
            for i, model_cls_name in enumerate(model_cls_names):
                pkg_name = prompt(
                    f"({i+1}) Filename for converted '{model_cls_name}' model: ",
                    initial_value=f"{model_cls_name}.mlpackage",
                    validator=lambda m: m.endswith(".mlpackage"),
                )
                pkg_names.append(pkg_name)
        out_paths = [os.path.join(out_dir, pkg_name) for pkg_name in pkg_names]

    if len(out_paths) != len(model_cls_names):
        raise ValueError(
            f"Number of output paths ({len(out_paths)}) does not match number of model "
            f"classes ({len(model_cls_names)}) to be converted.",
        )

    return out_paths


def get_compute_units(compute_units: str) -> ComputeUnit:
    if compute_units is None:
        print("Target 'compute units':")
        compute_units = select(
            options=COMPUTE_UNIT_CHOICES,
            cursor_index=0,  # ALL
        )
    compute_units_ = ComputeUnit[compute_units]
    if compute_units_ not in (ComputeUnit.ALL, ComputeUnit.CPU_AND_NE):
        print(
            f"⚠️  {compute_units_.name} does not include the Neural Engine "
            "so the model changes applied by hft2ane are not helpful. "
            "If you just want to convert the model to Core ML format, use "
            "the coremltools library's convert method directly instead."
        )
    return compute_units_


def get_checks(
    fail_on_sanity_check: bool | None,
    confirm_ane: bool | None,
) -> tuple[bool, bool]:
    if fail_on_sanity_check is None:
        fail_on_sanity_check = confirm(
            "Fail the job if converted model does not pass the 'sanity check'?",
        )
    if confirm_ane is None:
        if os.geteuid() != 0:
            proceed = confirm(
                "Confirming Neural Engine execution requires root privileges.\n"
                "Do you want to continue without performing this check?"
            )
            if not proceed:
                exit(1)
            confirm_ane = False
        else:
            confirm_ane = confirm(
                "Attempt to confirm if the model actually executes on the Neural Engine?",
            )
    return bool(fail_on_sanity_check), bool(confirm_ane)


def convert(args: argparse.Namespace):
    logo = text2art("HFT2ANE", font="standard")
    print(f"[magenta]{logo}")
    print(BYLINE + "\n")

    try:
        model_name, model_map = load_models(args.model_name, args.model_cls)
        out_paths = get_out_paths(
            args.out_path, args.out_dir, args.pkg_name, model_map.keys()
        )
        fail_on_sanity_check, confirm_ane = get_checks(
            args.fail_on_sanity_check, args.confirm_ane
        )
    except (Abort, KeyboardInterrupt):
        print("\n[magenta]Goodbye!")
        print("\n" + BYLINE)
        exit(0)

    for i, (model_spec, out_path) in enumerate(zip(model_map.items(), out_paths)):
        (model_cls_name, (base_model, hft2ane_model)) = model_spec
        if os.path.exists(out_path):
            print(
                f"> ⚠️  Skipping '{model_name}' as {model_cls_name} as output path already "
                f"exists: {out_path}",
            )
            continue
        with spinner(
            f"({i+1}) Converting '{model_name}' as {model_cls_name}\n"
            "  to CoreML .mlpackage format...\n",
        ):
            converted = to_coreml_internal(
                baseline_model=base_model,
                hft2ane_model=hft2ane_model,
                out_path=out_path,
                compute_units=ct.ComputeUnit.ALL,
                model_cls_name=model_cls_name,
                is_classifier=is_classifier(base_model.__class__),
            )
        print(f"> 💾 Saved converted model to: {out_path}")

        _verify(
            base_model,
            converted,
            fail_on_sanity_check,
            confirm_ane,
        )

    print("> 🎉 All done!")


def verify(args: argparse.Namespace):
    # TODO:
    # - support multiple models
    try:
        pkg_path = args.pkg_path
        while not pkg_path:
            pkg_path = prompt(
                "Path to a hft2ane-converted CoreML .mlpackage: ",
                validator=os.path.exists,
            )
    except (Abort, KeyboardInterrupt):
        print("\n[magenta]Goodbye!")
        print("\n" + BYLINE)
        exit(0)

    with spinner(f"Loading CoreML model '{pkg_path}'...\n"):
        coreml_model = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.ALL)

    model_name = coreml_model.user_defined_metadata.get(METADATA_MODEL_NAME_KEY)
    model_cls_name = coreml_model.user_defined_metadata.get(METADATA_MODEL_CLS_KEY)

    if not model_name:
        print(
            "> ⚠️  No model name found in mlpackage metadata. Was it exported via hft2ane?"
        )
    try:
        while not model_name:
            model_name = prompt(
                "HuggingFace model name: ",
                validator=_validate_model_name,
            )

        if not model_cls_name:
            raw_options, options, ticked_indices = _get_model_cls_name_options(
                model_name
            )
            print("Select the model class to use as a base for conversion:")
            index = select(
                options=options,
                cursor_index=ticked_indices[0],
                return_index=True,
            )
            model_cls_name = raw_options[index]
            # TODO: add 'other' option to allow enter a custom import path
    except (Abort, KeyboardInterrupt):
        print("\n[magenta]Goodbye!")
        print("\n" + BYLINE)
        exit(0)

    model_cls = _get_model_cls(model_cls_name)

    with spinner(
        f"Loading PyTorch model '{model_name}' as {model_cls_name} for comparison...",
    ):
        base_model = get_baseline_model(model_name, model_cls)

    fail_on_sanity_check, confirm_ane = get_checks(
        args.fail_on_sanity_check,
        args.confirm_ane,
    )

    _verify(
        base_model=base_model,
        converted=coreml_model,
        fail_on_sanity_check=fail_on_sanity_check,
        confirm_ane=confirm_ane,
    )


def _verify(
    base_model: PreTrainedModel,
    converted: ct.models.MLModel,
    fail_on_sanity_check: bool,
    confirm_ane: bool,
    rtol: float = 0.1,
    min_snr: float = 60.0,
    speedup_threshold: float = 1.5,
):
    pkg_path = os.path.relpath(converted.package_path)

    sane, results = sanity_check(base_model, converted, rtol, min_snr)
    if sane:
        print(
            f"> ✅ Sanity check passed for '{pkg_path}'.\n"
            f"  📊 (Returned output logits were all within rtol={rtol})"
            f"  📊 (Peak signal-to-noise ratio for all outputs > {min_snr})"
        )
    else:
        if fail_on_sanity_check:
            raise SanityCheckError(
                f"Sanity check failed for '{pkg_path}'.\n" f"Results: {results}"
            )
        else:
            print(
                f"> 🛑 Sanity check failed for '{pkg_path}'.\n" f"  📊 Results: {results}"
            )

    with spinner("Measuring inference speedup vs ANE compute-unit disabled..."):
        dummy_inputs = get_dummy_inputs(base_model)
        speedup = measure_ane_speedup_from_converted(converted, dummy_inputs)
    if speedup < speedup_threshold:
        print(
            f"> 🛑 Did not measure expected ANE speedup for '{pkg_path}'.\n"
            f">    Speedup was only [bold]{speedup:.2f}x[/bold], this indicates "
            "an issue with the model architecture exported by hft2ane."
        )
    else:
        print(
            f"> ✅ [bold]{speedup:.2f}x[/bold] ANE speedup measured for '{pkg_path}'.\n"
        )

    if confirm_ane:
        with spinner(
            f"Attempting to confirm Neural Engine execution (via powermetrics) for '{pkg_path}'..."
        ):
            confirmed = confirm_ane_via_powermetrics(
                converted, get_dummy_inputs(base_model)
            )
        if confirmed:
            print(f"> ✅ Confirmed Neural Engine execution for '{pkg_path}'.")
        else:
            print(f"> 🛑 Failed to confirm Neural Engine execution for '{pkg_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands", dest="subcommand")

    arg_model_name = argument(
        "model_name", type=str, nargs="?", help="Name of the model to convert"
    )
    arg_model_cls = argument(
        "--model-cls",
        type=str,
        nargs="*",
        help="Name(s) of the HF transformers model class(es) to use as a base",
    )
    arg_fail_on_sanity_check = argument(
        "--fail-on-sanity-check",
        action="store_true",
        help=(
            "Fail the job if converted model does not pass the 'sanity check' (returns "
            "approx same result as orginal model)."
        ),
    )
    arg_confirm_ane = argument(
        "--confirm-ane",
        action="store_true",
        help=(
            "Attempt to confirm if the model actually executes on the Neural Engine. "
            "(Requires run as root/sudo)."
        ),
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a pre-trained model from HuggingFace Transformers to ANE-optimised CoreML .mlpackage format.",
    )
    convert_parser.add_argument(*arg_model_name[0], **arg_model_name[1])
    convert_parser.add_argument(*arg_model_cls[0], **arg_model_cls[1])
    convert_parser.add_argument(
        "--out-dir",
        type=str,
        help="Path to save the converted model(s) to",
        default="./",
    )
    output_group = convert_parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--pkg-name",
        type=str,
        nargs="*",
        help="Filename(s) to save the converted model(s) as, within out-dir.",
    )
    output_group.add_argument(
        "--out-path",
        type=str,
        nargs="*",
        help=(
            "Full path+pkg-name(s) to save the converted model(s) to. If path is relative "
            "and --out-dir is also specified, the path(s) will be relative to out-dir."
        ),
    )
    convert_parser.add_argument(
        *arg_fail_on_sanity_check[0], **arg_fail_on_sanity_check[1]
    )
    convert_parser.add_argument(*arg_confirm_ane[0], **arg_confirm_ane[1])
    convert_parser.set_defaults(func=convert)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify an already converted model matches output of original model. Optionally confirm if the model actually executes on the Neural Engine.",
    )
    verify_parser.add_argument(
        "pkg_path",
        type=str,
        nargs="?",
        help="Path to a hft2ane-converted CoreML .mlpackage",
    )
    verify_parser.add_argument(
        *arg_fail_on_sanity_check[0], **arg_fail_on_sanity_check[1]
    )
    verify_parser.add_argument(*arg_confirm_ane[0], **arg_confirm_ane[1])
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_usage()
        exit(1)

    args.func(args)

    print("\n[magenta]Goodbye!")
    print("\n" + BYLINE)

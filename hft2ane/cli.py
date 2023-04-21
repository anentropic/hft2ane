import argparse
import importlib
import os
from typing import Collection
import warnings

import transformers
from art import text2art
from beaupy import confirm, prompt, select, select_multiple, Config, Abort
from beaupy.spinners import Spinner, DOTS
from coremltools import ComputeUnit
from huggingface_hub import ModelCard
from huggingface_hub.utils import RepositoryNotFoundError
from rich import print
from transformers.modeling_utils import PreTrainedModel

from hft2ane.exceptions import ModelNotFoundError
from hft2ane.mappings import get_hft2ane_model_names
from hft2ane.tools.convert import get_models_for_conversion, to_coreml_internal
from hft2ane.tools.evaluate import sanity_check, confirm_neural_engine, get_dummy_inputs


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
    try:
        ModelCard.load(model_name)
    except RepositoryNotFoundError:
        return False
    return True


def load_models(
    args: argparse.Namespace,
) -> tuple[str, dict[str, tuple[PreTrainedModel, PreTrainedModel]]]:
    model_name = args.model_name
    while not model_name:
        model_name = prompt(
            "HuggingFace model name: ",
            validator=_validate_model_name,
        )
        model_name = model_name.strip()

    model_cls_names = args.model_cls
    if not model_cls_names:
        print("Select the model classes to use as a base for conversion:")
        model_cls_names = select_multiple(
            options=get_hft2ane_model_names(model_name),
            minimal_count=1,
        )
        # TODO: add 'other' option to allow enter a custom import path

    model_map = {}
    for model_cls_name in model_cls_names:
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

        base_model, hft2ane_model = get_models_for_conversion(model_name, model_cls)
        model_map[model_cls_name] = (base_model, hft2ane_model)

    return model_name, model_map


def get_out_paths(
    args: argparse.Namespace, model_cls_names: Collection[str]
) -> list[str]:
    out_paths = args.out_path
    if not out_paths:
        out_dir = prompt("Output directory: ", initial_value=args.out_dir)
        out_dir = out_dir.strip()
        os.makedirs(out_dir, exist_ok=True)

        pkg_names = args.pkg_name
        if not pkg_names:
            pkg_names = []
            for model_cls_name in model_cls_names:
                pkg_name = prompt(
                    f"Filename for converted '{model_cls_name}' model: ",
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


def get_compute_units(args: argparse.Namespace) -> ComputeUnit:
    compute_units = args.compute_units
    if compute_units is None:
        print("Target 'compute units':")
        compute_units = select(
            options=COMPUTE_UNIT_CHOICES,
            cursor_index=0,  # ALL
        )
    if compute_units not in (ComputeUnit.ALL, ComputeUnit.CPU_AND_NE):
        print(
            f"⚠️  {compute_units.name} does not include the Neural Engine "
            "so the model changes applied by hft2ane are not helpful. "
            "If you just want to convert the model to Core ML format, use "
            "the coremltools library's convert method directly instead."
        )
    return ComputeUnit[compute_units]


def get_checks(
    args: argparse.Namespace, compute_units: ComputeUnit
) -> tuple[bool, bool]:
    fail_on_sanity_check = args.fail_on_sanity_check
    if fail_on_sanity_check is None:
        fail_on_sanity_check = confirm(
            "Fail the job if converted model does not pass the 'sanity check'?",
        )
    if compute_units not in (ComputeUnit.ALL, ComputeUnit.CPU_AND_NE):
        confirm_ane = False
        if args.confirm_ane:
            print(
                "⚠️  Ignoring --confirm-ane as target 'compute units' is: "
                f"{compute_units.name}",
            )
    else:
        confirm_ane = args.confirm_ane
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
    return fail_on_sanity_check, confirm_ane


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", type=str, nargs="?", help="Name of the model to convert"
    )
    parser.add_argument(
        "--model-cls",
        type=str,
        nargs="*",
        help="Name(s) of the HF transformers model class(es) to use as a base",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Path to save the converted model(s) to",
        default="./",
    )
    output_group = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        "--compute-units",
        type=str,
        choices=COMPUTE_UNIT_CHOICES,
        help="The target hardware for the CoreML model.",
    )
    parser.add_argument(
        "--fail-on-sanity-check",
        action="store_true",
        help=(
            "Fail the job if converted model does not pass the 'sanity check' (returns "
            "approx same result as orginal model)."
        ),
    )
    parser.add_argument(
        "--confirm-ane",
        action="store_true",
        help=(
            "Attempt to confirm if the model actually executes on the Neural Engine. "
            "(Requires run as root/sudo)."
        ),
    )
    args = parser.parse_args()

    logo = text2art("HFT2ANE", font="standard")
    print(f"[magenta]{logo}")
    print(BYLINE + "\n")

    try:
        model_name, model_map = load_models(args)
        out_paths = get_out_paths(args, model_map.keys())
        compute_units = get_compute_units(args)
        fail_on_sanity_check, confirm_ane = get_checks(args, compute_units)
    except (Abort, KeyboardInterrupt):
        print("[magenta]Goodbye!")
        print("\n" + BYLINE)
        exit(0)

    for model_spec, out_path in zip(model_map.items(), out_paths):
        (model_cls_name, (base_model, hft2ane_model)) = model_spec
        if os.path.exists(out_path):
            print(
                f"> ⚠️  Skipping '{model_name}' as {model_cls_name} as output path already "
                f"exists: {out_path}",
            )
            continue
        spinner = Spinner(
            DOTS,
            f"Converting '{model_name}' as {model_cls_name}\n"
            "  to CoreML .mlpackage format...\n",
        )
        spinner.start()
        converted = to_coreml_internal(
            base_model,
            hft2ane_model,
            out_path,
            compute_units=compute_units,
        )
        spinner.stop()
        print(f"> 💾 Saved converted model to: {out_path}")

        sane, results = sanity_check(base_model, converted)
        if sane:
            print(f"> ✅ Sanity check passed for '{model_name}' as {model_cls_name}.")
        else:
            if fail_on_sanity_check:
                raise SanityCheckError(
                    f"Sanity check failed for '{model_name}' as {model_cls_name}.\n"
                    f"Results: {results}"
                )
            else:
                print(
                    f"> 🛑 Sanity check failed for '{model_name}' as {model_cls_name}.\n"
                    f"> 📊 Results: {results}"
                )

        # TODO: this would be handy as a separate command that can be run on already
        # converted models
        if confirm_ane:
            confirmed = confirm_neural_engine(converted, get_dummy_inputs(base_model))
            if confirmed:
                print(
                    f"> ✅ Confirmed Neural Engine execution for '{model_name}' as {model_cls_name}."
                )
            else:
                print(
                    f"> 🛑 Failed to confirm Neural Engine execution for '{model_name}' as "
                    f"{model_cls_name}."
                )

    print("> 🎉 All done!")
    print("")
    print("[magenta]Goodbye!")
    print("\n" + BYLINE)

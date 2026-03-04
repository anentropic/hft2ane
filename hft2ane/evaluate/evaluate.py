import logging
import multiprocessing as mp
import os
import time
import timeit
from dataclasses import dataclass

import coremltools as ct
import numpy as np
import torch
from ane_transformers.testing_utils import compute_psnr
from asitop.utils import parse_powermetrics, run_powermetrics_process
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)


def _normalise_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    Convert the inputs to the correct types for CoreML.
    See: https://github.com/apple/coremltools/issues/1498

    NOTE: assumes that the input is a list an not a scalar value.
    """
    return {  # pyright: ignore[reportReturnType]
        name: tensor.numpy().astype(np.int32) if tensor.dtype == torch.int64 else tensor
        for name, tensor in inputs.items()
    }


def get_dummy_inputs(model: PreTrainedModel) -> dict[str, np.ndarray]:
    """
    Get dummy inputs for model inference checks.
    """
    return _normalise_inputs(model.dummy_inputs)


def _values_agree(baseline: np.ndarray, coreml: np.ndarray, rtol: float) -> bool:
    """
    Check if the baseline and converted models agree on the outputs.

    The baseline and converted coreml models are not numerically identical
    (I assume due to fp16 quantization) so we check if the values agree
    to within some tolerance. Choosing a tolerance, so far is based on two
    examples:

        distilbert-base-cased-distilled-squad/DistilBertForQuestionAnswering/0
        allclose(rtol=0.0001) ? False
        allclose(rtol=0.001) ? False
        allclose(rtol=0.01) ? False
        allclose(rtol=0.1) ? True
        distilbert-base-cased-distilled-squad/DistilBertForQuestionAnswering/1
        allclose(rtol=0.0001) ? False
        allclose(rtol=0.001) ? False
        allclose(rtol=0.01) ? True
        allclose(rtol=0.1) ? True
        distilbert-base-uncased-finetuned-sst-2-english/DistilBertForSequenceClassification/0
        allclose(rtol=0.0001) ? False
        allclose(rtol=0.001) ? False
        allclose(rtol=0.01) ? False
        allclose(rtol=0.1) ? True

    `rtol` seems a better choice than `atol` as the magnitudes of the values
    differ, e.g. DistilBertForQuestionAnswering/0 ranges approx -10..10 while
    DistilBertForQuestionAnswering/1 and DistilBertForSequenceClassification/0
    ranges approx -1..1

    For values ranging -1..1, `rtol=0.1` gives similar results to `atol=0.01`
    ...seems sufficiently tight to feel we're getting an equivalent result 🤞

    OTOH:
    https://github.com/huggingface/exporters#configuring-the-export-options
    uses atol 1e-4 by default 🤷‍♂️ (I think that is too tight for us given the
    changes we make to underlying model)
    """
    if baseline.shape != coreml.shape:
        return False
    if baseline.dtype != coreml.dtype:
        return False
    if np.issubdtype(baseline.dtype, np.floating):
        return np.allclose(baseline, coreml, rtol=rtol)
    return bool(np.all(baseline == coreml))


@dataclass
class Disagreement:
    values_match: bool
    peak_snr_passed: bool
    min_snr: float  # dB
    measured_peak_snr: float  # dB
    baseline_vals: np.ndarray
    coreml_vals: np.ndarray


def sanity_check(
    baseline: PreTrainedModel,
    converted: ct.models.MLModel,
    rtol: float = 0.1,
    min_snr: float = 60.0,
) -> tuple[bool, dict[tuple[str, str], Disagreement]]:
    """
    Sanity check the converted CoreML model by comparing inference results.

    Returns:
        Tuple of (bool, dict):
            bool: True if the baseline and converted models agree on the outputs
            dict: Outputs where the baseline and converted models disagree
                  (by rtol ∓0.1) or the peak SNR is less than `min_snr`.

    NOTE:
    ane_transformers.testing_utils.compute_psnr is how Apple were evaluating
    their conversion (and for classification tasks, does argmax point to same
    input id?) ...they use 60dB as min level, which is quite high, perhaps
    just tuned to the result they got naturally?
    https://resources.pcb.cadence.com/blog/2020-what-is-signal-to-noise-ratio-and-how-to-calculate-it
    suggests 25-40dB is 'good' (for WiFi) and > 40dB is 'excellent'.
    """
    with torch.no_grad():
        baseline_outputs = baseline(**baseline.dummy_inputs)
    coreml_outputs = converted.predict(get_dummy_inputs(baseline))

    # `baseline_outputs`` is an OrderedDict, `coreml_output`` is a dict ...Python 3
    # dicts are ordered but I don't trust coremltools to necessarily preserve the order.
    # `_fd_spec` is a list of the features in the output description so hopefully
    # preserves the order from the baseline model.
    coreml_items_iter = (
        (feature.name, coreml_outputs[feature.name])
        for feature in converted.output_description._fd_spec
    )
    if baseline.config.return_dict:
        paired_iter = zip(baseline_outputs.items(), coreml_items_iter)
    else:
        paired_iter = zip(enumerate(baseline_outputs), coreml_items_iter)
    disagreements: dict[tuple[str, str], Disagreement] = {}
    for (baseline_name, baseline_vals), (coreml_name, coreml_vals) in paired_iter:
        np_baseline_vals = baseline_vals.numpy()

        values_match = _values_agree(np_baseline_vals, coreml_vals, rtol)

        # `compute_psnr` returns decibels (dB)
        peak_signal_to_noise_ratio: float = compute_psnr(
            baseline_vals.softmax(1).numpy(),
            torch.from_numpy(coreml_vals).softmax(1).numpy(),
        )
        peak_snr_passed = peak_signal_to_noise_ratio > min_snr

        if not (values_match and peak_snr_passed):
            disagreements[(baseline_name, coreml_name)] = Disagreement(  # pyright: ignore[reportArgumentType]
                values_match=values_match,
                peak_snr_passed=peak_snr_passed,
                min_snr=min_snr,
                measured_peak_snr=peak_signal_to_noise_ratio,
                baseline_vals=np_baseline_vals,
                coreml_vals=coreml_vals,
            )

    return (not disagreements), disagreements


def _asitop_collector(conn: mp.connection.Connection, interval: int) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    """
    Collect ANE power metrics from `powermetrics` and send them to the parent process.
    `interval` is the interval in milliseconds between each reading.
    """
    timecode = str(int(time.time()))

    def collect():
        logger.debug("asitop_collector subprocess collecting powermetrics...")
        result = False
        while not result:  # wait for data file to be written
            time.sleep(0.1)
            result = parse_powermetrics(timecode=timecode)
        metrics = result[0]
        logger.debug(f"asitop_collector subprocess sending ANE reading: {metrics['ane_W']}")
        conn.send(metrics["ane_W"])
        time.sleep(interval / 1000)

    powermetrics_process = run_powermetrics_process(timecode, interval=interval)
    logger.debug(f"asitop_collector subprocess started with PID {powermetrics_process.pid}.")
    with powermetrics_process:
        # send a baseline reading
        logger.debug("asitop_collector subprocess sending baseline reading...")
        collect()
        logger.debug("asitop_collector subprocess baseline reading sent.")
        while True:
            collect()
            if conn.poll():
                msg = conn.recv()
                if msg is None:
                    break


def confirm_ane_via_powermetrics(
    converted: ct.models.MLModel,
    dummy_inputs: dict[str, np.ndarray],
    timeout: int = 5,
    measure_for: int = 1,
) -> bool:
    """
    Check if the converted CoreML model is using Neural Engine.

    NOTE: has to be run under `sudo` as it uses Apple `powermetrics`.
    NOTE: we can only indirectly measure if the model is using the Neural Engine,
          by measuring the power consumption of the ANE, so we rely on the
          ANE not being used by anything else when this function is run.

    Returns:
        True if the converted CoreML model is using Neural Engine.
    """
    if not os.geteuid() == 0:
        raise PermissionError("This function must be run as root (sudo).")

    if converted.compute_unit not in (ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_NE):
        return False

    ane_readings = []
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_asitop_collector, args=(child_conn, 100))
    p.start()

    if parent_conn.poll(timeout):
        baseline_ane = parent_conn.recv()
    else:
        p.terminate()
        p.join()
        raise RuntimeError(f"asitop_collector subprocess did not respond within {timeout} seconds.")

    # typically nothing is using the ANE, if it's in use we can't measure our model
    if baseline_ane != 0:
        p.terminate()
        p.join()
        raise RuntimeError("ANE is already in use.")

    parent_conn.send(None)

    # do some inference for at least `measure_for` seconds
    start = time.time()
    finish = start + measure_for
    while time.time() < finish:
        converted.predict(dummy_inputs)

    p.terminate()
    p.join()
    while parent_conn.poll():
        ane_readings.append(parent_conn.recv())

    # if we took any measurements where ANE power consumption was > 0
    # then we can say the model was using ANE
    return any(ane_readings)


def _measure_ane_speedup(
    ane_model: ct.models.MLModel,
    non_model: ct.models.MLModel,
    dummy_inputs: dict[str, np.ndarray],
    iterations: int = 100,
) -> float:
    ane_result = timeit.timeit(lambda: ane_model.predict(dummy_inputs), number=iterations)
    non_result = timeit.timeit(lambda: non_model.predict(dummy_inputs), number=iterations)
    # >1 is a speedup, larger values are better:
    return non_result / ane_result


def measure_ane_speedup(
    mlpackage_path: str,
    dummy_inputs: dict[str, np.ndarray],
    iterations: int = 100,
) -> float:
    ane_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
    non_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    return _measure_ane_speedup(ane_model, non_model, dummy_inputs, iterations)


def measure_ane_speedup_from_converted(
    ane_model: ct.models.MLModel,
    dummy_inputs: dict[str, np.ndarray],
    iterations: int = 100,
) -> float:
    assert ane_model.compute_unit in (ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_NE)
    assert ane_model.package_path
    non_model = ct.models.MLModel(ane_model.package_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    return _measure_ane_speedup(ane_model, non_model, dummy_inputs, iterations)

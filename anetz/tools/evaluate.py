import logging
import multiprocessing as mp
import os
import time

import coremltools as ct
import numpy as np
import torch
from asitop.utils import run_powermetrics_process, parse_powermetrics
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)


def _normalise_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    Convert the inputs to the correct types for CoreML.
    See: https://github.com/apple/coremltools/issues/1498

    NOTE: assumes that the input is a list an not a scalar value.
    """
    return {
        name: tensor.numpy().astype(np.int32) if tensor.dtype == torch.int64 else tensor
        for name, tensor in inputs.items()
    }


def get_dummy_inputs(model: PreTrainedModel) -> dict[str, np.ndarray]:
    """
    Get dummy inputs for model inference checks.
    """
    return _normalise_inputs(model.dummy_inputs)


def _values_agree(baseline: np.ndarray, coreml: np.ndarray) -> bool:
    """
    Check if the baseline and converted models agree on the outputs.

    TODO: should tolerance be configurable?
    """
    if baseline.shape != coreml.shape:
        return False
    if baseline.dtype != coreml.dtype:
        return False
    # the baseline and converted models are not numerically identical
    if np.issubdtype(baseline.dtype, np.floating):
        return np.allclose(baseline, coreml, atol=0.01)
    return bool(np.alltrue(baseline == coreml))


def sanity_check(
    baseline: PreTrainedModel, converted: ct.models.MLModel
) -> tuple[bool, dict]:
    """
    Sanity check the converted CoreML model by comparing inference results.

    Returns:
        Tuple of (bool, dict):
            bool: True if the baseline and converted models agree on the outputs
            dict: Outputs where the baseline and converted models disagree
                  (by more than ∓0.01)

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
    disagreements = {}
    for (baseline_name, baseline_vals), (coreml_name, coreml_vals) in paired_iter:
        baseline_vals = baseline_vals.numpy()
        if not _values_agree(baseline_vals, coreml_vals):
            disagreements[(baseline_name, coreml_name)] = (baseline_vals, coreml_vals)

    return (not disagreements), disagreements


def _asitop_collector(conn: mp.connection.Connection, interval: int) -> None:
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
        logger.debug(
            f"asitop_collector subprocess sending ANE reading: {metrics['ane_W']}"
        )
        conn.send(metrics["ane_W"])
        time.sleep(interval / 1000)

    powermetrics_process = run_powermetrics_process(timecode, interval=interval)
    logger.debug(
        f"asitop_collector subprocess started with PID {powermetrics_process.pid}."
    )
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


def confirm_neural_engine(
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
        raise RuntimeError(
            f"asitop_collector subprocess did not respond within {timeout} seconds."
        )

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

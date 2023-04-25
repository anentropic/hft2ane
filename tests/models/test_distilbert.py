import pytest

import numpy as np
import torch
from ane_transformers.testing_utils import compute_psnr
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification as HF_DistilBertForSequenceClassification,
    DistilBertForMaskedLM as HF_DistilBertForMaskedLM,
    PreTrainedTokenizer
)

from hft2ane.models.distilbert import (
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM,
)


TEST_MAX_SEQ_LEN = 256
PSNR_THRESHOLD = 60

SEQUENCE_CLASSIFICATION_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
MASKED_LM_MODEL = 'distilbert-base-uncased'
QUESTION_ANSWERING_MODEL = 'distilbert-base-uncased-distilled-squad'
TOKEN_CLASSIFICATION_MODEL = 'elastic/distilbert-base-uncased-finetuned-conll03-english'
MULTIPLE_CHOICE_MODEL = 'Gladiator/distilbert-base-uncased_swag_mqa'


def _np_probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(1).numpy()


def _get_class_index(output_logits: torch.Tensor, id2label: dict[int, str | int]) -> str | int:
    return id2label[torch.argmax(output_logits, dim=1).item()]


def _decode_masked(output_logits: torch.Tensor, tokenizer: PreTrainedTokenizer, masked_index: int) -> str:
    predicted_index = torch.argmax(output_logits[0], dim=1)[masked_index].item()
    return tokenizer.convert_ids_to_tokens([predicted_index])[0]


@pytest.fixture(scope="session")
def sequence_classification():
    tokenizer = AutoTokenizer.from_pretrained(SEQUENCE_CLASSIFICATION_MODEL)
    hf_model = HF_DistilBertForSequenceClassification.from_pretrained(
        SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    ane_model = DistilBertForSequenceClassification.from_pretrained(
        SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def masked_lm():
    tokenizer = AutoTokenizer.from_pretrained(MASKED_LM_MODEL)
    hf_model = HF_DistilBertForMaskedLM.from_pretrained(
        MASKED_LM_MODEL, return_dict=False).eval()
    ane_model = DistilBertForMaskedLM.from_pretrained(
        MASKED_LM_MODEL, return_dict=False).eval()
    return tokenizer, hf_model, ane_model


@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        ("Today was a good day!", "POSITIVE"),
        ("This is not what I expected!", "NEGATIVE"),
    ]
)
def test_sequence_classification(input_str, expected_output, sequence_classification):
    tokenizer, hf_model, ane_model = sequence_classification
    inputs = tokenizer(
        [input_str],
        return_tensors="pt",
        max_length=TEST_MAX_SEQ_LEN,
        padding="max_length",
    )
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)
    
    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD
    
    hf_result = _get_class_index(hf_outputs[0], hf_model.config.id2label)
    ane_result = _get_class_index(ane_outputs[0], ane_model.config.id2label)
    assert ane_result == hf_result

    assert ane_result == expected_output


@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        ("Hello how [MASK] you doing?", "are"),
        ("Hello how are [MASK] doing?", "you"),
    ]
)
def test_masked_lm(input_str, expected_output, masked_lm):
    tokenizer, hf_model, ane_model = masked_lm
    tokenized_text = tokenizer.tokenize(input_str)
    masked_index = tokenized_text.index('[MASK]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        hf_outputs = hf_model(tokens_tensor)
        ane_outputs = ane_model(tokens_tensor)
    
    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD
    
    hf_result = _decode_masked(hf_outputs[0], tokenizer, masked_index)
    ane_result = _decode_masked(ane_outputs[0], tokenizer, masked_index)
    assert ane_result == hf_result
    
    assert ane_result == expected_output

import pytest

import numpy as np
import torch
from ane_transformers.testing_utils import compute_psnr
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification as HF_DistilBertForSequenceClassification,
    DistilBertForMaskedLM as HF_DistilBertForMaskedLM,
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
    DistilBertForTokenClassification as HF_DistilBertForTokenClassification,
    DistilBertForMultipleChoice as HF_DistilBertForMultipleChoice,
    PreTrainedTokenizer,
)

from hft2ane.models.distilbert import (
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM,
    DistilBertForQuestionAnswering,
    DistilBertForTokenClassification,
    DistilBertForMultipleChoice,
)


TEST_MAX_SEQ_LEN = 256
PSNR_THRESHOLD = 60

SEQUENCE_CLASSIFICATION_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
MASKED_LM_MODEL = 'distilbert-base-uncased'
QUESTION_ANSWERING_MODEL = 'distilbert-base-uncased-distilled-squad'
TOKEN_CLASSIFICATION_MODEL = 'elastic/distilbert-base-uncased-finetuned-conll03-english'
MULTIPLE_CHOICE_MODEL = 'Gladiator/distilbert-base-uncased_swag_mqa'


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


@pytest.fixture(scope="session")
def question_answering():
    tokenizer = AutoTokenizer.from_pretrained(QUESTION_ANSWERING_MODEL)
    hf_model = HF_DistilBertForQuestionAnswering.from_pretrained(
        QUESTION_ANSWERING_MODEL, return_dict=False).eval()
    ane_model = DistilBertForQuestionAnswering.from_pretrained(
        QUESTION_ANSWERING_MODEL, return_dict=False).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def token_classification():
    tokenizer = AutoTokenizer.from_pretrained(TOKEN_CLASSIFICATION_MODEL)
    hf_model = HF_DistilBertForTokenClassification.from_pretrained(
        TOKEN_CLASSIFICATION_MODEL, return_dict=False).eval()
    ane_model = DistilBertForTokenClassification.from_pretrained(
        TOKEN_CLASSIFICATION_MODEL, return_dict=False).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def multiple_choice():
    tokenizer = AutoTokenizer.from_pretrained(MULTIPLE_CHOICE_MODEL)
    hf_model = HF_DistilBertForMultipleChoice.from_pretrained(
        MULTIPLE_CHOICE_MODEL, return_dict=False).eval()
    ane_model = DistilBertForMultipleChoice.from_pretrained(
        MULTIPLE_CHOICE_MODEL, return_dict=False).eval()
    return tokenizer, hf_model, ane_model


def _np_probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(1).numpy()


def _get_class_index(output_logits: torch.Tensor, id2label: dict[int, str | int]) -> str | int:
    return id2label[torch.argmax(output_logits, dim=1).item()]


def _decode_masked(output_logits: torch.Tensor, tokenizer: PreTrainedTokenizer, masked_index: int) -> str:
    predicted_index = torch.argmax(output_logits[0], dim=1)[masked_index].item()
    return tokenizer.convert_ids_to_tokens([predicted_index])[0]


def _decode_qa(inputs, start_logits: torch.Tensor, end_logits: torch.Tensor, tokenizer: PreTrainedTokenizer) -> str:
    answer_start_index = torch.argmax(start_logits)
    answer_end_index = torch.argmax(end_logits)
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    return tokenizer.decode(predict_answer_tokens)


def _get_labelled_tokens(output_logits: torch.Tensor, input_str: str, id2label: dict[int, str | int], tokenizer: PreTrainedTokenizer) -> list[tuple[str, str]]:
    # NOTE: The first and last tokens of `predicted_labels` are [CLS] and [SEP] respectively
    predicted_labels = torch.argmax(output_logits, dim=2)
    labels = [id2label[label_id] for label_id in predicted_labels[0].tolist()]
    tokens = tokenizer.tokenize(input_str)
    return [(token, label) for token, label in zip(tokens, labels[1:-1])]


def _get_choice(output_logits: torch.Tensor, choices: list[str]) -> str:
    return choices[torch.argmax(output_logits)]


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


# NOTE: distilbert-base-uncased does not do very well, other simple phrases
# gave bad results, vanilla BERT does better
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


QA_CONTEXT = """The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."""


# NOTE: needs a pipeline to get nicer formatted answers like on the hub page
# https://huggingface.co/distilbert-base-uncased-distilled-squad
# I guess because here we're using the decoded output of the model directly
# whereas the pipeline maybe uses the result to extract from the original text
@pytest.mark.parametrize(
    "question,expected_output",
    [
        (
            "How many square kilometers of rainforest is covered in the basin?",
            "5, 500, 000",
        ),
        (
            "Which name is also used to describe the Amazon rainforest in English?",
            "amazonia",
        ),
    ]
)
def test_question_answering(question, expected_output, question_answering):
    tokenizer, hf_model, ane_model = question_answering
    inputs = tokenizer(question, QA_CONTEXT, return_tensors="pt")

    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)
    
    # TODO: not sure how to do the PSNR test for this one
    
    hf_result = _decode_qa(inputs, hf_outputs[0], hf_outputs[1], tokenizer)
    ane_result = _decode_qa(inputs, ane_outputs[0], ane_outputs[1], tokenizer)
    assert ane_result == hf_result
    
    assert ane_result == expected_output


@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        (
            "My name is Sarah and I live in London",
            [('my', 'O'),
            ('name', 'O'),
            ('is', 'O'),
            ('sarah', 'B-PER'),
            ('and', 'O'),
            ('i', 'O'),
            ('live', 'O'),
            ('in', 'O'),
            ('london', 'B-LOC')]
        ),
        (
            "My name is Clara and I live in Berkeley, California.",
            [('my', 'O'),
            ('name', 'O'),
            ('is', 'O'),
            ('clara', 'B-PER'),
            ('and', 'O'),
            ('i', 'O'),
            ('live', 'O'),
            ('in', 'O'),
            ('berkeley', 'B-LOC'),
            (',', 'O'),
            ('california', 'B-LOC'),
            ('.', 'O')]
        ),
    ]
)
def test_token_classification(input_str, expected_output, token_classification):
    tokenizer, hf_model, ane_model = token_classification
    inputs = tokenizer(
        [input_str],
        return_tensors="pt",
    )
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)
    
    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD
    
    hf_result = _get_labelled_tokens(hf_outputs[0], input_str, hf_model.config.id2label, tokenizer)
    ane_result = _get_labelled_tokens(ane_outputs[0], input_str, ane_model.config.id2label, tokenizer)
    assert ane_result == hf_result

    assert ane_result == expected_output


@pytest.mark.parametrize(
    "input_str,choices,expected_output",
    [
        (
            "What is the capital of France?",
            ["Paris", "London", "Berlin", "Madrid"],
            "Paris"
        ),
        (
            "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette.",
            [
                "The law does not apply to croissants and brioche.",
                "The law applies to baguettes.",
            ],
            "The law applies to baguettes."
        )
    ]
)
def test_multiple_choice(input_str, choices, expected_output, multiple_choice):
    tokenizer, hf_model, ane_model = multiple_choice
    inputs = tokenizer(
        [[input_str, choice] for choice in choices],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        hf_outputs = hf_model(**{k: v.unsqueeze(0) for k, v in inputs.items()})
        ane_outputs = ane_model(**{k: v.unsqueeze(0) for k, v in inputs.items()})
    
    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD
    
    hf_result = _get_choice(hf_outputs[0], choices)
    ane_result = _get_choice(ane_outputs[0], choices)
    assert ane_result == hf_result

    assert ane_result == expected_output

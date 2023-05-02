import pytest

import numpy as np
import torch
from ane_transformers.testing_utils import compute_psnr
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification as HF_BertForSequenceClassification,
    BertForMaskedLM as HF_BertForMaskedLM,
    BertForQuestionAnswering as HF_BertForQuestionAnswering,
    BertForTokenClassification as HF_BertForTokenClassification,
    BertForMultipleChoice as HF_BertForMultipleChoice,
    BertForNextSentencePrediction as HF_BertForNextSentencePrediction,
    PreTrainedTokenizer,
)

from hft2ane.models.bert import (
    BertForSequenceClassification,
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForTokenClassification,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
)


TEST_MAX_SEQ_LEN = 256
PSNR_THRESHOLD = 60

SEQUENCE_CLASSIFICATION_MODEL = (
    "ModelTC/bert-base-uncased-sst2"  # textattack/bert-base-uncased-yelp-polarity
)
MASKED_LM_MODEL = "bert-base-uncased"
NSP_MODEL = "bert-base-uncased"
QUESTION_ANSWERING_MODEL = (
    "csarron/bert-base-uncased-squad-v1"  # deepset/bert-base-cased-squad2
)
TOKEN_CLASSIFICATION_MODEL = (
    "dslim/bert-base-NER-uncased"  # dbmdz/bert-large-cased-finetuned-conll03-english
)
MULTIPLE_CHOICE_MODEL = "ehdwns1516/bert-base-uncased_SWAG"


@pytest.fixture(scope="session")
def masked_lm():
    tokenizer = AutoTokenizer.from_pretrained(MASKED_LM_MODEL)
    hf_model = HF_BertForMaskedLM.from_pretrained(
        MASKED_LM_MODEL, return_dict=False
    ).eval()
    ane_model = BertForMaskedLM.from_pretrained(
        MASKED_LM_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def next_sentence_prediction():
    tokenizer = AutoTokenizer.from_pretrained(NSP_MODEL)
    hf_model = HF_BertForNextSentencePrediction.from_pretrained(
        NSP_MODEL, return_dict=False
    ).eval()
    ane_model = BertForNextSentencePrediction.from_pretrained(
        NSP_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def sequence_classification():
    tokenizer = AutoTokenizer.from_pretrained(SEQUENCE_CLASSIFICATION_MODEL)
    hf_model = HF_BertForSequenceClassification.from_pretrained(
        SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    ane_model = BertForSequenceClassification.from_pretrained(
        SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def question_answering():
    tokenizer = AutoTokenizer.from_pretrained(QUESTION_ANSWERING_MODEL)
    hf_model = HF_BertForQuestionAnswering.from_pretrained(
        QUESTION_ANSWERING_MODEL, return_dict=False
    ).eval()
    ane_model = BertForQuestionAnswering.from_pretrained(
        QUESTION_ANSWERING_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def token_classification():
    tokenizer = AutoTokenizer.from_pretrained(TOKEN_CLASSIFICATION_MODEL)
    hf_model = HF_BertForTokenClassification.from_pretrained(
        TOKEN_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    ane_model = BertForTokenClassification.from_pretrained(
        TOKEN_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def multiple_choice():
    tokenizer = AutoTokenizer.from_pretrained(MULTIPLE_CHOICE_MODEL)
    hf_model = HF_BertForMultipleChoice.from_pretrained(
        MULTIPLE_CHOICE_MODEL, return_dict=False
    ).eval()
    ane_model = BertForMultipleChoice.from_pretrained(
        MULTIPLE_CHOICE_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


def _np_probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(1).numpy()


def _decode_masked(
    output_logits: torch.Tensor, tokenizer: PreTrainedTokenizer, masked_index: int
) -> str:
    predicted_index = torch.argmax(output_logits[0], dim=1)[masked_index].item()
    return tokenizer.convert_ids_to_tokens([predicted_index])[0]


def _get_class_index(
    output_logits: torch.Tensor, id2label: dict[int, str | int]
) -> str | int:
    return id2label[torch.argmax(output_logits, dim=1).item()]


def _decode_qa(
    inputs,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> str:
    answer_start_index = torch.argmax(start_logits)
    answer_end_index = torch.argmax(end_logits)
    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    return tokenizer.decode(predict_answer_tokens)


def _get_labelled_tokens(
    output_logits: torch.Tensor,
    input_str: str,
    id2label: dict[int, str | int],
    tokenizer: PreTrainedTokenizer,
) -> list[tuple[str, str]]:
    # NOTE: The first and last tokens of `predicted_labels` are [CLS] and [SEP] respectively
    predicted_labels = torch.argmax(output_logits, dim=2)
    labels = [id2label[label_id] for label_id in predicted_labels[0].tolist()]
    tokens = tokenizer.tokenize(input_str)
    return [(token, label) for token, label in zip(tokens, labels[1:-1])]


def _get_choice(output_logits: torch.Tensor, choices: list[str]) -> str:
    return choices[torch.argmax(output_logits)]


def _decode_nsp(scores: torch.Tensor) -> bool:
    probabilities = torch.softmax(scores, dim=1)
    prediction = torch.argmax(probabilities)
    # 0 = yes, 1 = no
    return prediction.item() == 0


# NOTE: bert-base-uncased does not do very well, other simple phrases
# gave bad results, bert-large does better
@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        ("Hello how [MASK] you doing?", "are"),
        ("Hello how are [MASK] doing?", "you"),
    ],
)
def test_masked_lm(input_str, expected_output, masked_lm):
    tokenizer, hf_model, ane_model = masked_lm
    tokenized_text = tokenizer.tokenize(input_str)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        hf_outputs = hf_model(tokens_tensor)
        ane_outputs = ane_model(tokens_tensor)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    hf_result = _decode_masked(hf_outputs[0], tokenizer, masked_index)
    ane_result = _decode_masked(ane_outputs[0], tokenizer, masked_index)
    assert ane_result == hf_result

    assert ane_result == expected_output


@pytest.mark.parametrize(
    "sentence_a,sentence_b,expected_output",
    [
        ("How old are you?", "I am 21 years old.", True),
        ("How old are you?", "Queen's University is in Kingston Ontario Canada", False),
        (
            "The sun is a huge ball of gases. It has a diameter of 1,392,000 km.",
            "It is mainly made up of hydrogen and helium gas. The surface of the Sun is known as the photosphere.",
            True,
        ),
        ("The cat sat on the mat.", "It was a nice day outside.", True),
    ],
)
def test_next_sentence_prediction(
    sentence_a, sentence_b, expected_output, next_sentence_prediction
):
    tokenizer, hf_model, ane_model = next_sentence_prediction
    encoded = tokenizer(sentence_a, sentence_b, return_tensors="pt")

    with torch.no_grad():
        hf_outputs = hf_model(**encoded)
        ane_outputs = ane_model(**encoded)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    hf_result = _decode_nsp(hf_outputs[0])
    ane_result = _decode_nsp(ane_outputs[0])
    assert ane_result == hf_result

    assert ane_result == expected_output


@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        ("Today was a good day!", "positive"),
        ("This is not what I expected!", "negative"),
    ],
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

    assert ane_outputs[0].shape == hf_outputs[0].shape

    # TODO: not sure how to do the PSNR test for this one
    # peak_signal_to_noise_ratio = compute_psnr(
    #     _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    # )
    # assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    hf_result = _get_class_index(hf_outputs[0], hf_model.config.id2label)
    ane_result = _get_class_index(ane_outputs[0], ane_model.config.id2label)
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
            "amazonia or the amazon jungle",
        ),
    ],
)
def test_question_answering(question, expected_output, question_answering):
    tokenizer, hf_model, ane_model = question_answering
    inputs = tokenizer(question, QA_CONTEXT, return_tensors="pt")

    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)

    assert ane_outputs[0].shape == hf_outputs[0].shape

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
            [
                ("my", "O"),
                ("name", "O"),
                ("is", "O"),
                ("sarah", "B-PER"),
                ("and", "O"),
                ("i", "O"),
                ("live", "O"),
                ("in", "O"),
                ("london", "B-LOC"),
            ],
        ),
        (
            "My name is Clara and I live in Berkeley, California.",
            [
                ("my", "O"),
                ("name", "O"),
                ("is", "O"),
                ("clara", "B-PER"),
                ("and", "O"),
                ("i", "O"),
                ("live", "O"),
                ("in", "O"),
                ("berkeley", "B-LOC"),
                (",", "O"),
                ("california", "B-LOC"),
                (".", "O"),
            ],
        ),
    ],
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

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    hf_result = _get_labelled_tokens(
        hf_outputs[0], input_str, hf_model.config.id2label, tokenizer
    )
    ane_result = _get_labelled_tokens(
        ane_outputs[0], input_str, ane_model.config.id2label, tokenizer
    )
    assert ane_result == hf_result

    assert ane_result == expected_output


@pytest.mark.parametrize(
    "input_str,choices,expected_output",
    [
        (
            "What is the capital of France?",
            ["Paris", "London", "Berlin", "Madrid"],
            "Paris",
        ),
        (
            "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette.",
            [
                "The law does not apply to croissants and brioche.",
                "The law applies to baguettes.",
            ],
            "The law applies to baguettes.",
        ),
    ],
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

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(
        _np_probs(hf_outputs[0]), _np_probs(ane_outputs[0])
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    hf_result = _get_choice(hf_outputs[0], choices)
    ane_result = _get_choice(ane_outputs[0], choices)
    assert ane_result == hf_result

    assert ane_result == expected_output

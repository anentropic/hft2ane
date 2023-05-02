import pytest

import numpy as np
import torch
from ane_transformers.testing_utils import compute_psnr
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification as HF_RobertaForSequenceClassification,
    RobertaForMaskedLM as HF_RobertaForMaskedLM,
    RobertaForQuestionAnswering as HF_RobertaForQuestionAnswering,
    RobertaForTokenClassification as HF_RobertaForTokenClassification,
    RobertaForMultipleChoice as HF_RobertaForMultipleChoice,
    RobertaForCausalLM as HF_RobertaForCausalLM,
    PreTrainedTokenizer,
)

from hft2ane.models.roberta import (
    RobertaForSequenceClassification,
    RobertaForMaskedLM,
    RobertaForQuestionAnswering,
    RobertaForTokenClassification,
    RobertaForMultipleChoice,
    RobertaForCausalLM,
)


TEST_MAX_SEQ_LEN = 256
PSNR_THRESHOLD = 60

SEQUENCE_CLASSIFICATION_MODEL = "WillHeld/roberta-base-sst2"
MASKED_LM_MODEL = "roberta-base"
QUESTION_ANSWERING_MODEL = "deepset/roberta-base-squad2"
TOKEN_CLASSIFICATION_MODEL = "Jean-Baptiste/roberta-large-ner-english"
MULTIPLE_CHOICE_MODEL = "nmb-paperspace-hf/roberta-base-finetuned-swag"
CAUSAL_LM_MODEL = "GItaf/roberta-base-finetuned-mbti-0901"


@pytest.fixture(scope="session")
def masked_lm():
    tokenizer = AutoTokenizer.from_pretrained(MASKED_LM_MODEL)
    hf_model = HF_RobertaForMaskedLM.from_pretrained(
        MASKED_LM_MODEL, return_dict=False
    ).eval()
    ane_model = RobertaForMaskedLM.from_pretrained(
        MASKED_LM_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def sequence_classification():
    tokenizer = AutoTokenizer.from_pretrained(SEQUENCE_CLASSIFICATION_MODEL)
    hf_model = HF_RobertaForSequenceClassification.from_pretrained(
        SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    ane_model = RobertaForSequenceClassification.from_pretrained(
        SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def question_answering():
    tokenizer = AutoTokenizer.from_pretrained(QUESTION_ANSWERING_MODEL)
    hf_model = HF_RobertaForQuestionAnswering.from_pretrained(
        QUESTION_ANSWERING_MODEL, return_dict=False
    ).eval()
    ane_model = RobertaForQuestionAnswering.from_pretrained(
        QUESTION_ANSWERING_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def token_classification():
    tokenizer = AutoTokenizer.from_pretrained(TOKEN_CLASSIFICATION_MODEL)
    hf_model = HF_RobertaForTokenClassification.from_pretrained(
        TOKEN_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    ane_model = RobertaForTokenClassification.from_pretrained(
        TOKEN_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def multiple_choice():
    tokenizer = AutoTokenizer.from_pretrained(MULTIPLE_CHOICE_MODEL)
    hf_model = HF_RobertaForMultipleChoice.from_pretrained(
        MULTIPLE_CHOICE_MODEL, return_dict=False
    ).eval()
    ane_model = RobertaForMultipleChoice.from_pretrained(
        MULTIPLE_CHOICE_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def causal_lm():
    tokenizer = AutoTokenizer.from_pretrained(CAUSAL_LM_MODEL)
    hf_model = HF_RobertaForCausalLM.from_pretrained(
        CAUSAL_LM_MODEL, return_dict=False
    ).eval()
    ane_model = RobertaForCausalLM.from_pretrained(
        CAUSAL_LM_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


def _np_probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(1).numpy()


def _decode_masked(
    output_logits: torch.Tensor, tokenizer: PreTrainedTokenizer, masked_index: int
) -> str:
    predicted_index = torch.argmax(output_logits[0], dim=1)[masked_index].item()
    tokens = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    return tokenizer.convert_tokens_to_string([tokens])


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
    tokens = map(
        lambda s: tokenizer.convert_tokens_to_string([s]), tokenizer.tokenize(input_str)
    )
    return [(token, label) for token, label in zip(tokens, labels[1:-1])]


def _get_choice(output_logits: torch.Tensor, choices: list[str]) -> str:
    return choices[torch.argmax(output_logits)]


def _decode_causal_lm(output_ids: torch.Tensor, tokenizer: PreTrainedTokenizer) -> str:
    return tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)


@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        ("Hello how <mask> you doing?", " are"),
        ("Hello how are <mask> doing?", " you"),
    ],
)
def test_masked_lm(input_str, expected_output, masked_lm):
    tokenizer, hf_model, ane_model = masked_lm
    tokenized_text = tokenizer.tokenize(input_str)
    masked_index = tokenized_text.index(" <mask>")
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
            " 5,500,000",
        ),
        (
            "Which name is also used to describe the Amazon rainforest in English?",
            " Amazonia or the Amazon Jungle",
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
                (" My", "O"),
                (" name", "O"),
                (" is", "O"),
                (" Sarah", "PER"),
                (" and", "O"),
                (" I", "O"),
                (" live", "O"),
                (" in", "O"),
                (" London", "LOC"),
            ],
        ),
        (
            "My name is Clara and I live in Berkeley, California.",
            [
                (" My", "O"),
                (" name", "O"),
                (" is", "O"),
                (" Clara", "PER"),
                (" and", "O"),
                (" I", "O"),
                (" live", "O"),
                (" in", "O"),
                (" Berkeley", "LOC"),
                (",", "O"),
                (" California", "LOC"),
                (".", "LOC"),
            ],  # hmm
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


@pytest.mark.skip(reason="CausalLM needs use_cache and is_decoder support")
@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        (
            "My name is Mariama, my favorite",
            "My name is Mariama, my favorite anime series is shoujo yuara hanata",
        ),
        (
            "Once upon a time there was a wizened old crone",
            "Once upon a time there was a wizened old crone who was a bit of a jerk",
        ),
    ],
)
def test_causal_lm(input_str, expected_output, causal_lm):
    torch.manual_seed(42)
    tokenizer, hf_model, ane_model = causal_lm
    input_ids = tokenizer.encode(
        input_str, add_special_tokens=False, return_tensors="pt"
    )

    with torch.no_grad():
        hf_output_ids = hf_model.generate(
            input_ids=input_ids,
            max_length=20,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_return_sequences=1,
        )
        ane_output_ids = ane_model.generate(
            input_ids=input_ids,
            max_length=20,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_return_sequences=1,
        )

    assert ane_output_ids.shape == hf_output_ids.shape

    # peak_signal_to_noise_ratio = compute_psnr(
    #     _np_probs(hf_output_ids), _np_probs(ane_output_ids)
    # )
    # assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    hf_result = _decode_causal_lm(hf_output_ids, tokenizer)
    ane_result = _decode_causal_lm(ane_output_ids, tokenizer)
    assert ane_result == hf_result

    assert ane_result == expected_output

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from ane_transformers.testing_utils import compute_psnr
from tests.conftest import materialize_params

from hft2ane.evaluate.evaluate import (
    get_dummy_inputs,
    measure_ane_speedup_from_converted,
)
from hft2ane.models.bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
)
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers import (
    BertForMaskedLM as HF_BertForMaskedLM,
)
from transformers import (
    BertForMultipleChoice as HF_BertForMultipleChoice,
)
from transformers import (
    BertForNextSentencePrediction as HF_BertForNextSentencePrediction,
)
from transformers import (
    BertForQuestionAnswering as HF_BertForQuestionAnswering,
)
from transformers import (
    BertForSequenceClassification as HF_BertForSequenceClassification,
)
from transformers import (
    BertForTokenClassification as HF_BertForTokenClassification,
)

TEST_MAX_SEQ_LEN = 256
PSNR_THRESHOLD = 33
MIN_SPEEDUP_FACTOR = 1.5

SEQUENCE_CLASSIFICATION_MODEL = (
    "ModelTC/bert-base-uncased-sst2"  # textattack/bert-base-uncased-yelp-polarity
)
MASKED_LM_MODEL = "bert-base-uncased"
NSP_MODEL = "bert-base-uncased"
QUESTION_ANSWERING_MODEL = "csarron/bert-base-uncased-squad-v1"  # deepset/bert-base-cased-squad2
TOKEN_CLASSIFICATION_MODEL = (
    "dslim/bert-base-NER-uncased"  # dbmdz/bert-large-cased-finetuned-conll03-english
)
MULTIPLE_CHOICE_MODEL = "ehdwns1516/bert-base-uncased_SWAG"

MASKED_LM_EXAMPLES = [
    ("Hello how [MASK] you doing?", "are"),
    ("Hello how are [MASK] doing?", "you"),
]
NEXT_SENTENCE_EXAMPLES = [
    ("How old are you?", "I am 21 years old.", True),
    ("How old are you?", "Queen's University is in Kingston Ontario Canada", False),
    (
        "The sun is a huge ball of gases. It has a diameter of 1,392,000 km.",
        "It is mainly made up of hydrogen and helium gas. The surface of the Sun is known as the photosphere.",
        True,
    ),
    ("The cat sat on the mat.", "It was a nice day outside.", True),
]
SEQUENCE_CLASSIFICATION_EXAMPLES = [
    ("Today was a good day!", "positive"),
    ("This is not what I expected!", "negative"),
]
QUESTION_ANSWERING_EXAMPLES = [
    (
        "How many square kilometers of rainforest is covered in the basin?",
        "5, 500, 000",
    ),
    (
        "Which name is also used to describe the Amazon rainforest in English?",
        "amazonia or the amazon jungle",
    ),
]
TOKEN_CLASSIFICATION_EXAMPLES = [
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
]
MULTIPLE_CHOICE_EXAMPLES = [
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
]


@pytest.fixture(scope="session")
def masked_lm():
    tokenizer = AutoTokenizer.from_pretrained(MASKED_LM_MODEL)
    hf_model = materialize_params(
        HF_BertForMaskedLM.from_pretrained(MASKED_LM_MODEL, return_dict=False).eval()
    )
    ane_model = BertForMaskedLM.from_pretrained(MASKED_LM_MODEL, return_dict=False).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def next_sentence_prediction():
    tokenizer = AutoTokenizer.from_pretrained(NSP_MODEL)
    hf_model = materialize_params(
        HF_BertForNextSentencePrediction.from_pretrained(NSP_MODEL, return_dict=False).eval()
    )
    ane_model = BertForNextSentencePrediction.from_pretrained(NSP_MODEL, return_dict=False).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def sequence_classification():
    tokenizer = AutoTokenizer.from_pretrained(SEQUENCE_CLASSIFICATION_MODEL)
    hf_model = materialize_params(
        HF_BertForSequenceClassification.from_pretrained(
            SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
        ).eval()
    )
    ane_model = BertForSequenceClassification.from_pretrained(
        SEQUENCE_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def question_answering():
    tokenizer = AutoTokenizer.from_pretrained(QUESTION_ANSWERING_MODEL)
    hf_model = materialize_params(
        HF_BertForQuestionAnswering.from_pretrained(
            QUESTION_ANSWERING_MODEL, return_dict=False
        ).eval()
    )
    ane_model = BertForQuestionAnswering.from_pretrained(
        QUESTION_ANSWERING_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def token_classification():
    tokenizer = AutoTokenizer.from_pretrained(TOKEN_CLASSIFICATION_MODEL)
    hf_model = materialize_params(
        HF_BertForTokenClassification.from_pretrained(
            TOKEN_CLASSIFICATION_MODEL, return_dict=False
        ).eval()
    )
    ane_model = BertForTokenClassification.from_pretrained(
        TOKEN_CLASSIFICATION_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


@pytest.fixture(scope="session")
def multiple_choice():
    tokenizer = AutoTokenizer.from_pretrained(MULTIPLE_CHOICE_MODEL)
    hf_model = materialize_params(
        HF_BertForMultipleChoice.from_pretrained(
            MULTIPLE_CHOICE_MODEL, return_dict=False
        ).eval()
    )
    ane_model = BertForMultipleChoice.from_pretrained(
        MULTIPLE_CHOICE_MODEL, return_dict=False
    ).eval()
    return tokenizer, hf_model, ane_model


def _np_probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(1).numpy()


def get_masked_lm_inputs(tokenizer, input_str):
    tokenized_text = tokenizer.tokenize(input_str)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    return tokens_tensor, masked_index


# NOTE: bert-base-uncased does not do very well, other simple phrases
# gave bad results, bert-large does better
@pytest.mark.parametrize(
    "input_str,expected_output",
    MASKED_LM_EXAMPLES,
)
def test_masked_lm(input_str, expected_output, masked_lm):
    tokenizer, hf_model, ane_model = masked_lm

    inputs, masked_index = get_masked_lm_inputs(tokenizer, input_str)

    with torch.no_grad():
        hf_outputs = hf_model(inputs)
        ane_outputs = ane_model(inputs)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(_np_probs(hf_outputs[0]), _np_probs(ane_outputs[0]))
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    def decode_masked(
        output_logits: torch.Tensor, tokenizer: PreTrainedTokenizer, masked_index: int
    ) -> str:
        predicted_index = torch.argmax(output_logits[0], dim=1)[masked_index].item()
        return tokenizer.convert_ids_to_tokens([predicted_index])[0]

    hf_result = decode_masked(hf_outputs[0], tokenizer, masked_index)
    ane_result = decode_masked(ane_outputs[0], tokenizer, masked_index)
    assert ane_result == hf_result

    assert ane_result == expected_output


def get_nsp_inputs(tokenizer, sentence_a, sentence_b):
    return tokenizer(sentence_a, sentence_b, return_tensors="pt")


@pytest.mark.parametrize(
    "sentence_a,sentence_b,expected_output",
    NEXT_SENTENCE_EXAMPLES,
)
def test_next_sentence_prediction(
    sentence_a, sentence_b, expected_output, next_sentence_prediction
):
    tokenizer, hf_model, ane_model = next_sentence_prediction
    inputs = get_nsp_inputs(tokenizer, sentence_a, sentence_b)

    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(_np_probs(hf_outputs[0]), _np_probs(ane_outputs[0]))
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    def decode_nsp(scores: torch.Tensor) -> bool:
        probabilities = torch.softmax(scores, dim=1)
        prediction = torch.argmax(probabilities)
        # 0 = yes, 1 = no
        return prediction.item() == 0

    hf_result = decode_nsp(hf_outputs[0])
    ane_result = decode_nsp(ane_outputs[0])
    assert ane_result == hf_result

    assert ane_result == expected_output


def get_sequence_classification_inputs(tokenizer, input_str):
    return tokenizer(
        [input_str],
        return_tensors="pt",
        max_length=TEST_MAX_SEQ_LEN,
        padding="max_length",
    )


@pytest.mark.parametrize(
    "input_str,expected_output",
    SEQUENCE_CLASSIFICATION_EXAMPLES,
)
def test_sequence_classification(input_str, expected_output, sequence_classification):
    tokenizer, hf_model, ane_model = sequence_classification
    inputs = get_sequence_classification_inputs(tokenizer, input_str)

    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(_np_probs(hf_outputs[0]), _np_probs(ane_outputs[0]))
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    def get_class_index(output_logits: torch.Tensor, id2label: dict[int, str | int]) -> str | int:
        return id2label[torch.argmax(output_logits, dim=1).item()]

    hf_result = get_class_index(hf_outputs[0], hf_model.config.id2label)
    ane_result = get_class_index(ane_outputs[0], ane_model.config.id2label)
    assert ane_result == hf_result

    assert ane_result == expected_output


def get_question_answering_inputs(tokenizer, question, context):
    return tokenizer(question, context, return_tensors="pt")


QA_CONTEXT = """The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."""


# NOTE: needs a pipeline to get nicer formatted answers like on the hub page
# https://huggingface.co/distilbert-base-uncased-distilled-squad
# I guess because here we're using the decoded output of the model directly
# whereas the pipeline maybe uses the result to extract from the original text
@pytest.mark.parametrize(
    "question,expected_output",
    QUESTION_ANSWERING_EXAMPLES,
)
def test_question_answering(question, expected_output, question_answering):
    tokenizer, hf_model, ane_model = question_answering
    inputs = get_question_answering_inputs(tokenizer, question, QA_CONTEXT)

    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(_np_probs(hf_outputs[0]), _np_probs(ane_outputs[0]))
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    def decode_qa(
        inputs,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        answer_start_index = torch.argmax(start_logits)
        answer_end_index = torch.argmax(end_logits)
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        return tokenizer.decode(predict_answer_tokens)

    hf_result = decode_qa(inputs, hf_outputs[0], hf_outputs[1], tokenizer)
    ane_result = decode_qa(inputs, ane_outputs[0], ane_outputs[1], tokenizer)
    assert ane_result == hf_result

    assert ane_result == expected_output


def get_token_classification_inputs(tokenizer, input_str):
    return tokenizer(
        [input_str],
        return_tensors="pt",
    )


@pytest.mark.parametrize(
    "input_str,expected_output",
    TOKEN_CLASSIFICATION_EXAMPLES,
)
def test_token_classification(input_str, expected_output, token_classification):
    tokenizer, hf_model, ane_model = token_classification
    inputs = get_token_classification_inputs(tokenizer, input_str)

    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(_np_probs(hf_outputs[0]), _np_probs(ane_outputs[0]))
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    def get_labelled_tokens(
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

    hf_result = get_labelled_tokens(hf_outputs[0], input_str, hf_model.config.id2label, tokenizer)
    ane_result = get_labelled_tokens(
        ane_outputs[0], input_str, ane_model.config.id2label, tokenizer
    )
    assert ane_result == hf_result

    assert ane_result == expected_output


def get_multiple_choice_inputs(tokenizer, input_str, choices):
    inputs = tokenizer(
        [[input_str, choice] for choice in choices],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {k: v.unsqueeze(0) for k, v in inputs.items()}


@pytest.mark.parametrize(
    "input_str,choices,expected_output",
    MULTIPLE_CHOICE_EXAMPLES,
)
def test_multiple_choice(input_str, choices, expected_output, multiple_choice):
    tokenizer, hf_model, ane_model = multiple_choice
    inputs = get_multiple_choice_inputs(tokenizer, input_str, choices)

    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        ane_outputs = ane_model(**inputs)

    assert ane_outputs[0].shape == hf_outputs[0].shape

    peak_signal_to_noise_ratio = compute_psnr(_np_probs(hf_outputs[0]), _np_probs(ane_outputs[0]))
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    def get_choice(output_logits: torch.Tensor, choices: list[str]) -> str:
        return choices[torch.argmax(output_logits)]

    hf_result = get_choice(hf_outputs[0], choices)
    ane_result = get_choice(ane_outputs[0], choices)
    assert ane_result == hf_result

    assert ane_result == expected_output


@pytest.mark.convert
@pytest.mark.parametrize(
    "model_name, hf_model_cls, get_inputs, example",
    [
        pytest.param(
            SEQUENCE_CLASSIFICATION_MODEL,
            HF_BertForSequenceClassification,
            get_sequence_classification_inputs,
            SEQUENCE_CLASSIFICATION_EXAMPLES[0],
            id="sequence_classification",
        ),
        pytest.param(
            MASKED_LM_MODEL,
            HF_BertForMaskedLM,
            get_masked_lm_inputs,
            MASKED_LM_EXAMPLES[0],
            id="masked_lm",
        ),
        pytest.param(
            NSP_MODEL,
            HF_BertForNextSentencePrediction,
            get_nsp_inputs,
            NEXT_SENTENCE_EXAMPLES[0],
            id="next_sentence_prediction",
        ),
        pytest.param(
            QUESTION_ANSWERING_MODEL,
            HF_BertForQuestionAnswering,
            get_question_answering_inputs,
            QUESTION_ANSWERING_EXAMPLES[0],
            id="question_answering",
        ),
        pytest.param(
            TOKEN_CLASSIFICATION_MODEL,
            HF_BertForTokenClassification,
            get_token_classification_inputs,
            TOKEN_CLASSIFICATION_EXAMPLES[0],
            id="token_classification",
        ),
        pytest.param(
            MULTIPLE_CHOICE_MODEL,
            HF_BertForMultipleChoice,
            get_multiple_choice_inputs,
            MULTIPLE_CHOICE_EXAMPLES[0],
            id="multiple_choice",
        ),
    ],
)
def test_export(model_name, hf_model_cls, get_inputs, example, tmp_path_factory):
    from hft2ane.convert.convert import to_coreml

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline = materialize_params(
        hf_model_cls.from_pretrained(model_name, return_dict=False).eval()
    )

    out_path = tmp_path_factory.mktemp("test_bert") / f"{hf_model_cls.__name__}.mlpackage"
    converted = to_coreml(model_name, hf_model_cls, out_path)

    *args, expected = example
    inputs = get_inputs(tokenizer, *args)
    if isinstance(inputs, tuple):
        # masked LM
        inputs = inputs[0]

    # TODO: currently our export/convert uses model.dummy_inputs to get the shape
    # however actual model usage requires more, e.g. `get_inputs` for sequence classification
    # returns a mapping with: `input_ids`, `token_type_ids`, `attention_mask`
    # (this is also what prevents MultipleChoice from converting currently)
    # TODO: see `tokenizer.model_input_names` for the names of the inputs
    np_inputs = {k: v.numpy().astype(np.int32) for k, v in inputs.items()}
    with torch.no_grad():
        baseline_outputs = baseline(**inputs) if isinstance(inputs, Mapping) else baseline(inputs)
    converted_outputs = list(converted.predict(np_inputs).values())

    peak_signal_to_noise_ratio: float = compute_psnr(
        baseline_outputs[0].softmax(1).numpy(),
        torch.from_numpy(converted_outputs[0]).softmax(1).numpy(),
    )
    assert peak_signal_to_noise_ratio > PSNR_THRESHOLD

    speedup = measure_ane_speedup_from_converted(converted, get_dummy_inputs(baseline))
    assert speedup > MIN_SPEEDUP_FACTOR

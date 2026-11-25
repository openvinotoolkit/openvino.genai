import pytest
from PIL import Image

from whowhatbench.inputs_preprocessors import MODEL_TYPE_TO_CLS_MAPPING
from whowhatbench.inputs_preprocessors.paddleocr_vl import PaddleOCRVLInputsPreprocessor


class _FakeProcessor:
    def __init__(self):
        self.conversations = []
        self.calls = []

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt and not tokenize
        self.conversations.append([dict(m) for m in conversation])
        return "prompt"

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        assert "videos" not in kwargs
        self.calls.append({"images": images, "text": text, "return_tensors": return_tensors})
        return {"input_ids": text}


def test_paddleocr_vl_registered():
    assert MODEL_TYPE_TO_CLS_MAPPING["paddleocr_vl"] is PaddleOCRVLInputsPreprocessor


def test_paddleocr_vl_question_answering():
    image = Image.new("RGB", (32, 32))
    processor = _FakeProcessor()
    PaddleOCRVLInputsPreprocessor().preprocess_inputs("OCR:", image, processor)

    content = processor.conversations[0][0]["content"]
    assert processor.conversations[0][0]["role"] == "user"
    assert content == [{"type": "image", "image": image}, {"type": "text", "text": "OCR:"}]
    assert processor.calls[0]["images"] == [image]
    assert processor.calls[0]["return_tensors"] == "pt"


def test_paddleocr_vl_text_only():
    processor = _FakeProcessor()
    PaddleOCRVLInputsPreprocessor().preprocess_inputs("Hi", None, processor)
    assert processor.calls[0]["images"] is None


def test_paddleocr_vl_chat_accumulates_history_and_images():
    first, second = Image.new("RGB", (32, 32)), Image.new("RGB", (16, 16))
    processor = _FakeProcessor()
    preprocessor = PaddleOCRVLInputsPreprocessor(chat_mode=True)
    preprocessor.preprocess_inputs("OCR:", first, processor)
    preprocessor.update_chat_history_with_answer("text")
    preprocessor.preprocess_inputs("Table Recognition:", second, processor)

    roles = [m["role"] for m in processor.conversations[-1]]
    assert roles == ["user", "assistant", "user"]
    assert processor.calls[-1]["images"] == [first, second]


@pytest.mark.parametrize("kwargs", [{"video": [object()]}, {"audio": object()}])
def test_paddleocr_vl_rejects_unsupported_modalities(kwargs):
    with pytest.raises(ValueError):
        PaddleOCRVLInputsPreprocessor().preprocess_inputs("OCR:", None, _FakeProcessor(), **kwargs)

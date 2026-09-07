"""Answer surface form is adapter-owned: default identity, Gemma capitalizes."""

from tests.stubs import CountingModel
from vlmflowprobe.adapters.registry import adapter_class
from vlmflowprobe.adapters.stub import StubAdapter


def test_default_is_strip_only():
    adapter = StubAdapter(model=CountingModel(d_model=4))
    assert adapter.format_answer(" blue ") == "blue"
    assert adapter.answer_prefix == " "


def test_gemma_capitalizes_first_letter_only():
    cls = adapter_class("hf-gemma3")
    assert cls.format_answer(None, "blue") == "Blue"
    assert cls.format_answer(None, " square ") == "Square"
    assert cls.format_answer(None, "") == ""

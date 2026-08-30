from __future__ import annotations

from types import SimpleNamespace

from llm.client import VLLMClient, VLLMClientError


class FakeCompletions:
    def __init__(self) -> None:
        self.max_tokens = None

    def create(self, **kwargs):
        self.max_tokens = kwargs["max_tokens"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}')
                )
            ],
            usage=None,
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_client_reduces_output_budget_to_fit_context_window() -> None:
    fake = FakeOpenAI()
    client = VLLMClient(context_window=512, client=fake)

    result = client.chat(
        [{"role": "user", "content": "x" * 600}],
        max_tokens=256,
    )

    assert result["content"] == '{"ok": true}'
    assert fake.completions.max_tokens == 148


def test_client_rejects_prompt_that_cannot_fit_context_window() -> None:
    client = VLLMClient(context_window=512, client=FakeOpenAI())

    try:
        client.chat([{"role": "user", "content": "x" * 1000}], max_tokens=256)
    except VLLMClientError as exc:
        assert "Prompt is too large" in str(exc)
    else:
        raise AssertionError("expected oversized prompt to be rejected")

import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().with_name("nemotron_interactive_forward.py")
SPEC = importlib.util.spec_from_file_location("nemotron_interactive_forward", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        del tokenize
        token_ids = []
        for message in messages:
            role = message["role"]
            token_ids.append(1000 + len(role))
            token_ids.extend(ord(ch) for ch in message["content"])
            token_ids.append(0)
        if add_generation_prompt:
            token_ids.append(9999)
        return token_ids


class InteractiveForwardTextTest(unittest.TestCase):
    def test_extract_visible_assistant_text_strips_scratchpad(self):
        raw = (
            "The user wants another joke.\n"
            "</think>\n"
            "Sure thing.\n"
        )
        self.assertEqual(MODULE.extract_visible_assistant_text(raw), "Sure thing.")

    def test_extract_visible_assistant_text_preserves_plain_text(self):
        raw = "Because they make up everything!"
        self.assertEqual(MODULE.extract_visible_assistant_text(raw), raw)

    def test_build_turn_prompt_token_ids_falls_back_to_rendered_turn_on_history_mismatch(self):
        tokenizer = FakeTokenizer()
        history_messages = [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Sure thing."},
        ]
        rendered_turn = MODULE.render_chat_token_ids(
            tokenizer,
            history_messages + [{"role": "user", "content": "Another one."}],
            add_generation_prompt=True,
        )
        prompt_token_ids = MODULE.build_turn_prompt_token_ids(
            tokenizer,
            history_messages,
            exact_history_token_ids=[1, 2, 3],
            user_text="Another one.",
        )
        self.assertEqual(prompt_token_ids, rendered_turn)


if __name__ == "__main__":
    unittest.main()

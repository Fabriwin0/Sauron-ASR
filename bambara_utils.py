from typing import List

from tokenizers import AddedToken
from transformers import WhisperTokenizer, WhisperProcessor
import transformers.models.whisper.tokenization_whisper as whisper_tokenization
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE, TASK_IDS

CUSTOM_TO_LANGUAGE_CODE = {**TO_LANGUAGE_CODE, "bambara": "bm"}

# Note: We update the whisper tokenizer constants. Not ideal but at least it works
whisper_tokenization.TO_LANGUAGE_CODE.update(CUSTOM_TO_LANGUAGE_CODE)


class BambaraWhisperTokenizer(WhisperTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_tokens(AddedToken(content="<|bm|>", lstrip=False, rstrip=False, normalized=False, special=True))

    @property
    def prefix_tokens(self) -> List[int]:
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in CUSTOM_TO_LANGUAGE_CODE:
                language_id = CUSTOM_TO_LANGUAGE_CODE[self.language]
            elif self.language in CUSTOM_TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(CUSTOM_TO_LANGUAGE_CODE.values()) if is_language_code else list(CUSTOM_TO_LANGUAGE_CODE.keys())}."
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            bos_sequence.append(self.convert_tokens_to_ids(f"<|{language_id}|>"))
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

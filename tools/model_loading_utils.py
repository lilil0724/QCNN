# SPDX-License-Identifier: Apache-2.0
"""
Small, self-contained model-loading helpers, extracted from QuantizedASR's
`qasr/model/bitnet_convert.py` (not modified logic, just relocated - this repo doesn't
carry the rest of that file's onebitllms-based BitNet conversion machinery, since
compare_qat_weights.py only ever needed these two functions).
"""
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
)


def _resolve_model_class(model_id: str, config):
    """Pick the right ``from_pretrained`` class for a (possibly LALM) checkpoint."""
    try:
        from transformers import (
            VoxtralForConditionalGeneration,
            Qwen2AudioForConditionalGeneration,
            Qwen2_5OmniForConditionalGeneration,
        )
    except ImportError:
        VoxtralForConditionalGeneration = None
        Qwen2AudioForConditionalGeneration = None
        Qwen2_5OmniForConditionalGeneration = None

    if 'Voxtral' in model_id and VoxtralForConditionalGeneration is not None:
        return VoxtralForConditionalGeneration
    if 'Qwen2.5-Omni' in model_id and Qwen2_5OmniForConditionalGeneration is not None:
        return Qwen2_5OmniForConditionalGeneration
    if ('Qwen2-Audio' in model_id or 'q2a' in model_id) and Qwen2AudioForConditionalGeneration is not None:
        return Qwen2AudioForConditionalGeneration
    if 'lite-whisper' in model_id:
        return AutoModel
    if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING:
        return AutoModelForSpeechSeq2Seq
    return AutoModelForCausalLM


def load_config_with_remote_code_fallback(model_id: str, revision: str = None,
                                          trust_remote_code: bool = True):
    """AutoConfig.from_pretrained, retrying with trust_remote_code=False if the repo's
    auto_map points at a dynamic-module file that no longer exists.

    Observed with microsoft/bitnet-b1.58-2B-4T: transformers has native BitNet support
    (transformers/models/bitnet/), but the repo's config.json still carries an auto_map
    from before that landed, pointing at a configuration_bitnet.py the repo no longer
    ships - trust_remote_code=True forces the (now-404) dynamic-module path instead of
    the native one. Returns (config, the trust_remote_code value actually used).
    """
    try:
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code)
        return config, trust_remote_code
    except OSError as exc:
        if trust_remote_code and 'does not appear to have a file named' in str(exc):
            print(f'AutoConfig with trust_remote_code=True failed ({exc}); '
                  f'retrying with trust_remote_code=False ...')
            config = AutoConfig.from_pretrained(
                model_id, revision=revision, trust_remote_code=False)
            return config, False
        raise

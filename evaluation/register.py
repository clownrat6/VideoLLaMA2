from transformers import AutoConfig

import sys
sys.path.append('./')
from videollama2 import model_init as videollama2_model_init, mm_infer as videollama2_mm_infer


def INFERENCES(model_path):
    config = AutoConfig.from_pretrained(model_path)

    # judge model type
    model_type = config.model_type
    if 'videollama2' in model_type.lower():
        return videollama2_model_init, videollama2_mm_infer
    else:
        raise NotImplementedError(f"Model path {model_path} not recognized.")

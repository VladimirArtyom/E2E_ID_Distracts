import math

import torch
from fairseq import options, utils
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder
)



@register_model('dg')
class DistractorsModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args


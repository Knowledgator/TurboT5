from .heads.t5_heads import (T5ForConditionalGeneration, T5ForQuestionAnswering, 
                                    T5ForSequenceClassification, T5ForTokenClassification)

from .heads.encoder_heads import (T5EncoderForQuestionAnswering, T5EncoderForTextClassification, T5ForTokenClassification)

from .model.config import T5Config
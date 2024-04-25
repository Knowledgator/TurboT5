from src.heads.t5_heads import T5ForConditionalGeneration

from transformers import T5Tokenizer


class TestModels:
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model_checkpoint = "google/flan-t5-small"
    input_text = "translate English to German: How old are you?"

    def test_full_attention_torch_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'full', 
                                                    use_triton=False).to('cuda')
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])

    def test_full_attention_triton_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'full', 
                                                    use_triton=True).to('cuda')
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])

    def test_local_attention_torch_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'local', 
                                                    use_triton=False).to('cuda')
        
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])

    def test_local_attention_triton_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'local', 
                                                    use_triton=True).to('cuda')
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])

    def test_block_attention_torch_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'block', 
                                                    use_triton=False).to('cuda')
        
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])

    def test_block_attention_triton_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'block', 
                                                    use_triton=True).to('cuda')
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])

    def test_transient_attention_torch_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'transient-global', 
                                                    use_triton=False).to('cuda')
        
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])

    def test_transient_attention_triton_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint,
                                                    attention_type = 'transient-global', 
                                                    use_triton=True).to('cuda')
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").input_ids.to('cuda')

        outputs = model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])





    

from transformers import MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from transformers import PreTrainedModel, MBartConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch
import os

import time
import functools


def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f"[{func.__name__}] Выполнено за {elapsed:.3f} сек.")
        return result
    return wrapper


class My_MBart(PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: MBartConfig, model_name=None):
        super().__init__(config)

        mbart_name = config.mbart_name
        # Основная модель
        self.mbart = MBartForConditionalGeneration.from_pretrained(mbart_name, config=config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.mbart.model.shared.num_embeddings)))

        vocab_size = self.mbart.model.shared.num_embeddings
        # self.lm_head = self.mbart.lm_head

        # ───────── кластерная часть ───────
        self.use_meta = getattr(config, "use_cluster", True)
        if self.use_meta:
            self.injection_type = config.injection_type         # "0" | "1" | "2" | "3"
            self.proj = torch.nn.Linear(config.embed_size,
                                        config.d_model)
            if self.injection_type == "3":
                # добавляем bias к логитам
                self.meta_bias = torch.nn.Linear(config.d_model,
                                                 vocab_size,
                                                 bias=False)
        self.post_init()

    # def get_output_embeddings(self):
    #     return self.mbart.get_output_embeddings()          # lm_head

    # def set_output_embeddings(self, new_emb):
    #     return self.mbart.set_output_embeddings(new_emb)

    def get_encoder(self):
        return self.mbart.get_encoder()

    def get_decoder(self):
        return self.mbart.get_decoder()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        meta_embs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=True,
        *args, **model_kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Автосдвиг декодера, если есть метки
        if labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        # print(input_ids)
        # Прогон основного текста через основной энкодер
        if "encoder_outputs" not in model_kwargs:
            encoder_outputs = self.mbart.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        else:
            encoder_outputs = model_kwargs["encoder_outputs"]

        if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        enc_hid = encoder_outputs.last_hidden_state

        # ───── Кластер‑инжект на стороне энкодера ─────
        if self.use_meta and meta_embs is not None:
            meta_vec = self.proj(meta_embs)              # [B, D]
            meta_vec_u = meta_vec.unsqueeze(1)           # [B, 1, D]

            if self.injection_type == "0":               # add‑to‑encoder
                enc_hid = enc_hid + meta_vec_u

            elif self.injection_type == "1":             # replace CLS‑token
                enc_hid = enc_hid.clone()                # не ломаем grad
                enc_hid[:, 0, :] = meta_vec              # [CLS] ← cluster

        # final encoder output
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_hid)

        decoder_kwargs = {
            "attention_mask": decoder_attention_mask,
            "encoder_hidden_states": enc_hid,
            "encoder_attention_mask": attention_mask,
            "return_dict": return_dict,
        }

        if self.use_meta and self.injection_type == "2":
            # add‑to‑decoder‑embeddings
            embeds = self.mbart.model.decoder.embed_tokens(decoder_input_ids)
            embeds = embeds + meta_vec_u                 # broadcast по seq_len
            decoder_kwargs["inputs_embeds"] = embeds
            decoder_input_ids = None                     # mutual exclus.

        decoder_outputs = self.mbart.model.decoder(
            input_ids=decoder_input_ids,
            **decoder_kwargs
        )

        # ───── LM‑логиты ─────
        lm_logits = self.mbart.lm_head(decoder_outputs.last_hidden_state)
        lm_logits = lm_logits + self.final_logits_bias

        if self.use_meta and self.injection_type == "3":
            bias = self.meta_bias(meta_vec)              # [B, vocab]
            lm_logits = lm_logits + bias.unsqueeze(1)    # broadcast

        # ───── Loss ─────
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        if not return_dict:
            return (lm_logits,) if loss is None else (loss, lm_logits)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=enc_hid,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # @log_time
    # def generate(
    #     self,
    #     input_ids,
    #     attention_mask=None,
    #     meta_embs=None,
    #     **gen_kwargs,
    # ):
    #     """
    #     Вызовем родительский `generate`, передавая `meta_embs`
    #     через `model_kwargs` – они будут
    #     автоматически прокинуты в `forward` на каждом шаге.
    #     """

    #     model_kwargs = {
    #         "attention_mask": attention_mask,
    #         "meta_embs": meta_embs,
    #     }
    #     # print(input_ids)
    #     return super().generate(
    #         inputs=input_ids,
    #         **model_kwargs,
    #         **gen_kwargs,
    #     )

    # @log_time
    def generate(
        self,
        input_ids,
        attention_mask=None,
        meta_embs=None,
        max_length=30,
        decoder_start_token_id=None,
        eos_token_id=None,
    ):
        """
        Полностью ручная реализация генерации с учетом всех вариантов injection_type.
        Поддерживает greedy decoding (жадную генерацию).
        """
        decoder_start_token_id = decoder_start_token_id or self.config.decoder_start_token_id
        eos_token_id = eos_token_id or self.config.eos_token_id
        batch_size = input_ids.size(0)

        with torch.no_grad():
            # === Шаг 1: Получаем encoder hidden states ===
            encoder_outputs = self.mbart.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            encoder_hidden = encoder_outputs.last_hidden_state  # [B, T, D]

            # === Шаг 2: Обработка meta_embs ===
            if self.use_meta and meta_embs is not None:
                meta_vec = self.proj(meta_embs)  # [B, D]
                meta_vec_u = meta_vec.unsqueeze(1)  # [B, 1, D]

                if self.injection_type == "0":
                    encoder_hidden = encoder_hidden + meta_vec_u

                elif self.injection_type == "1":
                    encoder_hidden = encoder_hidden.clone()
                    encoder_hidden[:, 0, :] = meta_vec

            # финальный encoder_outputs
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

            # === Шаг 3: Инициализируем декодер ===
            generated = torch.full(
                (batch_size, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            for step in range(max_length - 1):
                if self.use_meta and self.injection_type == "2":
                    # Добавляем кластер к embeddings каждого токена
                    decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(generated)
                    decoder_inputs_embeds = decoder_inputs_embeds + meta_vec_u
                    decoder_outputs = self.mbart.model.decoder(
                        inputs_embeds=decoder_inputs_embeds,
                        encoder_hidden_states=encoder_hidden,
                        encoder_attention_mask=attention_mask,
                        return_dict=True,
                    )
                else:
                    decoder_outputs = self.mbart.model.decoder(
                        input_ids=generated,
                        encoder_hidden_states=encoder_hidden,
                        encoder_attention_mask=attention_mask,
                        return_dict=True,
                    )

                last_hidden = decoder_outputs.last_hidden_state[:, -1, :]  # [B, D]
                logits = self.mbart.lm_head(last_hidden) + self.final_logits_bias  # [B, V]

                if self.use_meta and self.injection_type == "3":
                    # Добавляем кластерный bias к логитам
                    cluster_bias = self.meta_bias(meta_vec)  # [B, V]
                    logits = logits + cluster_bias

                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)  # [B, 1]

                # Прекращаем генерацию, если достигнут EOS
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, self.config.pad_token_id), next_token)
                generated = torch.cat([generated, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == eos_token_id)

                if finished.all():
                    break

            return generated

    @classmethod
    def from_pretrained_combined(cls,
                                 mbart_name,
                                 embed_size=1536,
                                 use_meta=True,
                                 injection_type: str = "0",
                                 **kwargs):
        config = MBartConfig.from_pretrained(mbart_name)
        config.mbart_name = mbart_name
        config.embed_size = embed_size
        config.use_cluster = use_meta
        config.injection_type = injection_type
        return cls(config, **kwargs)

    def save_pretrained(self, save_directory, *args, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # Сохраняем mbart и meta_encoder
        self.mbart.save_pretrained(os.path.join(save_directory, "mbart"))
        if self.use_meta:
            # Сохраняем линейные слои и веса
            if self.injection_type == "3":
                torch.save({
                    "proj": self.proj.state_dict(),
                    "meta_bias": self.meta_bias.state_dict(),
                }, os.path.join(save_directory, "proj_weights.bin"))
            else:
                torch.save({
                    "proj": self.proj.state_dict()
                }, os.path.join(save_directory, "proj_weights.bin"))

        # Обновляем конфиг
        self.config.mbart_name = os.path.join(save_directory, "mbart")
        self.config.meta_hidden_size = getattr(self, 'proj', torch.nn.Linear(1, 1)).in_features if self.use_meta else None
        self.config.use_meta = self.use_meta

        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory, *args, **kwargs):
        config = MBartConfig.from_pretrained(load_directory)
        model = cls(config)

        if getattr(config, "use_meta", True):
            weights = torch.load(os.path.join(load_directory, "proj_weights.bin"), map_location="cpu")
            model.proj.load_state_dict(weights["proj"])
            if config.injection_type == "3":
                model.meta_bias.load_state_dict(weights["meta_bias"])

        return model

    def _reorder_cache(self, past, beam_idx):
        """
        Перестраивает past_key_values в соответствии с выбранными beam‑ами.
        Обязателен для работы beam search.

        Parameters:
        - past: tuple из past_key_values [(k,v), ...]
        - beam_idx: индекс выбранных beam-ов, shape [num_beams]

        Возвращает перестроенный past.
        """
        return self.mbart._reorder_cache(past, beam_idx)

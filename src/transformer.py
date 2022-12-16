import math
import torch
import numpy as np
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder


class Transformer(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        vocab_size=None,
        logits_mode=1,
    ):
        super().__init__()

        num_heads_upsample = num_heads
        config = AutoConfig.from_pretrained('bert-base-uncased')
        # config.max_position_embeddings = 4096

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditional_gen = False  # TODO

        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.logits_mode = logits_mode

        self.word_embedding = torch.nn.Embedding(vocab_size, self.in_channels)
        self.lm_head = torch.nn.Linear(self.in_channels, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = model_channels * 4
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            torch.nn.Linear(time_embed_dim, config.hidden_size)
        )

        if self.num_classes is not None:
            self.label_emb = torch.nn.Embedding(num_classes, time_embed_dim)
        
        self.input_up_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, config.hidden_size),
            torch.nn.Tanh(), 
            torch.nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.input_transformers = BertEncoder(config)

        # print('max_position_embeddings', config.max_position_embeddings)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(0.1)

        self.output_down_proj = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Tanh(), 
            torch.nn.Linear(config.hidden_size, out_channels)
        )

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, timesteps, y=None, src_ids=None, src_mask=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.conditional_gen:
            src_emb = self.encoder_emb(src_ids)
            encoder_hidden_states = self.encoder(src_emb).last_hidden_state
            encoder_attention_mask = src_mask.unsqueeze(1).unsqueeze(1)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        # print('?', self.position_ids.size(), position_ids.size(), x.size())
        # print(self.position_embeddings(position_ids).size(), emb_x.size(), emb.unsqueeze(1).expand(-1, seq_length, -1).size())
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        if self.conditional_gen:
            input_trans_hidden_states = self.input_transformers(
                emb_inputs, encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask).last_hidden_state
        else:
            input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SiLU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

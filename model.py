from transformers import WavLMModel, Wav2Vec2FeatureExtractor, WavLMPreTrainedModel
from transformers import RobertaTokenizer, RobertaModel, RobertaPreTrainedModel
import torch.nn as nn
from typing import Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.distributed as dist
from accelerate.utils.operations import _gpu_gather_object

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose b*head*dim*len
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
class MultiHeadCrossAttention(nn.Module):

    def __init__(self, encode_dim, decode_dim, d_model, n_head, rate=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(decode_dim, d_model)
        self.w_k = nn.Linear(encode_dim, d_model)
        self.w_v = nn.Linear(encode_dim, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(rate)

    def forward(self, q, k, v, mask=None):
        residual = q
        # print(q.size(), k.size(), v.size())
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        out = self.dropout(out)
        if out.size(-1) != residual.size(-1):
            out = self.ln(out)
        else:
            out = self.ln(out + residual)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
    
class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atten_weight = nn.Parameter(torch.Tensor(hidden_dim, 1), requires_grad=True)
        # self.bais_weight = nn.Parameter(torch.zeros(time_setps), requires_grad=True)
        nn.init.uniform_(self.atten_weight)

    def compute_mask(self, inputs, mask):
        # mask = mask.unsqueeze(0)
        new_attn_mask = torch.zeros_like(mask, dtype=inputs.dtype)
        new_attn_mask.masked_fill_(mask, float("-inf")) #mask是True

        return new_attn_mask

    def forward(self, inputs, mask=None):
        
        eij = torch.matmul(inputs, self.atten_weight).squeeze(-1)
        
        eij = torch.tanh(eij)

        if mask is not None:
            mask = ~mask
            tmask = self.compute_mask(inputs, mask)
            # print(tmask)
            a = torch.softmax(eij+tmask, dim=1).unsqueeze(-1)
        else:
            a = torch.softmax(eij, dim=1).unsqueeze(-1)

        weighted_output = inputs * a

        return weighted_output.sum(dim=1)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioPipline(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wavlm = WavLMModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.wavlm.freeze_feature_extractor()

        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.post_init()

    def forward(self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        # hidden_states[~padding_mask] = 0.0
        # hidden_states = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        # # hidden_states = self.attention(hidden_states, padding_mask)

        # # hidden_states = torch.mean(hidden_states, dim=1) #聚合有问题
        # hidden_states = self.mlp(hidden_states)#whether to use activation function

        return hidden_states, padding_mask
    
class VideoPipline(nn.Module):
    def __init__(self, hidden_num, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.mlp = Projection(hidden_num, 512)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        ):
        
        input_ids = self.dropout(input_ids)
        input_ids = input_ids.sum(dim=1) / attention_mask.sum(dim=1).view(-1, 1)
        #attention
        
        sequence_output = self.mlp(input_ids)
        
        return sequence_output 

    
class MultiTaskModel(WavLMPreTrainedModel):
    def __init__(self, config, n_layers=2, d_model=512, rate=0.1, alpha=0.1, belta=0.1):
        super().__init__(config)

        self.wavlm = WavLMModel(config)
        self.dropout1 = nn.Dropout(config.final_dropout)
        self.dropout2 = nn.Dropout(config.final_dropout)
        self.dropout3 = nn.Dropout(config.final_dropout)
        self.wavlm.freeze_feature_extractor()

        #TODO 模态融合
        self.a2v = nn.ModuleList([MultiHeadCrossAttention(encode_dim=768, decode_dim=1024, d_model=1024, n_head=4, rate=rate) for _ in range(n_layers)])
        self.v2a = nn.ModuleList([MultiHeadCrossAttention(encode_dim=1024, decode_dim=768, d_model=768, n_head=4, rate=rate) for _ in range(n_layers)])


        self.audio_project = Projection(d_in=1024, d_out=512)
        self.video_project = Projection(d_in=768, d_out=512)
        self.a2v_head = nn.Linear(1024, 2)
        self.v2a_head = nn.Linear(768, 2)
        self.alpha = alpha * 0.1
        self.belta = belta * 0.1


        self.cls_head = nn.Linear(1024+768, 4)

        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.post_init()

    def forward_wavlm(self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout1(hidden_states)

        padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        return hidden_states, padding_mask
    
    def forward_a2v(self, q, k, v, tgt_mask, src_mask):
        for layer in self.a2v:
            q = layer(q, k, v, src_mask.unsqueeze(1).unsqueeze(1))

        q[~tgt_mask] = 0.0

        audio_out = q.sum(dim=1) / tgt_mask.sum(dim=1).view(-1, 1)

        return audio_out

    def forward_v2a(self, q, k, v, tgt_mask, src_mask):
        for layer in self.v2a:
            q = layer(q, k, v, src_mask.unsqueeze(1).unsqueeze(1))

        q[~tgt_mask] = 0.0

        video_out = q.sum(dim=1) / tgt_mask.sum(dim=1).view(-1, 1)

        return video_out

    def forward(self,
        input_values: Optional[torch.Tensor],
        vinput_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        vattention_mask: Optional[torch.Tensor] = None,
        emo_label: Optional[torch.Tensor] = None,
        gen_label: Optional[torch.Tensor] = None,
        spk_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        audio_hidden_state, audio_mask = self.forward_wavlm(input_values=input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        vinput_values = self.dropout2(vinput_values)
        
        audio_hidden_state[~audio_mask] = 0.0
        vinput_values[~vattention_mask] = 0.0 


        #################################Task 1########################################
        audio_avc = audio_hidden_state.sum(dim=1) / audio_mask.sum(dim=1).view(-1, 1)
        video_avc = vinput_values.sum(dim=1) / vattention_mask.sum(dim=1).view(-1, 1)

        video_avc = self.video_project(video_avc)
        audio_avc = self.audio_project(audio_avc)

        audio_avc = audio_avc / torch.norm(audio_avc, dim=-1, keepdim=True)
        video_avc = video_avc / torch.norm(video_avc, dim=-1, keepdim=True)

        audio_avc_all = concat_all_gather(audio_avc)
        video_avc_all = concat_all_gather(video_avc)

        sim_a2v = audio_avc @ video_avc_all.T / self.temp
        sim_v2a = video_avc @ audio_avc_all.T / self.temp

        emo_label = emo_label.view(-1, 1)
        spk_label = spk_label.view(-1, 1)
        gen_label = gen_label.view(-1, 1)
        
        emo_label_all = concat_all_gather(emo_label)

        pos_idx = torch.eq(emo_label, emo_label_all.t()).float()

        sim_targets = pos_idx

        loss_a2v = -torch.sum(F.log_softmax(sim_a2v, dim=1)*sim_targets, dim=1).mean()
        loss_v2a = -torch.sum(F.log_softmax(sim_v2a, dim=1)*sim_targets, dim=1).mean()

        #TODO KL_Loss
        loss_avc = (loss_a2v + loss_v2a) / 2

        # #################################Task 2########################################
        # #Audio is Q,
        vinput_values_all = concat_all_gather(vinput_values)
        vattention_mask_all = concat_all_gather(vattention_mask)
        audio_hidden_state_all = concat_all_gather(audio_hidden_state)
        audio_mask_all = concat_all_gather(audio_mask)

        with torch.no_grad():
            lab_mask = torch.eq(emo_label, emo_label_all.t())
            sim_a2v.masked_fill_(lab_mask, -10000)
            sim_v2a.masked_fill_(lab_mask, -10000)

            weights_a2v = F.softmax(sim_a2v, dim=1)
            weights_v2a = F.softmax(sim_v2a, dim=1)

        bs = input_values.size(0)

        video_embeds_neg = []
        video_atts_neg = []
        
        # # select a negative video for each audio
        for b in range(bs):
            # neg_idx = torch.multinomial(weights_a2v[b], 1).item()
            neg_idx = torch.multinomial(weights_a2v[b], 1)
            video_embeds_neg.append(vinput_values_all[neg_idx.item()])
            video_atts_neg.append(vattention_mask_all[neg_idx.item()])

        video_embeds_neg = torch.stack(video_embeds_neg, dim=0)
        video_atts_neg = torch.stack(video_atts_neg, dim=0)

        audio_embeds_neg = []
        audio_atts_neg = []

        # # select a negative audio for each video
        for b in range(bs):
            # neg_idx = torch.multinomial(weights_v2a[b], 1).item()
            neg_idx = torch.multinomial(weights_v2a[b], 1)
            audio_embeds_neg.append(audio_hidden_state_all[neg_idx.item()])
            audio_atts_neg.append(audio_mask_all[neg_idx.item()])

        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)
        audio_atts_neg = torch.stack(audio_atts_neg, dim=0)

        pos_audio_out = self.forward_a2v(torch.cat([audio_hidden_state, audio_hidden_state], dim=0),
                                         torch.cat([vinput_values, video_embeds_neg], dim=0),
                                         torch.cat([vinput_values, video_embeds_neg], dim=0),
                                         torch.cat([audio_mask, audio_mask], dim=0),
                                         torch.cat([vattention_mask, video_atts_neg], dim=0)
                                         )

        audio_out = self.forward_a2v(audio_hidden_state, vinput_values, vinput_values, audio_mask, vattention_mask)
        
        pos_video_out = self.forward_v2a(torch.cat([vinput_values, vinput_values], dim=0),
                                         torch.cat([audio_hidden_state, audio_embeds_neg], dim=0), 
                                         torch.cat([audio_hidden_state, audio_embeds_neg], dim=0),
                                         torch.cat([vattention_mask, vattention_mask], dim=0),
                                         torch.cat([audio_mask, audio_atts_neg], dim=0),
                                         )

        video_out = self.forward_v2a(vinput_values, audio_hidden_state, audio_hidden_state, vattention_mask, audio_mask)

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(bs,dtype=torch.long)],
                               dim=0).to(input_values.device)


        a2v_output = self.a2v_head(pos_audio_out)
        v2a_output = self.v2a_head(pos_video_out)

        loss_itm_a2v = F.cross_entropy(a2v_output, itm_labels)
        loss_itm_v2a = F.cross_entropy(v2a_output, itm_labels)

        loss_itm = (loss_itm_a2v + loss_itm_v2a) / 2
        #################################Task 3########################################

        cls_embedding = torch.cat([audio_out, video_out], dim=1)
        cls_embedding = self.dropout3(cls_embedding)
        cls_output = self.cls_head(cls_embedding)

        return cls_output, loss_avc, loss_itm  

if __name__ == "__main__":
    
    pass
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from transformers import T5ForConditionalGeneration, LlamaForCausalLM, Qwen2ForCausalLM
import PIL
import torchvision.transforms as transforms
from .inject_data import *
from time import time
import logging

class Qwen2DATA(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

    def wrap_model(self):
        if self.config.nomlp:
            logging.info('nomlp')
            inject_trainable_data(self, target_replace_module=[
                              "Qwen2SdpaAttention", "Qwen2Attention", "Qwen2FlashAttention2"], r=self.config.data_rank1, r2=self.config.data_rank2)
        else:
            inject_trainable_data(self, target_replace_module=[
                              "Qwen2MLP", "Qwen2SdpaAttention", "Qwen2Attention", "Qwen2FlashAttention2"], r=self.config.data_rank1, r2=self.config.data_rank2)
        if self.config.adaprompt:
            self.model.prompt_init()

    def unwrap_model(self):
        uninject_trainable_data(self, target_replace_module=[
                                "Qwen2MLP", "Qwen2SdpaAttention", "Qwen2Attention", "Qwen2FlashAttention2"])
        if self.config.adaprompt:
            for e in range(self.config.num_hidden_layers // self.config.gap_layers):
                delattr(self.model, f"data_a_{e}")
                delattr(self.model, f"data_k_{e}")
                delattr(self.model, f"data_w_{e}")
                delattr(self.model, f"data_w2_{e}")
                delattr(self.model, f"data_a2_{e}")
                delattr(self.model, f"data_k2_{e}")
                if self.config.scale_bakebone:
                    delattr(self.model, f"data_w3_{e}")
                    delattr(self.model, f"data_a3_{e}")
                    delattr(self.model, f"data_k3_{e}")

    def load_model(self, state_dict):
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    def generate(self, **kwargs):
        # kwargs.pop('labels', None)
        return super().generate(**kwargs)


class LlamaDATA(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

    def wrap_model(self):
        if self.config.nomlp:
            logging.info('nomlp')
            inject_trainable_data(self, target_replace_module=[
                              "LlamaSdpaAttention", "LlamaAttention", "LlamaFlashAttention2"], r=self.config.data_rank1, r2=self.config.data_rank2)
        else:
            inject_trainable_data(self, target_replace_module=[
                              "LlamaMLP", "LlamaSdpaAttention", "LlamaAttention", "LlamaFlashAttention2"], r=self.config.data_rank1, r2=self.config.data_rank2)
        if self.config.adaprompt:
            self.model.prompt_init()

    def unwrap_model(self):
        uninject_trainable_data(self, target_replace_module=[
                                "LlamaMLP", "LlamaSdpaAttention", "LlamaAttention", "LlamaFlashAttention2"])
        if self.config.adaprompt:
            for e in range(self.config.num_hidden_layers // self.config.gap_layers):
                delattr(self.model, f"data_a_{e}")
                delattr(self.model, f"data_k_{e}")
                delattr(self.model, f"data_w_{e}")
                delattr(self.model, f"data_w2_{e}")
                delattr(self.model, f"data_a2_{e}")
                delattr(self.model, f"data_k2_{e}")
                if self.config.scale_bakebone:
                    delattr(self.model, f"data_w3_{e}")
                    delattr(self.model, f"data_a3_{e}")
                    delattr(self.model, f"data_k3_{e}")

    def load_model(self, state_dict):
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    def generate(self, **kwargs):
        # kwargs.pop('labels', None)
        return super().generate(**kwargs)


class T5DATA(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

    def wrap_model(self):
        if self.config.nomlp:
            inject_trainable_data(self, target_replace_module=["T5Attention"], r=self.config.data_rank1, r2=self.config.data_rank2)
        else:
            inject_trainable_data(self, target_replace_module=[
                            "T5Attention", "T5DenseActDense"], r=self.config.data_rank1, r2=self.config.data_rank2)
        
        if self.config.adaprompt:
            self.decoder.prompt_init()
            self.encoder.prompt_init()

    def unwrap_model(self):
        # if self.config.adaprompt:
        #     uninject_trainable_data(self, target_replace_module=["T5Attention"])
        # else:
        uninject_trainable_data(self, target_replace_module=["T5Attention", "T5DenseActDense"])
        if self.config.adaprompt:
            for e in range(self.config.num_layers // self.config.gap_layers):
                delattr(self.decoder, f"data_a_{e}")
                delattr(self.decoder, f"data_k_{e}")
                delattr(self.decoder, f"data_w_{e}")
                delattr(self.decoder, f"data_w2_{e}")
                delattr(self.decoder, f"data_a2_{e}")
                delattr(self.decoder, f"data_k2_{e}")
                # if self.config.is_encoder_decoder:
                delattr(self.encoder, f"data_a_{e}")
                delattr(self.encoder, f"data_k_{e}")
                delattr(self.encoder, f"data_w_{e}")
                delattr(self.encoder, f"data_w2_{e}")
                delattr(self.encoder, f"data_a2_{e}")
                delattr(self.encoder, f"data_k2_{e}")
                
                if self.config.scale_bakebone:
                    delattr(self.encoder, f"data_w3_{e}")
                    delattr(self.encoder, f"data_a3_{e}")
                    delattr(self.encoder, f"data_k3_{e}")
                    delattr(self.decoder, f"data_w3_{e}")
                    delattr(self.decoder, f"data_a3_{e}")
                    delattr(self.decoder, f"data_k3_{e}")

    def load_model(self, state_dict):
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def generate(self, **kwargs):
        return super().generate(**kwargs)

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p  

    # code for this function is modified from:
# https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
def gram_schmidt(vv, adaprompt, task_id):
    def projection(u, v):
        denominator = (u * u).sum()

        if denominator < 1e-8:
            return None
        else:
            return (v * u).sum() / denominator * u

    # check if the tensor is 3D and flatten the last two dimensions if necessary
    is_3d = len(vv.shape) == 3
    if is_3d:
        shape_2d = copy.deepcopy(vv.shape)
        vv = vv.view(vv.shape[0],-1)

    # swap rows and columns
    vv = vv.T

    # process matrix size
    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)

    # get starting point
    pt = int(adaprompt)
    s = int(task_id * pt)
    f = int((task_id + 1) * pt)
    if s > 0:
        uu[:, 0:s] = vv[:, 0:s].clone()
    for k in range(s, f):
        redo = True
        while redo:
            redo = False
            vk = torch.randn_like(vv[:,k]).to(vv.device)
            uk = 0
            for j in range(0, k):
                if not redo:
                    uj = uu[:, j].clone()
                    proj = projection(uj, vk)
                    if proj is None:
                        redo = True
                        print('restarting!!!')
                    else:
                        uk = uk + proj
            if not redo: uu[:, k] = vk - uk
    for k in range(s, f):
        uk = uu[:, k].clone()
        uu[:, k] = uk / (uk.norm())

    # undo swapping of rows and columns
    uu = uu.T 

    # return from 2D
    if is_3d:
        uu = uu.view(shape_2d)
    
    return torch.nn.Parameter(uu) 
    
def get_scale(input, name, e, config, task_id, train_mode, model):
    if name == 'scale1':
        k_name = f'data_k_{e}'
        a_name = f'data_a_{e}'
        w_name = f'data_w_{e}'
    else:
        k_name = f'data_k2_{e}'
        a_name = f'data_a2_{e}'
        w_name = f'data_w2_{e}'
    K = getattr(model, k_name)
    A = getattr(model, a_name)
    W = getattr(model, w_name)
    pt = int(adaprompt)
    s = int(task_id * pt)
    f = int((task_id + 1) * pt)
    
    if train_mode:
        if task_id > 0:
            K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
            A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
            W = torch.cat((W[:s].detach().clone(), W[s:f]), dim=0)
        else:
            K = K[s:f]
            A = A[s:f]
            W = W[s:f]
    else:
        K = K[0:f]
        A = A[0:f]
        W = W[0:f]
    a_q = torch.einsum('bld, kd -> blk', input, A)
    K = nn.functional.normalize(K, dim=-1)
    a_q = nn.functional.normalize(a_q, dim=-1)
    sim = torch.einsum('blk, kd -> blk', a_q, K) / (input.size(-1) ** 0.5)
    return torch.sum(torch.einsum('blk, kd -> bld', sim, W), (1, 2)).unsqueeze(-1).unsqueeze(-1)

# , iteration):
def update_ema_variables(ema_model, model, alpha_teacher, alpha_data):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    # return ema_model
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        if "data_" in name:
            ema_param.data[:] = alpha_data * ema_param[:].data[:] + \
                (1 - alpha_data) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]
    return ema_model





@torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    data_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'data_' in name:
            data_params_list.append(param)
        else:
            model_params_lst.append(param)
    return model_params_lst, data_params_list


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, cfg):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.cpu()
    data_params, data_names = inject_trainable_data(model=model, target_replace_module=["T5Attention", "T5DenseActDense"],
                                                    r=cfg.data_rank1, r2=cfg.data_rank2)
    if cfg.data_name_or_path != None:
        model = torch.nn.DataParallel(model)  # make parallel
        checkpoint = torch.load(f'{cfg.data_name_or_path}/data.pt')
        model.load_state_dict(checkpoint, strict=True)
    # if cfg.TEST.ckpt!=None:
    #     checkpoint = torch.load(cfg.TEST.ckpt)
    #     model.load_state_dict(checkpoint, strict=True)
    # model.to(device)
    # model.train()
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

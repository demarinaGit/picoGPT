# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:50:18 2023

@author: cormac
"""
import gpt2
import numpy as np

def mha(x, c_attn, c_proj, n_head):
    x = gpt2.linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [gpt2.attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = gpt2.linear(np.hstack(out_heads), **c_proj)
    
    outX = dict(); oX = dict()
    for hd in range(n_head):
        tmp = list()
        for i in range(n_head): tmp.append(np.zeros(out_heads[0].shape))
        tmp[hd] = out_heads[hd].copy()
        #outX[hd] = gpt2.linear(np.hstack(tmp), **c_proj) 
        outX[hd] = np.hstack(tmp)
        oX[hd] = gpt2.linear(outX[hd], **c_proj)
    return x, outX, out_heads, oX




def getX(prompt):
    from utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params("1558M", "models")
    input_ids = encoder.encode(prompt)
    wte = params['wte']
    wpe = params['wpe']
    n_head = hparams["n_head"]
    
    x = wte[input_ids] + wpe[range(len(input_ids))]
    block = params['blocks'][0]
    
    outX = mha(gpt2.layer_norm(x, **block['ln_1']), **block['attn'], n_head=n_head)
    #y = np.array([out_heads[i][-1] for i in range(0,n_head)])
    return outX


def gpt2Prime(inputs, wte, wpe, blocks, ln_f, n_head, loBlocks=1):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks[:-loBlocks]:                                   # All blocks but the last
        x = transformer_block(x, **block, n_head=n_head)
    lastButOneX = x.copy()
    
    xx, outX, out_heads, oX = mha(gpt2.layer_norm(x, **blocks[-1]['ln_1']), **blocks[-1]['attn'], n_head=n_head)
    x = x + xx
    x = x + gpt2.ffn(gpt2.layer_norm(x, **blocks[-1]['ln_2']), **blocks[-1]['mlp'])
    
    altX = dict()
    for hd in outX.keys():
        altX[hd] = lastButOneX + outX[hd]
        altX[hd] += gpt2.ffn(gpt2.layer_norm(altX[hd], **blocks[-1]['ln_2']), **blocks[-1]['mlp'])
        altX[hd] = gpt2.layer_norm(altX[hd], **ln_f) @ wte.T
    return gpt2.layer_norm(x, **ln_f) @ wte.T, altX

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + gpt2.mha(gpt2.layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + gpt2.ffn(gpt2.layer_norm(x, **ln_2), **mlp)
    return x

 
import ast,contextlib,gc,hashlib,json,math,random,time

from array import array

from pathlib import Path

from collections import Counter

import torch

from torch import nn

from torch.nn import functional as F

PAD,BOS,EOS,TOKEN_OFFSET=0,1,2,3

from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer

VOCAB=8192

MAX_POSITIONS=2048

GENERATION_CAP=512

SEED=42

class SwiGLUInput(nn.Module):
 def __init__(self,width,hidden):
  super().__init__();self.gate=nn.Linear(width,hidden);self.value=nn.Linear(width,hidden)
 def forward(self,x):return F.silu(self.gate(x))*self.value(x)

def replace_ffn_with_swiglu(transformer,width,hidden):
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(SEED).get_state())
  for layer in list(transformer.encoder.layers)+list(transformer.decoder.layers):
   layer.linear1=SwiGLUInput(width,hidden);layer.linear2=nn.Linear(hidden,width);layer.activation=nn.Identity()
   if isinstance(layer,nn.TransformerEncoderLayer):layer.activation_relu_or_gelu=0
   for module in [layer.linear1.gate,layer.linear1.value,layer.linear2]:nn.init.xavier_uniform_(module.weight)

class SimpleTransformer(nn.Module):
 def __init__(self,d_model=256,heads=4,encoder_layers=4,decoder_layers=4,swiglu_dim=680,
              vocab=VOCAB,max_positions=MAX_POSITIONS,dropout=.1):
  super().__init__();self.width=d_model;self.vocab=vocab
  self.tokens=nn.Embedding(vocab,d_model,padding_idx=PAD);self.positions=nn.Embedding(max_positions,d_model)
  self.transformer=nn.Transformer(d_model=d_model,nhead=heads,num_encoder_layers=encoder_layers,
   num_decoder_layers=decoder_layers,dim_feedforward=4*d_model,dropout=dropout,batch_first=True,norm_first=True)
  self.transformer.encoder.enable_nested_tensor=False;self.transformer.encoder.use_nested_tensor=False
  self.head=nn.Linear(d_model,vocab);replace_ffn_with_swiglu(self.transformer,d_model,swiglu_dim)
 def embed(self,ids):
  if ids.shape[1]>self.positions.num_embeddings:raise ValueError('Position overflow')
  return self.tokens(ids)*math.sqrt(self.width)+self.positions(torch.arange(ids.shape[1],device=ids.device))
 def decode_prefix(self,tgt,memory,source_padding):
  mask=torch.ones(tgt.shape[1],tgt.shape[1],device=tgt.device,dtype=torch.bool).triu(1)
  return self.transformer.decoder(self.embed(tgt),memory,tgt_mask=mask,tgt_key_padding_mask=tgt.eq(PAD),memory_key_padding_mask=source_padding)
 def forward(self,src,tgt):
  memory=self.transformer.encoder(self.embed(src),src_key_padding_mask=src.eq(PAD))
  return self.head(self.decode_prefix(tgt,memory,src.eq(PAD)))
 @torch.inference_mode()
 def generate(self,src,max_new_tokens=512):
  self.eval();memory=self.transformer.encoder(self.embed(src),src_key_padding_mask=src.eq(PAD))
  tgt=torch.full((len(src),1),BOS,device=src.device,dtype=torch.long);done=torch.zeros(len(src),device=src.device,dtype=torch.bool)
  for _ in range(max_new_tokens):
   logits=self.head(self.decode_prefix(tgt,memory,src.eq(PAD))[:,-1]);logits[:,[PAD,BOS]]=-torch.inf
   token=logits.argmax(-1).masked_fill(done,PAD);done|=token.eq(EOS);tgt=torch.cat([tgt,token[:,None]],dim=1)
   if bool(done.all()):break
  return tgt[:,1:]

def sync(device):
 if device.type=='cuda':torch.cuda.synchronize(device)

def init_tokenizer(path):
 tokenizer=Tokenizer.from_file(str(path))
 if tokenizer.get_vocab_size()!=VOCAB-TOKEN_OFFSET:raise ValueError('Unexpected vocabulary')
 visible=list(range(33,127))+list(range(161,173))+list(range(174,256));bchars={b:chr(b)for b in visible}
 for i,b in enumerate(b for b in range(256)if b not in bchars):bchars[b]=chr(256+i)
 charbytes={c:b for b,c in bchars.items()}
 tokenbytes=[b'',b'',b'']+[bytes(charbytes[c]for c in tokenizer.id_to_token(i))for i in range(VOCAB-3)]
 return tokenizer,tokenbytes

def encode(tokenizer,text):return [i+TOKEN_OFFSET for i in tokenizer.encode(text,add_special_tokens=False).ids]

def decode_bytes(tokenbytes,ids):
 chunks=[]
 for i in ids:
  if i==EOS:break
  if i>=TOKEN_OFFSET:chunks.append(tokenbytes[i])
 return b''.join(chunks)

def source_batch(tokenizer,questions,device):
 return pad_sequence([torch.tensor([BOS]+encode(tokenizer,q)+[EOS])for q in questions],batch_first=True,padding_value=PAD).to(device)

def collate_cache(cache,indices,device):
 src=pad_sequence([torch.tensor(cache[i][0],dtype=torch.long)for i in indices],batch_first=True,padding_value=PAD)
 tgt=pad_sequence([torch.tensor([BOS]+list(cache[i][1]),dtype=torch.long)for i in indices],batch_first=True,padding_value=PAD)
 labels=pad_sequence([torch.tensor(list(cache[i][1])+[EOS],dtype=torch.long)for i in indices],batch_first=True,padding_value=PAD)
 counts={'source_tokens':sum(len(cache[i][0])for i in indices),'target_tokens':sum(len(cache[i][1])+1 for i in indices),
  'source_slots':src.numel(),'target_slots':labels.numel(),'pairs':len(indices)}
 return src.to(device),tgt.to(device),labels.to(device),counts

def bucket_plan(cache,batch,steps,seed=230042,chunk_pairs=4096):
 if chunk_pairs%batch:raise ValueError('Chunk must be divisible by batch')
 rng=random.Random(seed);plans=[]
 while len(plans)<steps:
  chunk=rng.sample(range(len(cache)),min(chunk_pairs,len(cache)))
  chunk.sort(key=lambda i:(len(cache[i][0])+len(cache[i][1])+1,len(cache[i][0]),len(cache[i][1])))
  batches=[chunk[i:i+batch]for i in range(0,len(chunk)-batch+1,batch)]
  if not batches:raise ValueError('Too few data for this batch size')
  rng.shuffle(batches);plans.extend(batches)
 return plans[:steps]

def sdpa_flags():
 return {k:bool(getattr(torch.backends.cuda,k+'_sdp_enabled')())for k in ['flash','mem_efficient','math','cudnn']if hasattr(torch.backends.cuda,k+'_sdp_enabled')}

def profile_sdpa(call,device):
 if device.type!='cuda':return {'status':'CPU_smoke','operators':[]}
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
  call();sync(device)
 ops=sorted({e.key for e in prof.key_averages()if 'scaled_dot_product'in e.key})
 return {'status':'observed','operators':ops,'scope':'one inference preflight outside timers; other shapes/backward may use a different backend'}

def fused_adamw_available(device):
 if device.type!='cuda':return False
 try:
  p=nn.Parameter(torch.ones(1,device=device));opt=torch.optim.AdamW([p],lr=1e-4,fused=True);p.grad=torch.ones_like(p);opt.step();sync(device)
  del opt,p;return True
 except (TypeError,RuntimeError):return False

class CachedDecoder:
    @torch.inference_mode()
    def __init__(self, model, src, capacity):
        model.eval()
        if not 0 < capacity <= model.positions.num_embeddings:
            raise ValueError('Invalid cache capacity')
        self.model, self.capacity, self.position = model, capacity, 0
        self.batch_size = len(src)
        self.width = model.tokens.embedding_dim
        self.heads = model.transformer.decoder.layers[0].self_attn.num_heads
        self.head_dim = self.width // self.heads
        memory = model.transformer.encoder(model.embed(src), src_key_padding_mask=src.eq(PAD))
        # SDPA bool masks use True for allowed keys (opposite to MHA padding masks).
        self.source_allowed = src.ne(PAD)[:, None, None, :]
        self.valid = torch.zeros(len(src), capacity, dtype=torch.bool, device=src.device)
        self.position_vectors = model.positions(torch.arange(capacity, device=src.device))
        self.layers = []
        for layer in model.transformer.decoder.layers:
            if not layer.norm_first:
                raise ValueError('This experiment requires Pre-LN decoder layers')
            for attention in (layer.self_attn, layer.multihead_attn):
                if (not attention._qkv_same_embed_dim or attention.num_heads != self.heads
                        or attention.bias_k is not None or attention.bias_v is not None
                        or attention.add_zero_attn):
                    raise ValueError('Unsupported attention configuration')
            attention = layer.multihead_attn
            projected_kv = F.linear(memory, attention.in_proj_weight[self.width:],
                                    attention.in_proj_bias[self.width:])
            cross_k, cross_v = projected_kv.chunk(2, dim=-1)
            shape = (len(src), self.heads, capacity, self.head_dim)
            self.layers.append({
                'self_k': memory.new_empty(shape), 'self_v': memory.new_empty(shape),
                'cross_k': self.split_heads(cross_k), 'cross_v': self.split_heads(cross_v),
            })

    def split_heads(self, values):
        return values.reshape(self.batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

    def attend(self, attention, q, k, v, allowed):
        # One query at the latest position; all supplied keys are from its past/current.
        # is_causal=True would use an upper-left 1 x T mask and incorrectly hide the past.
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).contiguous().reshape(self.batch_size, 1, self.width)
        return attention.out_proj(output)

    @torch.inference_mode()
    def step(self, token_ids):
        if self.position >= self.capacity:
            raise ValueError('Cache capacity exceeded')
        if token_ids.ndim != 1 or len(token_ids) != self.batch_size:
            raise ValueError('One token per batch row is required')
        t = self.position
        x = self.model.tokens(token_ids[:, None]) * math.sqrt(self.width)
        x = x + self.position_vectors[t][None, None, :]
        self.valid[:, t] = token_ids.ne(PAD)
        self_allowed = self.valid[:, None, None, :t + 1]
        for layer, cache in zip(self.model.transformer.decoder.layers, self.layers):
            attention = layer.self_attn
            q, k, v = F.linear(layer.norm1(x), attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
            cache['self_k'][:, :, t:t + 1] = self.split_heads(k)
            cache['self_v'][:, :, t:t + 1] = self.split_heads(v)
            self_output = self.attend(attention, self.split_heads(q),
                cache['self_k'][:, :, :t + 1], cache['self_v'][:, :, :t + 1], self_allowed)
            x = x + layer.dropout1(self_output)

            attention = layer.multihead_attn
            q = F.linear(layer.norm2(x), attention.in_proj_weight[:self.width], attention.in_proj_bias[:self.width])
            cross_output = self.attend(attention, self.split_heads(q),
                                      cache['cross_k'], cache['cross_v'], self.source_allowed)
            x = x + layer.dropout2(cross_output)
            feed_forward = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x)))))
            x = x + layer.dropout3(feed_forward)

        if self.model.transformer.decoder.norm is not None:
            x = self.model.transformer.decoder.norm(x)
        self.position += 1
        return self.model.head(x[:, 0])

def code_properties(code):
    def scoped_nodes(node):
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            if not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.extend(ast.iter_child_nodes(current))
    try:
        tree = ast.parse(code)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve']
        has_return = any(isinstance(n, ast.Return) and n.value is not None
            for f in functions for stmt in f.body for n in scoped_nodes(stmt))
        return {'parses_as_python': True, 'defines_solve': bool(functions), 'solve_returns_value': has_return,
                'syntax_error_type': None, 'syntax_error_message': None, 'indentation_error': False}
    except (SyntaxError, ValueError, RecursionError) as error:
        return {'parses_as_python': False, 'defines_solve': False, 'solve_returns_value': False,
                'syntax_error_type': type(error).__name__, 'syntax_error_message': str(error),
                'indentation_error': isinstance(error, IndentationError)}

class BudgetExpired(Exception):pass

def check_deadline(deadline):
 if deadline is not None and time.monotonic()>=deadline:raise BudgetExpired('Session budget reached; completed run boundaries are preserved')

def active_tokens(ids):
 ids=[int(i)for i in ids if i!=PAD]
 return ids[:ids.index(EOS)+1]if EOS in ids else ids

@torch.inference_mode()
def generate_preallocated(model,src,cap=GENERATION_CAP,deadline=None,check_every=16):
 """Only output storage and termination checks change; same encoder/KV/argmax."""
 cache=CachedDecoder(model,src,cap)
 output=torch.full((len(src),cap),PAD,device=src.device,dtype=torch.long)
 token=torch.full((len(src),),BOS,device=src.device,dtype=torch.long)
 done=torch.zeros(len(src),device=src.device,dtype=torch.bool)
 for step in range(cap):
  if step%check_every==0:check_deadline(deadline)
  logits=cache.step(token);logits[:,[PAD,BOS]]=-torch.inf
  token=logits.argmax(-1).masked_fill(done,PAD);output[:,step]=token;done|=token.eq(EOS)
  # Only one device->CPU synchronization per 16 steps, plus final output transfer.
  # Rows already at EOS remain PAD; active_tokens removes these extra slots.
  if (step+1)%check_every==0 and bool(done.all()):break
 return output[:,:cache.position],cache.position

def restore_order(ordered_values,original_indices,size):
 if len(ordered_values)!=size or sorted(original_indices)!=list(range(size)):
  raise ValueError('Not a complete permutation')
 restored=[None]*size
 for value,i in zip(ordered_values,original_indices):restored[i]=value
 return restored

def inspect_output(row,ids,tokenizer,tokenbytes):
 active=active_tokens(ids);raw=decode_bytes(tokenbytes,active)
 try:code=raw.decode('utf-8');valid=True
 except UnicodeDecodeError:code=raw.decode('utf-8',errors='replace');valid=False
 example={'question':row['question'],'generated_code':code,'generated_token_ids':active,
  'generated_tokens_including_eos':len(active),'generated_utf8_bytes':len(raw),'valid_utf8':valid,
  'eos_emitted':EOS in active,'reached_token_cap':len(active)==GENERATION_CAP,
  'token_cap_without_eos':len(active)==GENERATION_CAP and EOS not in active,
  'byte_budget_stop':False,'stop_reason':'eos'if EOS in active else'token_cap',**code_properties(code)}
 if 'code'in row:
  gold=encode(tokenizer,row['code'])+[EOS]
  example.update(reference_code=row['code'],reference_utf8_bytes=len(row['code'].encode()),
   reference_bpe_tokens_including_eos=len(gold),exact_code_match=valid and code==row['code'],exact_tokens_with_eos=active==gold)
 return example

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
 return h.hexdigest()

def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

MODEL_SPEC=dict(d_model=256,heads=4,encoder_layers=4,decoder_layers=4,swiglu_dim=680)

"""DNO research v0: recurrent, sparse, stateful relational neural operator.

This is a TRAINABLE algorithmic prototype, not a trained LLM and not a proof of
Delta theory. Every next-token prediction is causal. No transformer attention.
Requires PyTorch >= 2.1. See MATHEMATICS_ZH.md for explicit assumptions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DNOState:
    """Only CURRENT effective state. No token/event replay logs.

    h: [B,N,D] individual states
    boundary: [B,N,C] locally mutable interface constraints
    relations: [B,N,K,R] per-port current relation states
    neighbors: [B,N,K] identities of most recent neighbors (not a log)
    activity: [B,N,1] current activity
    tick: next local time step
    """
    h: Tensor
    boundary: Tensor
    relations: Tensor
    neighbors: Tensor
    activity: Tensor
    tick: int = 0

    def detach(self) -> 'DNOState':
        return DNOState(self.h.detach(), self.boundary.detach(),
                        self.relations.detach(), self.neighbors.detach(),
                        self.activity.detach(), self.tick)

    def clone(self) -> 'DNOState':
        return DNOState(self.h.clone(), self.boundary.clone(),
                        self.relations.clone(), self.neighbors.clone(),
                        self.activity.clone(), self.tick)


class DNO(nn.Module):
    """Small, causal LLM backbone substitute based on present-state relations.

    Each input token perturbs a rotating individual slot. An individual evolves
    autonomously, while conditional pair interactions update both the individual
    and the relation itself. Relations also influence *other relations* through
    summaries of edges sharing an endpoint. Boundaries change with action.

    Candidate contacts consist of local ring neighbors, neighbors in a sorted
    learned 1-D index, and one rotating exploratory candidate. K=5 fixed ports.
    Edge memory follows the same neighbor identity across steps. A changed
    contact starts with zero edge state; the old port state is released.

    Limits: hard sorting has non-differentiable topology; finite N/K; no arbitrary
    relation discovery, no proof of true language understanding or scale-up.
    """

    def __init__(self, vocab_size: int, slots: int = 7, d_model: int = 32,
                 d_constraint: int = 12, d_relation: int = 12,
                 dynamic_contacts: bool = True, relation_to_relation: bool = True):
        super().__init__()
        if slots < 5:
            raise ValueError('slots must be >= 5 for fixed 5-port contact rule')
        self.vocab_size = vocab_size
        self.slots = slots
        self.d_model = d_model
        self.d_constraint = d_constraint
        self.d_relation = d_relation
        self.ports = 5
        self.dynamic_contacts = dynamic_contacts
        self.relation_to_relation = relation_to_relation

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.inject = nn.Linear(d_model, d_model)
        self.index_key = nn.Linear(d_model + d_constraint, 1)
        self.autonomous = nn.Sequential(nn.Linear(d_model + d_constraint, d_model),
                                        nn.Tanh(), nn.Linear(d_model, d_model))
        pair_features = 2*d_model + 2*d_constraint + d_relation + 3
        self.pair_gate = nn.Sequential(nn.Linear(pair_features, d_model), nn.Tanh(),
                                       nn.Linear(d_model, 1))
        self.message = nn.Sequential(nn.Linear(pair_features, d_model), nn.Tanh(),
                                     nn.Linear(d_model, d_model))
        edge_features = pair_features + 2*d_relation
        self.relation_update = nn.Sequential(nn.Linear(edge_features, d_model), nn.Tanh(),
                                             nn.Linear(d_model, d_relation))
        self.boundary_update = nn.Sequential(
            nn.Linear(d_model + d_constraint + d_model + d_relation, d_model),
            nn.Tanh(), nn.Linear(d_model, d_constraint))
        self.unit_gate = nn.Linear(d_model + d_constraint + d_model, d_model)
        self.activity_update = nn.Linear(d_model + d_constraint + 1, 1)
        self.output_norm = nn.LayerNorm(d_model * 2)
        self.readout = nn.Linear(2*d_model, vocab_size)

        # Conservative initial updates: prevents immediate unbounded dynamics.
        nn.init.zeros_(self.relation_update[-1].weight)
        nn.init.normal_(self.relation_update[-1].weight, std=0.008)
        nn.init.zeros_(self.relation_update[-1].bias)
        nn.init.zeros_(self.boundary_update[-1].bias)

    def initial_state(self, batch: int, *, device=None, dtype=None) -> DNOState:
        weight = self.embedding.weight
        device = device if device is not None else weight.device
        dtype = dtype if dtype is not None else weight.dtype
        n, k = self.slots, self.ports
        return DNOState(
            torch.zeros(batch, n, self.d_model, device=device, dtype=dtype),
            torch.zeros(batch, n, self.d_constraint, device=device, dtype=dtype),
            torch.zeros(batch, n, k, self.d_relation, device=device, dtype=dtype),
            torch.full((batch, n, k), -1, device=device, dtype=torch.long),
            torch.zeros(batch, n, 1, device=device, dtype=dtype), 0)

    @staticmethod
    def _gather_units(source: Tensor, ids: Tensor) -> Tensor:
        # source [B,N,D], ids [B,N,K] => [B,N,K,D]
        b, n, d = source.shape
        k = ids.size(-1)
        return torch.gather(source.unsqueeze(1).expand(b,n,n,d), 2,
                            ids.unsqueeze(-1).expand(b,n,k,d))

    def candidate_neighbors(self, h: Tensor, c: Tensor, tick: int) -> Tensor:
        """O(N log N + NK) candidate discovery; does NOT form N x N scores.

        Sorting is hard / piecewise constant. index_key still receives gradient
        because scores also appear in pair_gate and relation_update features.
        """
        b,n,_ = h.shape
        base=torch.arange(n,device=h.device).view(1,n).expand(b,n)
        left=(base-1)%n
        right=(base+1)%n
        if self.dynamic_contacts:
            index=self.index_key(torch.cat([h,c],dim=-1)).squeeze(-1)
            order=torch.argsort(index,dim=1,stable=True)
            rank=torch.argsort(order,dim=1)
            sorted_left=torch.gather(order,1,(rank-1)%n)
            sorted_right=torch.gather(order,1,(rank+1)%n)
            # Deterministic bounded exploration that changes with present tick.
            explore=(base+2+(tick//2))%n
        else:
            sorted_left=(base-2)%n
            sorted_right=(base+2)%n
            explore=(base+3)%n
        return torch.stack([left,right,sorted_left,sorted_right,explore],dim=-1)

    def step(self, token_ids: Tensor, state: DNOState,
             *, return_diagnostics: bool=False):
        """One causal token step. No in-place mutation of previous state."""
        if token_ids.ndim != 1:
            raise ValueError('step tokens must be [batch]')
        b,n,d=state.h.shape
        if token_ids.shape[0]!=b or n!=self.slots:
            raise ValueError('token batch / slot mismatch')
        device=state.h.device
        tick=state.tick
        active_slot=tick%n
        mask=F.one_hot(torch.full((b,),active_slot,device=device),n).to(state.h.dtype)
        h_in=state.h + mask.unsqueeze(-1)*self.inject(self.embedding(token_ids)).unsqueeze(1)
        # Each unit has intrinsic dynamics regardless of contact.
        intrinsic=self.autonomous(torch.cat([h_in,state.boundary],dim=-1))
        neighbors=self.candidate_neighbors(h_in,state.boundary,tick)
        k=neighbors.size(-1)
        # Preserve edge state iff same directed pair remains in currently live ports.
        matches=(neighbors.unsqueeze(-1)==state.neighbors.unsqueeze(-2)) & (state.neighbors.unsqueeze(-2)>=0)
        was_present=matches.any(-1)
        old_port=matches.to(torch.int64).argmax(-1)
        old_edge=torch.gather(state.relations,2,old_port.unsqueeze(-1).expand(b,n,k,self.d_relation))
        old_edge=old_edge*was_present.unsqueeze(-1)
        hi=h_in.unsqueeze(2).expand(-1,-1,k,-1)
        hj=self._gather_units(h_in,neighbors)
        ci=state.boundary.unsqueeze(2).expand(-1,-1,k,-1)
        cj=self._gather_units(state.boundary,neighbors)
        idx=self.index_key(torch.cat([h_in,state.boundary],dim=-1))
        idxj=self._gather_units(idx,neighbors)
        idxi=idx.unsqueeze(2).expand(-1,-1,k,-1)
        pair=torch.cat([hi,hj,ci,cj,old_edge,idxi,idxj,idxi-idxj],dim=-1)
        # Deduplicate ports and exclude self-loops. Ports not used do no work.
        base=torch.arange(n,device=device).view(1,n,1)
        self_edge=neighbors==base
        prev_equal=(neighbors.unsqueeze(-1)==neighbors.unsqueeze(-2))
        lower=torch.tril(torch.ones(k,k,device=device,dtype=torch.bool),diagonal=-1)
        duplicate=(prev_equal & lower).any(-1)
        allowed=~(self_edge|duplicate)
        gate=torch.sigmoid(self.pair_gate(pair))*allowed.unsqueeze(-1)
        msg=gate*torch.tanh(self.message(pair))
        msg_sum=msg.sum(dim=2)/max(1,k)
        # Relations influence other relations through neighbor/shared-port context.
        old_summary=(old_edge*allowed.unsqueeze(-1)).sum(2)/max(1,k)
        if self.relation_to_relation:
            other_context=self._gather_units(old_summary,neighbors)
            own_context=old_summary.unsqueeze(2).expand(-1,-1,k,-1)
        else:
            own_context=torch.zeros_like(old_edge)
            other_context=torch.zeros_like(old_edge)
        edge_input=torch.cat([pair,own_context,other_context],dim=-1)
        edge_delta=torch.tanh(self.relation_update(edge_input))
        # A contact can be quiescent or become active; its state can decay.
        new_edge=torch.tanh(0.97*old_edge+0.18*gate*edge_delta)*allowed.unsqueeze(-1)
        # Live relation states affect local mutable boundaries.
        rel_sum=new_edge.sum(dim=2)/max(1,k)
        boundary_delta=torch.tanh(self.boundary_update(
            torch.cat([h_in,state.boundary,msg_sum,rel_sum],dim=-1)))
        new_boundary=torch.tanh(0.985*state.boundary+0.09*boundary_delta)
        unit_update=torch.tanh(h_in+0.12*intrinsic+0.25*msg_sum)
        learn_gate=torch.sigmoid(self.unit_gate(torch.cat([h_in,state.boundary,msg_sum],dim=-1)))
        new_h=(1.0-0.22*learn_gate)*h_in+0.22*learn_gate*unit_update
        # Present activity, NOT a count/log of past events.
        activity_delta=torch.sigmoid(self.activity_update(
            torch.cat([new_h,new_boundary,state.activity],dim=-1)))
        activity=0.90*state.activity+0.10*activity_delta
        new_state=DNOState(new_h,new_boundary,new_edge,neighbors,activity,tick+1)
        selected=new_h[:,active_slot,:]
        pooled=(new_h*activity).sum(dim=1)/(activity.sum(dim=1)+1e-6)
        logit=self.readout(self.output_norm(torch.cat([selected,pooled],dim=-1)))
        if return_diagnostics:
            return logit,new_state,{
                'mean_gate':gate.sum().detach()/allowed.sum().clamp(min=1),
                'active_contacts':allowed.sum().detach(),
                'contact_fraction_reused':(was_present & allowed).sum().detach()/allowed.sum().clamp(min=1),
                'mean_relation_norm':new_edge.norm(dim=-1).mean().detach(),
                'mean_boundary_norm':new_boundary.norm(dim=-1).mean().detach(),
            }
        return logit,new_state

    def forward(self,tokens: Tensor,state: Optional[DNOState]=None,
                *,return_state: bool=False):
        """tokens [B,T] => causal logits [B,T,V]; state after processing T.

        For next-token LM: cross_entropy(logits[:,:-1], tokens[:,1:]).
        """
        if tokens.ndim!=2 or tokens.size(1)==0:
            raise ValueError('tokens must be nonempty [batch,time]')
        b,t=tokens.shape
        if state is None:
            state=self.initial_state(b,device=tokens.device,dtype=self.embedding.weight.dtype)
        outs=[]
        for step_idx in range(t):
            logits,state=self.step(tokens[:,step_idx],state)
            outs.append(logits)
        logits=torch.stack(outs,dim=1)
        return (logits,state) if return_state else logits

    @torch.no_grad()
    def generate(self,prompt: Tensor,max_new_tokens: int=20,temperature: float=1.,
                 greedy:bool=True) -> Tensor:
        if prompt.ndim!=2 or prompt.size(1)==0:
            raise ValueError('prompt must be nonempty [B,T]')
        _,state=self.forward(prompt,return_state=True)
        logits=self.readout(self.output_norm(torch.cat([
            state.h[:,(state.tick-1)%self.slots,:],
            (state.h*state.activity).sum(1)/(state.activity.sum(1)+1e-6)
        ],dim=-1)))
        result=[prompt]
        for i in range(max_new_tokens):
            probs=logits/max(temperature,1e-5)
            next_token=probs.argmax(-1) if greedy else torch.distributions.Categorical(logits=probs).sample()
            result.append(next_token[:,None])
            logits,state=self.step(next_token,state)
        return torch.cat(result,dim=1)


def parameter_count(model:nn.Module)->int:
    return sum(p.numel() for p in model.parameters())

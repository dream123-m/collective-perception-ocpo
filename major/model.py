"""MAPPO with an on-policy conditional swap module; initialized from scratch."""
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
from MAPPO.mappo_network import ActorNetwork

class PairwiseRefiner(nn.Module):
    """Selected-object-or-STOP, then conditional unselected-object choice."""
    def __init__(self, state_dim, num_agents, num_slots, hidden=48,
                 normalize_position=False, balanced_gate=False, object_feature_dim=0):
        super().__init__()
        self.num_slots = num_slots
        self.normalize_position = normalize_position
        self.balanced_gate = balanced_gate
        self.agent_embedding = nn.Embedding(num_agents, 8)
        self.slot_embedding = nn.Embedding(num_slots, 8)
        self.context = nn.Sequential(nn.Linear(state_dim+2*num_slots+8, hidden), nn.Tanh())
        self.object_feature_dim=object_feature_dim
        self.object_encoder = nn.Sequential(nn.Linear(hidden+10+object_feature_dim, hidden), nn.Tanh())
        self.drop_head = nn.Linear(hidden, 1)
        self.stop_head = nn.Linear(hidden, 1)
        self.add_head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # CCRE regresses an absolute marginal gain for each (drop, add) pair.
        # Unlike a categorical preference score, zero has a fixed meaning:
        # STOP is preferred whenever no predicted gain is positive.
        self.gain_head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for head in (self.drop_head, self.stop_head, self.add_head[-1]):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)
        # A residual editor must begin by preserving the MAPPO proposal.
        # In balanced_gate mode, sigmoid(stop) is P(STOP).
        if balanced_gate:
            nn.init.constant_(self.stop_head.bias, 2.944439)  # logit(0.95)

    def distributions(self, states, base, probs, masks, object_features=None, drop=None, deterministic=False):
        b, n, t = base.shape
        ids = torch.arange(n, device=states.device)
        agent = self.agent_embedding(ids)[None].expand(b, -1, -1)
        features = states
        if self.normalize_position:
            features = states.clone()
            features[...,:2] = features[...,:2] / 1000.0
        ctx = self.context(torch.cat((features, base.float(), probs, agent), dim=-1))
        slots = self.slot_embedding.weight[None, None].expand(b, n, -1, -1)
        if object_features is None:
            object_features=torch.zeros(b,n,t,self.object_feature_dim,device=states.device)
        if object_features.shape != (b,n,t,self.object_feature_dim):
            raise ValueError('Object-relation feature shape does not match the editor.')
        tokens = self.object_encoder(torch.cat((ctx[:,:,None].expand(-1,-1,t,-1), slots,
                                               probs[...,None], base.float()[...,None], object_features), dim=-1))
        selected = base & masks
        unselected = (~base) & masks
        can_swap = selected.any(-1) & unselected.any(-1)
        drop_logits = self.drop_head(tokens).squeeze(-1).masked_fill(~selected | ~can_swap[...,None], -torch.inf)
        # A residual editor begins with a 95% STOP probability.
        stop = self.stop_head(ctx).squeeze(-1)
        if self.balanced_gate:
            # Explicit binary STOP/swap gate followed by a conditional drop.
            # Represent the joint gate/drop distribution in one categorical so
            # sampled/replayed likelihoods retain exactly the same contract.
            safe_drop = drop_logits.clone()
            safe_drop[~can_swap] = 0.0
            conditional_drop = torch.log_softmax(safe_drop, -1)
            combined = conditional_drop + torch.nn.functional.logsigmoid(-stop)[...,None]
            combined = combined.masked_fill(~can_swap[...,None], -torch.inf)
            stop_logp = torch.where(can_swap, torch.nn.functional.logsigmoid(stop), 0.0)
            first = Categorical(logits=torch.cat((combined, stop_logp[...,None]), -1))
        else:
            stop = stop + torch.log(4*selected.sum(-1).clamp_min(1).float())
            first = Categorical(logits=torch.cat((drop_logits, stop[...,None]), -1))
        if drop is None:
            # Deterministic evaluation must choose the mode of the *complete*
            # STOP/drop categorical.  Summing all swap probabilities can make
            # many individually unlikely exchanges overpower STOP and does not
            # correspond to any action trained by PPO.
            drop = first.logits.argmax(-1) if deterministic else first.sample()
        chosen = tokens.gather(2, drop.clamp_max(t-1)[...,None,None].expand(-1,-1,1,tokens.shape[-1])).squeeze(2)
        add_logits = self.add_head(torch.cat((tokens, chosen[:,:,None].expand(-1,-1,t,-1)), -1)).squeeze(-1)
        # STOP/no-unselected rows receive a dummy categorical. It contributes
        # neither likelihood nor an actual edit.
        add_mask = unselected.clone()
        inactive = drop == t
        add_mask[inactive] = False
        add_mask[...,0] |= inactive
        second = Categorical(logits=add_logits.masked_fill(~add_mask, -torch.inf))
        return first, second, drop

    def sample(self, states, base, probs, masks, object_features=None, deterministic=False):
        first, second, drop = self.distributions(states, base, probs, masks, object_features, deterministic=deterministic)
        add = second.logits.argmax(-1) if deterministic else second.sample()
        active = drop < self.num_slots
        logp = first.log_prob(drop) + torch.where(active, second.log_prob(add), 0.0)
        final = base.clone()
        bi, ai = active.nonzero(as_tuple=True)
        final[bi, ai, drop[bi, ai]] = False
        final[bi, ai, add[bi, ai]] = True
        return final, drop, add, logp, active

    def evaluate_path(self, states, base, probs, masks, drop, add, object_features=None):
        first, second, _ = self.distributions(states, base, probs, masks, object_features, drop=drop)
        logp = first.log_prob(drop) + torch.where(drop < self.num_slots, second.log_prob(add), 0.0)
        # Only first-stage entropy is regularized; do not label it full-path entropy.
        return logp, first.entropy()

    def candidate_values(self, states, base, probs, object_features=None):
        """Predicted immediate marginal reward for every ordered slot pair."""
        b,n,t=base.shape
        ids=torch.arange(n,device=states.device)
        agent=self.agent_embedding(ids)[None].expand(b,-1,-1)
        features=states.clone() if self.normalize_position else states
        if self.normalize_position: features[...,:2]=features[...,:2]/1000.
        ctx=self.context(torch.cat((features,base.float(),probs,agent),dim=-1))
        slots=self.slot_embedding.weight[None,None].expand(b,n,-1,-1)
        if object_features is None:
            object_features=torch.zeros(b,n,t,self.object_feature_dim,device=states.device)
        tokens=self.object_encoder(torch.cat((ctx[:,:,None].expand(-1,-1,t,-1),slots,
                                               probs[...,None],base.float()[...,None],object_features),dim=-1))
        pair=torch.cat((tokens[:,:,:,None,:].expand(-1,-1,-1,t,-1),
                        tokens[:,:,None,:,:].expand(-1,-1,t,-1,-1)),dim=-1)
        return self.gain_head(pair).squeeze(-1)

class JointPolicy(nn.Module):
    def __init__(self, state_dim, num_agents, num_slots, hidden_dims, use_editor=True, object_feature_dim=0):
        super().__init__()
        actors=[ActorNetwork(state_dim,num_slots,hidden_dims) for _ in range(num_agents)]
        keys=('shared_net.0','shared_net.2','action_head')
        saved=[a.state_dict() for a in actors]
        self.weights=nn.ParameterList([nn.Parameter(torch.stack([s[k+'.weight'] for s in saved])) for k in keys])
        self.biases=nn.ParameterList([nn.Parameter(torch.stack([s[k+'.bias'] for s in saved])) for k in keys])
        self.editor=PairwiseRefiner(state_dim,num_agents,num_slots,normalize_position=True,balanced_gate=True,
                                    object_feature_dim=object_feature_dim)
        self.editor.context=nn.Sequential(nn.Linear(state_dim+2*num_slots+8,48),nn.LayerNorm(48),nn.Tanh())
        self.num_agents=num_agents
        self.use_editor=use_editor
        if not use_editor:
            for p in self.editor.parameters(): p.requires_grad_(False)

    def probabilities(self, states):
        x=states
        for j,(w,b) in enumerate(zip(self.weights,self.biases)):
            x=torch.einsum('bni,noi->bno',x,w)+b[None]
            if j<2: x=x.tanh()
        return x.sigmoid().clamp(1e-6,1-1e-6)

    def sample(self, states, masks, deterministic=False, object_features=None):
        probs=self.probabilities(states)
        dist=Bernoulli(probs=probs)
        base=((probs>.5) if deterministic else dist.sample().bool()) & masks
        if not self.use_editor:
            drop=torch.full(base.shape[:2],self.editor.num_slots,dtype=torch.long)
            add=torch.zeros_like(drop)
            base_logp=(dist.log_prob(base.float())*masks).sum(-1)
            return base,base,drop,add,base_logp,torch.zeros_like(base_logp),torch.zeros_like(drop,dtype=torch.bool)
        final,drop,add,edit_logp,active=self.editor.sample(states,base,probs,masks,object_features,deterministic)
        base_logp=(dist.log_prob(base.float())*masks).sum(-1)
        return final,base,drop,add,base_logp,edit_logp,active

    def evaluate_path(self, states, masks, base, drop, add, object_features=None):
        probs=self.probabilities(states)
        dist=Bernoulli(probs=probs)
        if self.use_editor:
            edit_logp,edit_entropy=self.editor.evaluate_path(states,base,probs.detach(),masks,drop,add,object_features)
        else:
            edit_logp=edit_entropy=torch.zeros(base.shape[:2])
        base_logp=(dist.log_prob(base.float())*masks).sum(-1)
        base_entropy=(dist.entropy()*masks).sum(-1)/masks.sum(-1).clamp_min(1)
        return base_logp,edit_logp,base_entropy,edit_entropy

    def clip_gradients(self, max_norm):
        # Preserve the original per-agent actor clipping, rather than clipping
        # all independent actors together. The editor has shared parameters.
        params=list(self.weights)+list(self.biases)
        norms=torch.zeros(self.num_agents)
        for p in params:
            if p.grad is not None: norms+=p.grad.flatten(1).pow(2).sum(1)
        scales=(max_norm/(norms.sqrt()+1e-6)).clamp_max(1)
        for p in params:
            if p.grad is not None: p.grad.mul_(scales.reshape((-1,)+(1,)*(p.ndim-1)))
        nn.utils.clip_grad_norm_(self.editor.parameters(),max_norm)


class BudgetSetPolicy(JointPolicy):
    """MAPPO actor whose action is a sampled cardinality followed by a set path."""
    def __init__(self,state_dim,num_agents,num_slots,hidden_dims):
        super().__init__(state_dim,num_agents,num_slots,hidden_dims,use_editor=False)
        self.num_slots=num_slots
        self.count_head=nn.Linear(state_dim,num_slots+1)
        self.slot_embedding=nn.Embedding(num_slots,16)
        self.context_head=nn.Sequential(nn.Linear(state_dim+32,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.count_head.weight); nn.init.zeros_(self.count_head.bias)
        # Avoid deterministic tie-breaking to K=0 before PPO has observed a
        # rollout; five broadcasts is close to the established MAPPO regime.
        if num_slots>=5: self.count_head.bias.data[5]=1.
        nn.init.zeros_(self.context_head[-1].weight); nn.init.zeros_(self.context_head[-1].bias)

    def _paths(self,states,masks,order=None,counts=None,deterministic=False):
        masks=masks.expand(states.shape[0],-1,-1)
        b,n,t=masks.shape; valid=masks.sum(-1)
        count_logits=self.count_head(states)
        k=torch.arange(t+1,device=states.device)[None,None]
        count_logits=count_logits.masked_fill(k>valid[...,None],-torch.inf)
        count_dist=Categorical(logits=count_logits)
        counts=count_dist.logits.argmax(-1) if deterministic and counts is None else (count_dist.sample() if counts is None else counts)
        replay=order is not None
        if order is None: order=torch.full((b,n,t),t,dtype=torch.long,device=states.device)
        selected=torch.zeros_like(masks); action=torch.zeros_like(masks); logp=count_dist.log_prob(counts)
        entropy=count_dist.entropy()
        x=states
        for j,(w,bias) in enumerate(zip(self.weights,self.biases)):
            x=torch.einsum('bni,noi->bno',x,w)+bias[None]
            if j<2: x=x.tanh()
        base_logits=x
        slot=self.slot_embedding.weight[None,None]
        chosen_context=torch.zeros(b,n,16,device=states.device)
        for step in range(t):
            active=step<counts
            logits=base_logits+self.context_head(torch.cat((states[:,:,None].expand(-1,-1,t,-1),slot.expand(b,n,-1,-1),chosen_context[:,:,None].expand(-1,-1,t,-1)),dim=-1)).squeeze(-1)
            if hasattr(self, 'budget_logits'):
                logits=logits+self.budget_logits(counts,step)
            logits=logits.masked_fill(~masks|selected,-torch.inf)
            # inactive rows get a harmless dummy categorical.
            safe=torch.where(active[...,None],logits,torch.full_like(logits,-torch.inf))
            safe[...,0]=torch.where(active,safe[...,0],torch.zeros_like(safe[...,0]))
            dist=Categorical(logits=safe)
            if not replay:
                pick=dist.logits.argmax(-1) if deterministic else dist.sample()
            else:
                pick=order[...,step]
            safe_pick=torch.where(active,pick,torch.zeros_like(pick))
            logp=logp+torch.where(active,dist.log_prob(safe_pick),torch.zeros_like(logp))
            entropy=entropy+torch.where(active,dist.entropy(),torch.zeros_like(entropy))
            bi,ni=active.nonzero(as_tuple=True)
            action[bi,ni,safe_pick[bi,ni]]=True
            selected[bi,ni,safe_pick[bi,ni]]=True
            chosen_context=chosen_context+torch.where(active[...,None],self.slot_embedding(safe_pick),torch.zeros_like(chosen_context))
            order[...,step]=torch.where(active,pick,order[...,step])
        return action,order,counts,logp,entropy/(counts.float()+1)

    def sample(self,states,masks,deterministic=False,object_features=None):
        action,order,counts,logp,_=self._paths(states,masks,deterministic=deterministic)
        return action,action,order,counts,logp,torch.zeros_like(logp),torch.zeros_like(counts,dtype=torch.bool)

    def evaluate_path(self,states,masks,base,drop,add,object_features=None):
        action,_,_,logp,entropy=self._paths(states,masks,order=drop.clone(),counts=add)
        if not torch.equal(action,base):
            raise ValueError('Stored set path does not match the executed action.')
        return logp,torch.zeros_like(logp),entropy,torch.zeros_like(logp)

    def clip_gradients(self,max_norm):
        super().clip_gradients(max_norm)
        nn.utils.clip_grad_norm_(list(self.count_head.parameters())+
                                 list(self.slot_embedding.parameters())+
                                 list(self.context_head.parameters()),max_norm)


class IntentConditionedPolicy(nn.Module):
    """MAPPO actor with a target-wise neighbour-intent prediction pathway.

    The proposal logits retain the original independent MAPPO actors.  A
    zero-initialized residual conditions every target decision on a predicted
    probability that a local ROI neighbour will also broadcast that target.
    """
    def __init__(self,state_dim,num_agents,num_slots,hidden_dims,object_feature_dim=2):
        super().__init__()
        actors=[ActorNetwork(state_dim,num_slots,hidden_dims) for _ in range(num_agents)]
        keys=('shared_net.0','shared_net.2','action_head')
        saved=[a.state_dict() for a in actors]
        self.weights=nn.ParameterList([nn.Parameter(torch.stack([s[k+'.weight'] for s in saved])) for k in keys])
        self.biases=nn.ParameterList([nn.Parameter(torch.stack([s[k+'.bias'] for s in saved])) for k in keys])
        self.num_agents=num_agents; self.num_slots=num_slots; self.object_feature_dim=object_feature_dim
        self.agent_embedding=nn.Embedding(num_agents,8)
        self.slot_embedding=nn.Embedding(num_slots,8)
        # The final layer starts at zero, exactly recovering ordinary MAPPO.
        self.intent_head=nn.Sequential(nn.Linear(hidden_dims[-1]+8+8+object_feature_dim,48),nn.Tanh(),nn.Linear(48,1))
        self.action_residual=nn.Sequential(nn.Linear(hidden_dims[-1]+8+8+object_feature_dim+1,48),nn.Tanh(),nn.Linear(48,1))
        nn.init.zeros_(self.action_residual[-1].weight); nn.init.zeros_(self.action_residual[-1].bias)

    def _features(self,states,object_features=None):
        b,n,_=states.shape; t=self.num_slots
        x=states
        hidden=None
        for j,(w,bias) in enumerate(zip(self.weights,self.biases)):
            x=torch.einsum('bni,noi->bno',x,w)+bias[None]
            if j<2: x=x.tanh()
            if j==1: hidden=x
        base_logits=x
        if object_features is None:
            object_features=torch.zeros(b,n,t,self.object_feature_dim,device=states.device)
        if object_features.shape != (b,n,t,self.object_feature_dim):
            raise ValueError('Object-relation feature shape does not match intent actor.')
        agents=self.agent_embedding(torch.arange(n,device=states.device))[None,:,None].expand(b,-1,t,-1)
        slots=self.slot_embedding(torch.arange(t,device=states.device))[None,None].expand(b,n,-1,-1)
        repeated=hidden[:,:,None].expand(-1,-1,t,-1)
        token=torch.cat((repeated,agents,slots,object_features),dim=-1)
        intent=self.intent_head(token).squeeze(-1)
        residual=self.action_residual(torch.cat((token,torch.sigmoid(intent)[...,None]),dim=-1)).squeeze(-1)
        return base_logits+residual,intent

    def probabilities(self,states,object_features=None):
        logits,_=self._features(states,object_features)
        return logits.sigmoid().clamp(1e-6,1-1e-6)

    def intent_logits(self,states,object_features=None):
        _,intent=self._features(states,object_features)
        return intent

    def sample(self,states,masks,deterministic=False,object_features=None):
        probs=self.probabilities(states,object_features)
        dist=Bernoulli(probs=probs)
        action=((probs>.5) if deterministic else dist.sample().bool())&masks
        logp=(dist.log_prob(action.float())*masks).sum(-1)
        b,n,_=action.shape
        drop=torch.full((b,n),self.num_slots,dtype=torch.long,device=states.device)
        add=torch.zeros_like(drop)
        return action,action,drop,add,logp,torch.zeros_like(logp),torch.zeros_like(drop,dtype=torch.bool)

    def evaluate_path(self,states,masks,base,drop,add,object_features=None):
        probs=self.probabilities(states,object_features)
        dist=Bernoulli(probs=probs)
        logp=(dist.log_prob(base.float())*masks).sum(-1)
        entropy=(dist.entropy()*masks).sum(-1)/masks.sum(-1).clamp_min(1)
        return logp,torch.zeros_like(logp),entropy,torch.zeros_like(logp)

    def clip_gradients(self,max_norm):
        params=list(self.weights)+list(self.biases)
        norms=torch.zeros(self.num_agents,device=self.weights[0].device)
        for p in params:
            if p.grad is not None: norms+=p.grad.flatten(1).pow(2).sum(1)
        scales=(max_norm/(norms.sqrt()+1e-6)).clamp_max(1)
        for p in params:
            if p.grad is not None: p.grad.mul_(scales.reshape((-1,)+(1,)*(p.ndim-1)))
        nn.utils.clip_grad_norm_(list(self.agent_embedding.parameters())+list(self.slot_embedding.parameters())+
                                 list(self.intent_head.parameters())+list(self.action_residual.parameters()),max_norm)

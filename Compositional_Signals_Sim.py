from __future__ import annotations

from enum import Enum, auto
from typing import List

import numpy as np




N_TRIALS: int = 100_000          
REPLACE_AT: int = 50_000         
TEMPERATURE: float = 5.0          
LOG_MI_EVERY: int | None = None  
SEED: int = 42

rng = np.random.default_rng(SEED)


STATES: range = range(4)          
ACTS: range = range(4)            
MSGS_A: List[int] = [0, 1]        
MSGS_B: List[int] = [0, 1]        
NEW_MSG: int = 2                  





def sample_from(weights: np.ndarray) -> int:
    """Draw an index ∝ *weights* (1‑D numpy array)."""
    total = float(weights.sum())
    if total == 0:
        
        return int(rng.integers(len(weights)))
    return int(rng.choice(len(weights), p=weights / total))


def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.exp((x - x.max()) / T)
    return z / z.sum()





class Sender:
    def __init__(self, messages: List[int]):
        
        self.R = np.ones((len(STATES), len(messages)), dtype=float)
        self.messages = messages
        self._m_index = {m: i for i, m in enumerate(messages)}

    def send(self, state: int) -> int:
        idx = sample_from(self.R[state])
        return self.messages[idx]

    def reinforce(self, state: int, msg: int, reward: int) -> None:
        self.R[state, self._m_index[msg]] += reward

    
    def relabel(self, pos: int, new_msg: int) -> None:
        """Relabel the symbol at *pos* with *new_msg* while keeping counts."""
        old_msg = self.messages[pos]
        self.messages[pos] = new_msg
        
        self._m_index[new_msg] = pos
        self._m_index.pop(old_msg, None)





class RecType(Enum):
    CONVENTIONAL = auto()
    MINIMALIST   = auto()
    GEN_ERASE    = auto()
    GEN_KEEP     = auto()


class Receiver:
    """One learner implementing a particular reinforcement rule."""

    def __init__(self, rtype: RecType):
        self.rtype = rtype

        
        self.R_pair = np.ones((len(MSGS_A), len(MSGS_B), len(ACTS)), dtype=float)

        
        self.R_A = np.ones((len(MSGS_A), len(ACTS)), dtype=float)
        self.R_B = np.ones((len(MSGS_B), len(ACTS)), dtype=float)

        
        self._idx_A = {m: i for i, m in enumerate(MSGS_A)}
        self._idx_B = {m: i for i, m in enumerate(MSGS_B)}

        self.total_reward = 0
        self.mi_log: list[float] = []

    
    def _observe_message(self, mA: int, mB: int) -> None:
        """Update *frequency* counts for generalists, independent of reward."""
        if self.rtype in {RecType.GEN_ERASE, RecType.GEN_KEEP}:
            iA, iB = self._idx_A[mA], self._idx_B[mB]
            
            self.R_A[iA]      += 1.0
            self.R_B[iB]      += 1.0
            self.R_pair[iA, iB] += 1.0

    
    def act(self, mA: int, mB: int) -> int:
        
        self._observe_message(mA, mB)

        if self.rtype == RecType.MINIMALIST:
            scores = self.R_A[self._idx_A[mA]] + self.R_B[self._idx_B[mB]]
            probs = softmax(scores, T=TEMPERATURE)
            return sample_from(probs)
        else:  
            probs = self.R_pair[self._idx_A[mA], self._idx_B[mB]]
            return sample_from(probs)

    
    def reinforce(self, mA: int, mB: int, act: int, reward: int) -> None:
        if reward == 0:
            return  

        self.total_reward += 1
        iA, iB = self._idx_A[mA], self._idx_B[mB]

        if self.rtype == RecType.MINIMALIST:
            self.R_A[iA, act] += 1.0
            self.R_B[iB, act] += 1.0

        else:  
            self.R_pair[iA, iB, act] += 1.0
            if self.rtype in {RecType.GEN_ERASE, RecType.GEN_KEEP}:
                self.R_A[iA, act] += 1.0
                self.R_B[iB, act] += 1.0

    
    def register_new_message(self, old_B: int = 0) -> None:
        """Called once, right after *REPLACE_AT*, when Sender B starts
        emitting NEW_MSG instead of *old_B* (default mB0)."""

        
        if NEW_MSG not in self._idx_B:
            
            self.R_B = np.vstack([self.R_B, np.ones((1, len(ACTS)))]).astype(float)
            self._idx_B[NEW_MSG] = self.R_B.shape[0] - 1

            
            extra_col = np.ones((self.R_pair.shape[0], 1, len(ACTS)), dtype=float)
            self.R_pair = np.concatenate([self.R_pair, extra_col], axis=1)

        j_new = self._idx_B[NEW_MSG]
        j_old = self._idx_B[old_B]

        if self.rtype == RecType.GEN_ERASE:
            
            self.R_pair[:, j_new, :] = 1.0
            self.R_B[j_new, :] = 1.0

        elif self.rtype == RecType.GEN_KEEP:
            
            eps = 1e-9
            
            totals_per_act = self.R_A.sum(axis=0)  
            for iA in range(len(MSGS_A)):
                for a in ACTS:
                    joint = (
                        self.R_A[iA, a] * self.R_B[j_old, a] / max(totals_per_act[a], eps)
                    )
                    self.R_pair[iA, j_new, a] = max(joint, 1.0)  
            
            self.R_B[j_new] = self.R_B[j_old]

    
    def log_mutual_information(self) -> None:
        """Append current mutual information I(A;M) in *bits* to *mi_log*."""
        if self.rtype not in {RecType.CONVENTIONAL, RecType.GEN_ERASE, RecType.GEN_KEEP}:
            return

        joint = self.R_pair.sum(axis=2)  
        prob = joint / joint.sum()
        pA = prob.sum(axis=1, keepdims=True)
        pB = prob.sum(axis=0, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            mi_matrix = prob * np.log2(prob / (pA * pB))
        self.mi_log.append(float(np.nansum(mi_matrix)))






def run_game(rtype: RecType, n_trials: int = N_TRIALS) -> float:
    """Run one game with the specified receiver type. Returns mean accuracy."""

    A_sender = Sender(MSGS_A)
    B_sender = Sender(MSGS_B.copy())  
    recv = Receiver(rtype)

    for t in range(n_trials):
        
        if t == REPLACE_AT:
            B_sender.relabel(0, NEW_MSG)  
            recv.register_new_message(old_B=0)

        
        s = int(rng.integers(len(STATES)))

        
        mA = A_sender.send(s)
        mB = B_sender.send(s)

        
        a = recv.act(mA, mB)

        
        r = int(a == s)

        
        A_sender.reinforce(s, mA, r)
        B_sender.reinforce(s, mB, r)
        recv.reinforce(mA, mB, a, r)

        
        if LOG_MI_EVERY and (t + 1) % LOG_MI_EVERY == 0:
            recv.log_mutual_information()

    return recv.total_reward / n_trials  
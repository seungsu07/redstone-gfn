from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rsim
from gfn_env import P

N_TYPES = 9         # 블록 종류 (XOR 과제)
N_TYPES_DOOR = 13   # 블록 종류 (문 과제)
N_DIRS = 6          # 방향 6면
N_POWER = 3         # 전력 평면 수 (레버 off / on / 변화 플래그)

# 입력 채널: 타입 one-hot + 방향 one-hot + scaffold + 타깃 마커 + 진행도
# n_types와 c_in을 맞추지 않으면 XOR시절 모델이 될 수 있으니 주의
C_IN = N_TYPES + N_DIRS + 3
C_IN_DOOR_BASE = N_TYPES_DOOR + N_DIRS + 3
C_IN_DOOR = C_IN_DOOR_BASE + N_POWER

# 인코딩
def encode(types: np.ndarray, datas: np.ndarray, placed: np.ndarray,
           budget: int, task, scaffold: np.ndarray,
           device, n_types: int = N_TYPES,
           power: np.ndarray | None = None) -> torch.Tensor:

    B = types.shape[0]
    sy, sz, sx = task.sy, task.sz, task.sx
    t3 = types.reshape(B, sy, sz, sx)
    d3 = datas.reshape(B, sy, sz, sx)

    ch_scaffold = n_types + N_DIRS
    ch_target = ch_scaffold + 1
    ch_progress = ch_scaffold + 2
    ch_power = ch_scaffold + 3

    c_in = ch_power + (N_POWER if power is not None else 0)
    x = np.zeros((B, c_in, sy, sz, sx), dtype=np.float32)

    for t in range(n_types):
        x[:, t] = t3 == t

    # 방향: 붙는 면 / 출력 방향 / 바라보는 방향. 방향은 전부 하위 3비트에 있고
    # 상위 비트는 리피터 지연·비교기 모드라 마스킹해야 한다.
    attached = (t3 == rsim.BT_TORCH) | (t3 == rsim.BT_LEVER)
    emitting = (t3 == rsim.BT_REPEATER) | (t3 == rsim.BT_COMPARATOR)
    facing = ((t3 == rsim.BT_PISTON) | (t3 == rsim.BT_STICKY) |
              (t3 == rsim.BT_OBSERVER))
    for d in range(N_DIRS):
        x[:, n_types + d] = ((attached & (d3 == d)) |
                             ((emitting | facing) & ((d3 & 7) == d)))

    blocked = scaffold.reshape(sy, sz, sx).astype(np.float32).copy()
    corr = getattr(task, "corr_idx", None)
    if corr:
        blocked.reshape(-1)[list(corr)] = 1.0
    x[:, ch_scaffold] = blocked[None]

    targets = ([task.out_idx] if hasattr(task, "out_idx") else task.open_idx)
    for c in targets:
        tx, ty, tz = task.coords(c)
        x[:, ch_target, ty, tz, tx] = 1.0

    x[:, ch_progress] = (placed / budget)[:, None, None, None]

    if power is not None:
        p = power.reshape(B, N_POWER, sy, sz, sx).astype(np.float32)
        p[:, :2] /= 15.0 # 레드스톤 세기는 0..15, 세 번째는 0/1 플래그
        x[:, ch_power:] = p
    return torch.from_numpy(x).to(device)

# 스캐폴드/통로 마스킹, STOP 확률 계산
def masked_logp(out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = out.float()
    place_mask = mask[:, :-1]
    any_place = place_mask.any(dim=1, keepdim=True) # 합법 공간 없음 -> 예산 소진

    logits = out[:, :-1].masked_fill(~place_mask, -torch.inf)
    logits = torch.where(any_place, logits, torch.zeros_like(logits)) # Softmax NaN 방지

    log_stop = F.logsigmoid(out[:, -1:])
    log_cont = F.logsigmoid(-out[:, -1:]) # log(1 - sigmoid(x))

    lp_place = F.log_softmax(logits, dim=1) + log_cont
    lp_place = torch.where(any_place, lp_place, torch.full_like(lp_place, -torch.inf))
    lp_stop = torch.where(any_place, log_stop, torch.zeros_like(log_stop)) # 예산 소진시
    return torch.cat([lp_place, lp_stop], dim=1)

# 잔차 블록
class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

# 모델
class GFN(nn.Module):
    def __init__(self, width: int = 96, depth: int = 4,
                 c_in: int = C_IN, n_actions: int = P,
                 stop_bias: float = -3.5, effect: bool = False,
                 act_emb: int = 8):
        super().__init__()
        self.stem = nn.Conv3d(c_in, width, 3, padding=1)
        self.tower = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.place_head = nn.Conv3d(width, n_actions, 1)
        # 팔레트 엔트리별 고정 가산 prior(Palette.log_weights). 버퍼로 두면
        # 체크포인트에 같이 실려서 어느 로더가 읽어도 같은 정책이 된다.
        self.register_buffer("act_prior", torch.zeros(n_actions))
        self.stop_head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1))
        nn.init.constant_(self.stop_head[-1].bias, stop_bias)
        self.logZ = nn.Parameter(torch.zeros(()))
        # Effect 헤드
        if effect:
            self.act_emb = nn.Embedding(n_actions, act_emb)
            self.effect_head = nn.Sequential(
                nn.Conv3d(width + 1 + act_emb, width // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv3d(width // 2, N_POWER, 1),
            )
        else:
            self.act_emb = None
            self.effect_head = None

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(self.stem(x))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        place = self.place_head(h)                       # (B, P, sy, sz, sx)
        place = place + self.act_prior.view(1, -1, 1, 1, 1)
        place = place.permute(0, 2, 3, 4, 1).reshape(h.shape[0], -1)
        stop = self.stop_head(h.mean(dim=(2, 3, 4)))     # (B, 1)
        return torch.cat([place, stop], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(self.trunk(x))

    def effect(self, h: torch.Tensor, cell: torch.Tensor,
               pal: torch.Tensor) -> torch.Tensor:
        if self.effect_head is None:
            raise RuntimeError("model built without an effect head")
        B, _, sy, sz, sx = h.shape
        n = sy * sz * sx
        marker = h.new_zeros(B, 1 + self.act_emb.embedding_dim, n)
        idx = cell.view(B, 1, 1).expand(-1, marker.shape[1], -1)
        # bf16 autocast에서 임베딩은 자기 dtype으로 나오는데 scatter_는 목적지와
        # dtype이 정확히 같기를 요구한다
        emb = torch.cat([h.new_ones(B, 1),
                         self.act_emb(pal).to(h.dtype)], dim=1)
        marker.scatter_(2, idx, emb.unsqueeze(2))
        marker = marker.view(B, -1, sy, sz, sx)
        return self.effect_head(torch.cat([h, marker], dim=1))

    # logZ는 학습률 높아야해서 따로 등록
    def split_params(self):
        zs = [self.logZ]
        rest = [p for n, p in self.named_parameters() if n != "logZ"]
        return rest, zs
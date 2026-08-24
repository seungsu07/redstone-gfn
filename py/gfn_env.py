from __future__ import annotations

import numpy as np

import rsim
from rsim import (BT_AIR, BT_SOLID, BT_GLASS, BT_DUST, BT_TORCH, BT_REPEATER,
                  BT_RBLOCK, BT_COMPARATOR,
                  BT_PISTON, BT_STICKY, BT_OBSERVER,
                  DOWN, UP, NORTH, SOUTH, WEST, EAST)

# 꽉 찬 큐브: dust가 위에 앉고 토치/레버가 옆에 붙는다.
# rs_core.c의 is_full()과 반드시 같아야 한다 (옵저버 포함).
FULL_TYPES = (BT_SOLID, BT_GLASS, BT_RBLOCK, BT_OBSERVER)
# 지지 없이 서는 블록: 면만 닿으면 빈 칸 어디든 합법
FREE_TYPES = (BT_SOLID, BT_GLASS, BT_RBLOCK, BT_PISTON, BT_STICKY, BT_OBSERVER)

# 배치 규칙. 팔레트 51개 엔트리가 답해야 할 합법성 검사는 실제로 7가지뿐이다.
# 엔트리마다 마스크를 만들던 옛 방식은 `base & below_full`을 스무 번, 토치 방향마다
# `shifted(full, d)`를 매 스텝 다시 계산해서 학습 시간의 6분의 1을 먹었다.
RULE_FREE = 0        # 자립: 면이 닿은 빈 칸이면 된다
RULE_BELOW = 1       # 아래에 꽉 찬 블록이 필요
RULE_TORCH = 2       # + 방향 인덱스: 그 면에 꽉 찬 블록이 필요
TORCH_DIRS = (DOWN, NORTH, SOUTH, WEST, EAST)
N_RULES = RULE_TORCH + len(TORCH_DIRS)


# 엔트리 하나가 답하는 합법성 규칙
def _rule_of(bt: int, bd: int) -> int:
    if bt in FREE_TYPES:
        return RULE_FREE
    if bt in (BT_DUST, BT_REPEATER, BT_COMPARATOR):
        return RULE_BELOW
    if bt == BT_TORCH:
        return RULE_TORCH + TORCH_DIRS.index(bd)
    return -1


# 과제의 배치 어휘 = C ACTIONS 표의 인덱스 묶음.
# 하드코딩한 인덱스가 아니라 (type, data)로 찾으므로 C쪽 팔레트가 늘어나도
# 매핑이 조용히 밀리지 않는다.
class Palette:
    def __init__(self, want: list[tuple[int, int]]):
        idx = []
        for t, d in want:
            hits = np.flatnonzero((rsim.ACTION_TYPES == t) & (rsim.ACTION_DATAS == d))
            if len(hits) != 1:
                raise RuntimeError(f"palette entry (type={t}, data={d}) not in C table")
            idx.append(int(hits[0]))
        self.indices = np.array(idx, dtype=np.int64)
        self.P = len(idx)
        self.types = rsim.ACTION_TYPES[self.indices]
        self.datas = rsim.ACTION_DATAS[self.indices]
        # (type, data) -> 팔레트 인덱스. 이미 놓인 블록을 "그걸 놓았을 액션"으로
        # 되돌리는 데 쓴다 (리플레이가 필요로 한다)
        self.lookup = {(int(self.types[p]), int(self.datas[p])): p
                       for p in range(self.P)}
        # 부품 '타입' 단위로 공평하게 만드는 엔트리별 가중치. 액션 공간이 평평해서
        # 피스톤은 12장(6방향 x 2종), 리피터는 16장(4방향 x 4지연), dust는 1장을
        # 들고 있었고, 학습 전 정책과 epsilon 탐색기가 파라미터화 때문에 피스톤을
        # dust보다 10배 자주 뽑았다. 1/변종수로 나누면 총 질량이 같아진다.
        counts = {int(t): int(np.count_nonzero(self.types == t))
                  for t in np.unique(self.types)}
        self.weights = np.array([1.0 / counts[int(t)] for t in self.types],
                                dtype=np.float64)
        self.log_weights = np.log(self.weights)

        self.rule = np.array([_rule_of(int(self.types[p]), int(self.datas[p]))
                              for p in range(self.P)], dtype=np.int64)
        if (self.rule < 0).any():
            bad = [(int(self.types[p]), int(self.datas[p]))
                   for p in np.flatnonzero(self.rule < 0)]
            raise RuntimeError(f"palette entries with no placement rule: {bad}")


# 게이트(XOR) 어휘. 일부러 뺀 것: AIR(고정 순서 시절의 잔재), RBLOCK(레버만이
# 유일한 전원이어야 반응성 측정이 의미를 갖는다), 리피터 지연 2-4(게이트 과제에선
# worst_delay만 나빠졌다).
def _gate_entries():
    want = [(BT_SOLID, 0), (BT_GLASS, 0), (BT_DUST, 0)]
    want += [(BT_TORCH, d) for d in (DOWN, NORTH, SOUTH, WEST, EAST)]
    want += [(BT_REPEATER, rsim.repeater(d, 1)) for d in (NORTH, SOUTH, WEST, EAST)]
    # 비교기: 토치만으로는 정확히 한 회로 계열만 3/4에 닿고 그 2편집 이내에
    # 유효 설계가 없다는 것이 전수 확인됐다(모드 붕괴 부검). 빼기 모드 비교기가
    # 마지막 사다리 칸을 매끄럽게 만든다 — 하나면 이미 3/4, 둘을 합치면 XOR.
    want += [(BT_COMPARATOR, rsim.comparator(d, subtract=True))
             for d in (NORTH, SOUTH, WEST, EAST)]
    want += [(BT_COMPARATOR, rsim.comparator(d, subtract=False))
             for d in (NORTH, SOUTH, WEST, EAST)]
    return want


# 문 어휘 = 게이트 어휘 + 블록을 움직이는 것 전부.
# RBLOCK 허용(문은 전원을 실제로 실어 나른다), 리피터 지연 2-4 포함(타이밍 사슬이
# 문의 순서를 만든다), 피스톤·옵저버는 6방향 전부.
def _door_entries():
    want = _gate_entries()
    want += [(BT_RBLOCK, 0)]
    want += [(BT_REPEATER, rsim.repeater(d, dl))
             for d in (NORTH, SOUTH, WEST, EAST) for dl in (2, 3, 4)]
    all_dirs = (DOWN, UP, NORTH, SOUTH, WEST, EAST)
    want += [(BT_PISTON, d) for d in all_dirs]
    want += [(BT_STICKY, d) for d in all_dirs]
    want += [(BT_OBSERVER, d) for d in all_dirs]
    return want


GATE_PALETTE = Palette(_gate_entries())
DOOR_PALETTE = Palette(_door_entries())

# 옛 이름 별칭. 게이트 팔레트에 묶어 두어야 예전 체크포인트가 그대로 로드된다.
PALETTE = GATE_PALETTE.indices
P = GATE_PALETTE.P
PAL_TYPES = GATE_PALETTE.types
PAL_DATAS = GATE_PALETTE.datas
PAL_LOOKUP = GATE_PALETTE.lookup


# 문 과제의 통로 = 아무것도 지을 수 없는 칸 (n,) bool. 통로가 없으면 None.
def corridor_mask(task) -> np.ndarray | None:
    corr = getattr(task, "corr_idx", None)
    if not corr:
        return None
    banned = np.zeros(task.n, dtype=bool)
    banned[list(corr)] = True
    return banned


# out[..., cell] = a[..., cell + 방향 d], 경계는 0으로. 배열은 (B, sy, sz, sx)라
# 축 1..3이 각각 y, z, x다.
def shifted(a: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros_like(a)
    if d == DOWN:    out[:, 1:]        = a[:, :-1]
    elif d == UP:    out[:, :-1]       = a[:, 1:]
    elif d == NORTH: out[:, :, 1:]     = a[:, :, :-1]
    elif d == SOUTH: out[:, :, :-1]    = a[:, :, 1:]
    elif d == WEST:  out[:, :, :, 1:]  = a[:, :, :, :-1]
    elif d == EAST:  out[:, :, :, :-1] = a[:, :, :, 1:]
    return out


# (B, n*P + 1) 합법 마스크. 메서드가 아니라 모듈 함수인 이유: TB 업데이트가 이미
# 지나간 에피소드 중간 상태의 마스크를 나중에 다시 계산해야 한다.
# banned: 절대 지을 수 없는 칸 (문 통로). scaffold는 넣을 필요가 없다 — 비어 있지
# 않으므로 어차피 배치 대상이 아니다.
def legal_mask_arrays(task, types: np.ndarray,
                      placed: np.ndarray, budget: int,
                      palette: Palette = GATE_PALETTE,
                      banned: np.ndarray | None = None) -> np.ndarray:
    B = types.shape[0]
    t3 = types.reshape(B, task.sy, task.sz, task.sx)
    empty = t3 == BT_AIR
    full = np.isin(t3, FULL_TYPES)
    below_full = shifted(full, DOWN)

    # 생존 규칙: 새 블록은 기존 면에 닿아야 한다. 이웃이 전부 공기인 칸은 어차피
    # 어떤 회로에도 영향을 못 준다 — 여기의 모든 전력 상호작용이 면을 통한다.
    nonair = t3 != BT_AIR
    touch = np.zeros_like(nonair)
    for d in (DOWN, UP, NORTH, SOUTH, WEST, EAST):
        touch |= shifted(nonair, d)
    base = empty & touch
    if banned is not None:
        base = base & ~banned.reshape(1, task.sy, task.sz, task.sx)

    Pn = palette.P
    rules = np.empty((B, task.n, N_RULES), dtype=bool)
    rules[:, :, RULE_FREE] = base.reshape(B, task.n)
    rules[:, :, RULE_BELOW] = (base & below_full).reshape(B, task.n)
    for i, d in enumerate(TORCH_DIRS):
        rules[:, :, RULE_TORCH + i] = (base & shifted(full, d)).reshape(B, task.n)

    # 마지막 축을 한 번 gather해서 답 7개를 액션 id 배치(cell*Pn + p) 그대로
    # 51개 엔트리로 펼친다. 엔트리별 루프 대비 큰 tb_update 청크의 마스크 비용이
    # 절반으로 줄었고, 마스크가 실제로 아팠던 곳이 거기다.
    m = rules[:, :, palette.rule].reshape(B, task.n * Pn)
    m[placed >= budget] = False   # 예산 소진 -> STOP 강제
    return np.concatenate([m, np.ones((B, 1), dtype=bool)], axis=1)


# (B, n) 정확한 부모 마스크: 떼어내도 도달 가능한 상태가 남는 블록.
# 아무것도 그 위에 얹히거나 매달려 있지 않고, 남은 구조가 scaffold와 면으로
# 계속 연결돼 있어야 한다 (배치가 면 접촉을 요구하므로, 떠 있는 덩어리가 남는
# 선행 상태는 존재할 수 없다). 연결성 판정은 numpy로 벡터화가 안 돼서 C 플러드필에 맡긴다.
def removable_mask_arrays(task: rsim.Task, types: np.ndarray, datas: np.ndarray,
                          scaffold: np.ndarray) -> np.ndarray:
    return rsim.parents_mask(task, types, datas, scaffold).astype(bool)


# 과제 하나 위의 벡터화된 빌드 에피소드
class Env:
    def __init__(self, task, batch: int, budget: int = 64,
                 palette: Palette = GATE_PALETTE):
        self.task = task
        self.B = batch
        self.budget = budget
        self.palette = palette
        self.stop_action = task.n * palette.P

        self.scaffold = np.zeros(task.n, dtype=bool)
        self.scaffold[list(task.scaffold)] = True
        self.banned = corridor_mask(task)

        self.types = self.datas = None
        self.placed = self.done = None
        self.reset()

    def reset(self):
        self.types, self.datas = self.task.blank(self.B)
        self.placed = np.zeros(self.B, dtype=np.int64)
        self.done = np.zeros(self.B, dtype=bool)

    def grids(self):
        t = self.task
        return (self.types.reshape(self.B, t.sy, t.sz, t.sx),
                self.datas.reshape(self.B, t.sy, t.sz, t.sx))

    # (B, n*P + 1). STOP은 항상 합법, 배치는 지지 규칙·예산·통로 금지를 따른다.
    def legal_mask(self) -> np.ndarray:
        return legal_mask_arrays(self.task, self.types, self.placed, self.budget,
                                 palette=self.palette, banned=self.banned)

    # (B,) 유효 부모 수 = 놓인 것 중 scaffold가 아니고 하중을 받지 않는 블록 수.
    # 도달 가능한 비초기 상태에는 항상 최소 하나가 있다.
    def n_parents(self) -> np.ndarray:
        return removable_mask_arrays(self.task, self.types, self.datas,
                                     self.scaffold).sum(axis=1)

    # active인 행마다 액션 하나를 적용하고, 이번 스텝에 블록을 놓은 행을 돌려준다
    # (호출자가 P_B 누적에 쓴다).
    def step(self, actions: np.ndarray, active: np.ndarray) -> np.ndarray:
        placing = active & (actions != self.stop_action)
        stopping = active & (actions == self.stop_action)

        rows = np.flatnonzero(placing)
        if len(rows):
            cells = actions[rows] // self.palette.P
            pals = actions[rows] % self.palette.P
            self.types[rows, cells] = self.palette.types[pals]
            self.datas[rows, cells] = self.palette.datas[pals]
            self.placed[rows] += 1
        self.done[stopping] = True
        return placing

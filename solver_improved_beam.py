import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm


BUDGET = 10_000
COST_PER_DEGREE = 300
INCOME_PER_VIRAL = 50
MAX_DAYS = 60
MAX_CONTRACTS_PER_DAY = 10


def load_graph(path: str) -> Dict[int, List[int]]:
    graph = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            u, v = map(int, line.split())
            graph[u].append(v)
            graph[v].append(u)
    return dict(graph)


def build_submission(plan: Dict[int, List[int]], out_path: str, sample_path: str | None = None) -> None:
    if sample_path:
        try:
            sub = pd.read_csv(sample_path)
            if list(sub.columns) != ["day", "node_ids"] or len(sub) != MAX_DAYS:
                raise ValueError("Bad sample format")
            sub["node_ids"] = "-1"
        except Exception:
            sub = pd.DataFrame({"day": range(MAX_DAYS), "node_ids": ["-1"] * MAX_DAYS})
    else:
        sub = pd.DataFrame({"day": range(MAX_DAYS), "node_ids": ["-1"] * MAX_DAYS})

    for day in range(MAX_DAYS):
        ids = plan.get(day, [])
        sub.loc[sub["day"] == day, "node_ids"] = " ".join(map(str, ids)) if ids else "-1"

    sub.to_csv(out_path, index=False)


@dataclass
class CandidateEval:
    node: int
    cost: int
    score: float
    expected_profit: float
    true_profit: int
    true_extra: int
    accelerated: int
    roi: float


@dataclass
class BeamState:
    active: Set[int]
    anc: Dict[int, int]
    budget: int
    bought: List[int]
    score_sum: float
    expected_profit_sum: float


class CampaignSolver:
    def __init__(
        self,
        graph: Dict[int, List[int]],
        candidate_pool: int = 350,
        branch_k: int = 30,
        beam_width: int = 40,
        beam_depth: int = 3,
        time_bonus: float = 0.45,
        acceleration_bonus: float = 0.12,
        min_true_profit: int = -600,
        rng_seed: int = 42,
        randomized: bool = False,
        random_top: int = 12,
        temperature: float = 0.35,
    ):
        self.graph = graph
        self.nodes = list(graph.keys())
        self.deg = {v: len(graph[v]) for v in self.nodes}
        self.thr = {v: math.ceil(0.18 * self.deg[v]) for v in self.nodes}

        self.candidate_pool = candidate_pool
        self.branch_k = branch_k
        self.beam_width = beam_width
        self.beam_depth = beam_depth

        self.time_bonus = time_bonus
        self.acceleration_bonus = acceleration_bonus
        self.min_true_profit = min_true_profit

        self.rng = random.Random(rng_seed)
        self.randomized = randomized
        self.random_top = random_top
        self.temperature = temperature

    def contract_cost(self, v: int) -> int:
        return COST_PER_DEGREE * self.deg[v]

    def activate_morning(
        self,
        active: Set[int],
        anc: Dict[int, int],
    ) -> List[int]:
        """Один реальный дневной шаг вирусного распространения перед покупками."""
        new_active = [
            v for v in self.nodes
            if v not in active and anc[v] >= self.thr[v]
        ]

        for v in new_active:
            active.add(v)
            for u in self.graph[v]:
                anc[u] += 1

        return new_active

    def simulate_future_timed(
        self,
        active_now: Set[int],
        anc_now: Dict[int, int],
        current_day: int,
    ) -> Dict[int, int]:
        """
        Time-limited look-ahead.

        В отличие от closure "до упора", тут учитывается горизонт 60 дней.
        Возвращает только будущие вирусные активации: node -> day.
        Уже активные/купленные узлы в словарь не попадают.
        """
        active = set(active_now)
        anc = anc_now.copy()
        activation_day: Dict[int, int] = {}

        # Покупки сделаны в current_day после утренней волны.
        # Их влияние может вызвать новых вирусных пользователей только утром следующего дня.
        for day in range(current_day + 1, MAX_DAYS):
            new_active = [
                v for v in self.nodes
                if v not in active and anc[v] >= self.thr[v]
            ]

            if not new_active:
                break

            for v in new_active:
                active.add(v)
                activation_day[v] = day

            for v in new_active:
                for u in self.graph[v]:
                    anc[u] += 1

        return activation_day

    def pre_score(
        self,
        v: int,
        active: Set[int],
        anc: Dict[int, int],
        budget: int,
    ) -> float:
        """
        Дешевая предварительная оценка.
        Она нужна только для отбора top-K кандидатов перед дорогой симуляцией.
        """
        cost = self.contract_cost(v)
        if cost > budget or v in active:
            return float("-inf")

        direct = 0.0
        two_hop = 0.0

        for u in self.graph[v]:
            if u in active:
                continue

            need = self.thr[u] - anc[u]
            if need <= 0:
                direct += 4.0
            else:
                direct += 1.0 / (need * need)

            # Легкий 2-hop frontier pressure.
            # Ограничиваемся локальной суммой без тяжелой симуляции.
            for w in self.graph[u][:30]:
                if w in active:
                    continue
                need2 = self.thr[w] - anc[w]
                if need2 > 0:
                    two_hop += 0.08 / (need2 * need2)

        # Дешевые low-degree триггеры получают бонус,
        # но дорогие узлы не выкидываются полностью.
        return (direct + two_hop) * 1000.0 / max(cost, 1) + 0.03 * self.deg[v]

    def candidate_pool_nodes(
        self,
        active: Set[int],
        anc: Dict[int, int],
        budget: int,
        purchased: Set[int],
    ) -> List[int]:
        raw = [
            v for v in self.nodes
            if v not in active
            and v not in purchased
            and self.contract_cost(v) <= budget
        ]

        if len(raw) <= self.candidate_pool:
            return raw

        scored = [
            (self.pre_score(v, active, anc, budget), v)
            for v in raw
        ]
        scored.sort(reverse=True)

        return [v for _, v in scored[: self.candidate_pool]]

    def evaluate_candidate(
        self,
        candidate: int,
        active: Set[int],
        anc: Dict[int, int],
        base_future_day: Dict[int, int],
        current_day: int,
    ) -> CandidateEval:
        cost = self.contract_cost(candidate)

        temp_active = set(active)
        temp_anc = anc.copy()

        temp_active.add(candidate)
        for u in self.graph[candidate]:
            temp_anc[u] += 1

        cand_future_day = self.simulate_future_timed(temp_active, temp_anc, current_day)

        true_extra = 0
        accelerated = 0
        weighted_revenue = 0.0

        for node, new_day in cand_future_day.items():
            old_day = base_future_day.get(node)

            if old_day is None:
                true_extra += 1

                # Ранний пользователь ценнее, потому что дает деньги для будущих покупок.
                remaining = MAX_DAYS - new_day
                weighted_revenue += INCOME_PER_VIRAL * (
                    1.0 + self.time_bonus * remaining / MAX_DAYS
                )
            elif new_day < old_day:
                accelerated += 1

                # Это не добавляет финального пользователя напрямую,
                # но помогает реинвестированию и может ускорить следующие каскады.
                delta = old_day - new_day
                weighted_revenue += (
                    INCOME_PER_VIRAL
                    * self.acceleration_bonus
                    * delta
                    / MAX_DAYS
                )

        true_profit = true_extra * INCOME_PER_VIRAL - cost
        expected_profit = weighted_revenue - cost
        roi = expected_profit / cost if cost else 0.0

        # В начале важнее ROI, в конце — абсолютная прибыль.
        t = current_day / max(1, MAX_DAYS - 1)
        gamma = 1.10 * (1.0 - t) + 0.25 * t

        if expected_profit <= 0:
            score = expected_profit
        else:
            score = expected_profit * ((1.0 + max(roi, 0.0)) ** gamma)

        return CandidateEval(
            node=candidate,
            cost=cost,
            score=score,
            expected_profit=expected_profit,
            true_profit=true_profit,
            true_extra=true_extra,
            accelerated=accelerated,
            roi=roi,
        )

    def best_expansions(
        self,
        state: BeamState,
        purchased: Set[int],
        current_day: int,
    ) -> List[CandidateEval]:
        base_future_day = self.simulate_future_timed(state.active, state.anc, current_day)

        pool = self.candidate_pool_nodes(
            state.active,
            state.anc,
            state.budget,
            purchased | set(state.bought),
        )

        evals = [
            self.evaluate_candidate(v, state.active, state.anc, base_future_day, current_day)
            for v in pool
        ]

        # Важное отличие от исходной версии:
        # не требуем обязательно положительный true_profit.
        # Иногда узел убыточен как single, но ускоряет каскад и полезен в комбинации.
        evals = [
            e for e in evals
            if e.expected_profit > 0 and e.true_profit >= self.min_true_profit
        ]

        evals.sort(key=lambda x: x.score, reverse=True)

        if self.randomized and evals:
            return self.sample_top(evals, self.branch_k)

        return evals[: self.branch_k]

    def sample_top(self, evals: List[CandidateEval], k: int) -> List[CandidateEval]:
        """Softmax sampling из top-N, чтобы уйти от детерминированной жадности."""
        top = evals[: max(k, self.random_top)]
        if len(top) <= k:
            return top

        scores = [e.score for e in top]
        mx = max(scores)
        temp = max(self.temperature, 1e-9)
        weights = [math.exp((s - mx) / (abs(mx) * temp + 1e-9)) for s in scores]

        picked = []
        available = list(top)
        available_weights = list(weights)

        while available and len(picked) < k:
            total = sum(available_weights)
            r = self.rng.random() * total
            acc = 0.0
            idx = 0
            for i, w in enumerate(available_weights):
                acc += w
                if acc >= r:
                    idx = i
                    break

            picked.append(available.pop(idx))
            available_weights.pop(idx)

        return picked

    def apply_contract_to_state(
        self,
        state: BeamState,
        candidate_eval: CandidateEval,
    ) -> BeamState:
        v = candidate_eval.node

        new_active = set(state.active)
        new_anc = state.anc.copy()

        new_active.add(v)
        for u in self.graph[v]:
            new_anc[u] += 1

        return BeamState(
            active=new_active,
            anc=new_anc,
            budget=state.budget - candidate_eval.cost,
            bought=state.bought + [v],
            score_sum=state.score_sum + candidate_eval.score,
            expected_profit_sum=state.expected_profit_sum + candidate_eval.expected_profit,
        )

    def choose_purchases_beam(
        self,
        active: Set[int],
        anc: Dict[int, int],
        budget: int,
        purchased: Set[int],
        current_day: int,
    ) -> List[int]:
        """
        Beam search внутри одного дня.
        Ищет не только лучший single, но и хорошие комбинации из 2-3+ покупок.
        """
        initial = BeamState(
            active=set(active),
            anc=anc.copy(),
            budget=budget,
            bought=[],
            score_sum=0.0,
            expected_profit_sum=0.0,
        )

        beam = [initial]
        best = initial

        max_depth = min(self.beam_depth, MAX_CONTRACTS_PER_DAY)

        for _depth in range(max_depth):
            next_states: List[BeamState] = []

            for state in beam:
                expansions = self.best_expansions(state, purchased, current_day)

                for e in expansions:
                    if e.cost > state.budget:
                        continue
                    next_states.append(self.apply_contract_to_state(state, e))

            if not next_states:
                break

            # Сохраняем разнообразие: сортируем по суммарному score,
            # но состояние без покупки тоже остается допустимым через best.
            next_states.sort(
                key=lambda s: (s.score_sum, s.expected_profit_sum, len(s.bought)),
                reverse=True,
            )
            beam = next_states[: self.beam_width]

            if beam[0].score_sum > best.score_sum:
                best = beam[0]

        return best.bought[:MAX_CONTRACTS_PER_DAY]

    def solve_once(self, verbose: bool = True) -> Tuple[Dict[int, List[int]], int, int, int]:
        active: Set[int] = set()
        purchased: Set[int] = set()
        viral: Set[int] = set()
        anc = defaultdict(int)

        budget = BUDGET
        total_spent = 0
        plan: Dict[int, List[int]] = defaultdict(list)

        iterator = range(MAX_DAYS)
        if verbose:
            iterator = tqdm(iterator, total=MAX_DAYS, desc="campaign")

        for day in iterator:
            # Утреннее вирусное распространение и реинвестирование.
            new_viral = self.activate_morning(active, anc)
            for v in new_viral:
                viral.add(v)
            budget += INCOME_PER_VIRAL * len(new_viral)

            # Поздно в кампании покупаем осторожнее: меньше глубина,
            # потому что каскад может не успеть раскрыться.
            old_depth = self.beam_depth
            if day >= 45:
                self.beam_depth = min(self.beam_depth, 2)

            buys = self.choose_purchases_beam(active, anc, budget, purchased, day)

            self.beam_depth = old_depth

            for v in buys:
                if v in active or v in purchased:
                    continue
                cost = self.contract_cost(v)
                if cost > budget:
                    continue

                budget -= cost
                total_spent += cost
                purchased.add(v)
                active.add(v)
                plan[day].append(v)

                for u in self.graph[v]:
                    anc[u] += 1

            if verbose and plan.get(day):
                tqdm.write(
                    f"[day {day}] buy={plan[day]} "
                    f"spent_today={sum(self.contract_cost(v) for v in plan[day])} "
                    f"budget={budget}"
                )

        profit = len(viral) * INCOME_PER_VIRAL - total_spent
        return dict(plan), profit, len(viral), total_spent

    def simulate_plan(self, plan: Dict[int, List[int]], verbose: bool = False) -> Tuple[int, int, int]:
        active: Set[int] = set()
        viral: Set[int] = set()
        purchased: Set[int] = set()
        anc = defaultdict(int)

        budget = BUDGET
        total_spent = 0

        for day in range(MAX_DAYS):
            new_viral = self.activate_morning(active, anc)
            for v in new_viral:
                viral.add(v)
            budget += INCOME_PER_VIRAL * len(new_viral)

            buys = plan.get(day, [])
            if len(buys) > MAX_CONTRACTS_PER_DAY:
                raise ValueError(f"More than {MAX_CONTRACTS_PER_DAY} contracts on day {day}")

            for v in buys:
                if v not in self.graph:
                    raise ValueError(f"Unknown node {v}")
                if v in purchased:
                    raise ValueError(f"Duplicate purchase: {v}")
                if v in active:
                    # Невыгодно и обычно бессмысленно, но не валим запуск.
                    continue

                cost = self.contract_cost(v)
                if cost > budget:
                    raise ValueError(
                        f"Budget violation on day {day}: node={v}, cost={cost}, budget={budget}"
                    )

                budget -= cost
                total_spent += cost
                purchased.add(v)
                active.add(v)

                for u in self.graph[v]:
                    anc[u] += 1

            if verbose:
                print(
                    day,
                    "new_viral=", len(new_viral),
                    "buys=", buys,
                    "budget=", budget,
                    "profit=", len(viral) * INCOME_PER_VIRAL - total_spent,
                )

        return len(viral) * INCOME_PER_VIRAL - total_spent, len(viral), total_spent


def normalize_plan(plan: Dict[int, List[int]]) -> Dict[int, List[int]]:
    return {d: ids for d, ids in plan.items() if ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="marketing_edges.txt")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--out", default="submission_improved.csv")

    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--candidate-pool", type=int, default=350)
    parser.add_argument("--branch-k", type=int, default=30)
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--beam-depth", type=int, default=3)

    parser.add_argument("--time-bonus", type=float, default=0.45)
    parser.add_argument("--acceleration-bonus", type=float, default=0.12)
    parser.add_argument("--min-true-profit", type=int, default=-600)

    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--verbose-plan", action="store_true")

    args = parser.parse_args()

    graph = load_graph(args.edges)

    best_plan = None
    best_profit = -10**18
    best_viral = 0
    best_spent = 0

    for run in range(args.runs):
        solver = CampaignSolver(
            graph=graph,
            candidate_pool=args.candidate_pool,
            branch_k=args.branch_k,
            beam_width=args.beam_width,
            beam_depth=args.beam_depth,
            time_bonus=args.time_bonus,
            acceleration_bonus=args.acceleration_bonus,
            min_true_profit=args.min_true_profit,
            rng_seed=args.seed + run,
            randomized=args.randomized or args.runs > 1,
            temperature=args.temperature,
        )

        plan, approx_profit, _approx_viral, _approx_spent = solver.solve_once(
            verbose=args.runs == 1
        )

        # Обязательная честная проверка итогового плана обычной симуляцией.
        checked_profit, checked_viral, checked_spent = solver.simulate_plan(plan)

        print(
            f"run={run} profit={checked_profit} "
            f"viral={checked_viral} spent={checked_spent}"
        )

        if checked_profit > best_profit:
            best_profit = checked_profit
            best_plan = normalize_plan(plan)
            best_viral = checked_viral
            best_spent = checked_spent

    assert best_plan is not None

    build_submission(best_plan, args.out, args.sample)

    print("=" * 60)
    print(f"BEST PROFIT: {best_profit}")
    print(f"viral users: {best_viral}")
    print(f"spent: {best_spent}")
    print(f"saved: {args.out}")
    print("=" * 60)

    if args.verbose_plan:
        for day in range(MAX_DAYS):
            if best_plan.get(day):
                print(day, best_plan[day])


if __name__ == "__main__":
    main()

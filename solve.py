from collections import defaultdict
import argparse
import math
import pandas as pd


def load_graph(path):
    graph = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            graph[u].add(v)
            graph[v].add(u)
    return graph


# Fixed reinvestment strategy.
# Cost(node) = 300 * degree(node)
SCHEDULE = {
    0: [3057, 3775, 2528, 2788, 154],
    12: [1304],
    16: [2415],
    22: [1398],
}


def simulate(graph, schedule, days=60, initial_budget=10000):
    deg = {v: len(graph[v]) for v in graph}
    thr = {v: math.ceil(0.18 * deg[v]) for v in graph}

    active = set()
    contracted = set()

    budget = initial_budget
    total_income = 0
    total_cost = 0

    log = []

    for day in range(days):
        buys = schedule.get(day, [])

        if len(buys) > 10:
            raise ValueError(f'More than 10 contracts on day {day}')

        start_budget = budget
        day_cost = 0

        for v in buys:
            if v not in graph:
                raise ValueError(f'Unknown node {v}')

            if v in contracted:
                raise ValueError(f'Duplicate contract for node {v}')

            c = 300 * deg[v]

            if budget < c:
                raise ValueError(
                    f'Budget violation on day {day}: node={v}, cost={c}, budget={budget}'
                )

            budget -= c
            total_cost += c
            day_cost += c

            active.add(v)
            contracted.add(v)

        new_viral = []

        for v in graph:
            if v in active:
                continue

            active_neighbors = sum((u in active) for u in graph[v])

            if active_neighbors >= thr[v]:
                new_viral.append(v)

        for v in new_viral:
            active.add(v)

        income_today = 50 * len(new_viral)
        total_income += income_today
        budget += income_today

        log.append({
            'day': day,
            'contracts': ' '.join(map(str, buys)) if buys else '-1',
            'cost_today': day_cost,
            'viral_new': len(new_viral),
            'income_today': income_today,
            'budget_end': budget,
            'profit': total_income - total_cost,
        })

    return log, total_income - total_cost


def build_submission(schedule, out_path, sample_path=None):
    if sample_path:
        try:
            df = pd.read_csv(sample_path)
            if list(df.columns) != ['day', 'node_ids']:
                raise ValueError
            if len(df) != 60:
                raise ValueError
            sub = df.copy()
            sub['node_ids'] = '-1'
        except Exception:
            sub = pd.DataFrame({
                'day': range(60),
                'node_ids': ['-1'] * 60,
            })
    else:
        sub = pd.DataFrame({
            'day': range(60),
            'node_ids': ['-1'] * 60,
        })

    for day, ids in schedule.items():
        sub.loc[sub['day'] == day, 'node_ids'] = ' '.join(map(str, ids))

    sub.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', required=True)
    parser.add_argument('--sample', default=None)
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--print-log', action='store_true')

    args = parser.parse_args()

    graph = load_graph(args.edges)

    log, profit = simulate(graph, SCHEDULE)

    build_submission(SCHEDULE, args.out, args.sample)

    print(f'Profit: {profit}')
    print(f'Saved submission to: {args.out}')

    if args.print_log:
        for row in log:
            print(row)

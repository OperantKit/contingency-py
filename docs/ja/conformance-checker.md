# 自作の実装を fixture で検算する

[English](../en/conformance-checker.md)

contingency-py を入れると（[README](../../README.ja.md) の「インストール」を参照）、適合性 fixture と `contingency-conformance` コマンドも入ります。強化スケジュールの実装であれば、どの言語で書いたものでも、同じ 39 個の fixture に照らして検算できます。

```bash
contingency-conformance list                    # fixture・種別・スケジュール型・step 数
contingency-conformance run --only 'atomic/fr_*' -- python my_adapter.py
```

`contingency-conformance path` は fixture のディレクトリを表示するので、他言語の検査器はそこから JSON を直接読めます。`CONTINGENCY_CONFORMANCE_DIR` か `--dir` で別の写しを指定できます。

## 観測 JSON

実装は fixture ごとに、fixture の step 1 つにつき 1 要素を返します。

```json
{"fixture": "atomic/fr_basic.json",
 "steps": [{"reinforced": false},
           {"reinforced": true, "reinforcer": {"time": 5.0, "magnitude": 1.0, "label": "SR+"}}]}
```

`reinforced` は完全に一致する必要があります。強化された step では `reinforcer.time` と `magnitude` の差が 1e-9 以内で、`label` が等しいこと。fixture の `meta` にあるキーはすべて同じ値で存在すること（観測側に余分なキーがあってもかまいません）。step 数も一致すること。このファイルを自分で書いて `contingency-conformance check observed/*.json` で比べるか、`run` にアダプタを起動させます。

## `run -- CMD` 用のアダプタ

`run` は fixture ごとに CMD を 1 回起動し、fixture の JSON を標準入力に渡して、標準出力の観測 JSON を受け取ります（`fixture` は省略できます）。

```python
# my_adapter.py — 手書きの FR スケジュールを fixture で検算する
import json
import sys


class MyFR:
    def __init__(self, n):
        self.n, self.count = n, 0

    def step(self, now, event):  # event: None か {"time", "operandum"}
        if event is None:
            return None
        self.count += 1
        if self.count < self.n:
            return None
        self.count = 0
        return {"time": event["time"], "magnitude": 1.0, "label": "SR+"}


def observe(fixture):
    schedule = MyFR(fixture["schedule"]["params"]["n"])
    steps = []
    for step in fixture["steps"]:
        sr = schedule.step(step["now"], step["event"])
        steps.append({"reinforced": True, "reinforcer": sr} if sr else {"reinforced": False})
    return {"implementation": "my-fr", "steps": steps}


if __name__ == "__main__":
    json.dump(observe(json.load(sys.stdin)), sys.stdout)
```

このアダプタは FR だけを実装しているので、合格するのは `atomic/fr_basic.json` だけです。`--only` を外すと、他の 38 本はアダプタがスケジュールを組み立てられず `ERROR` になります。他も通すには `observe` で各スケジュールの `type` を組み立てます。複合スケジュールの 12 本（Alternative・Chained・Concurrent・Conjunctive・Interpolate・Mixed・Multiple・Overlay・SecondOrder・Tandem）、包み型の 2 本（Timeout・ResponseCost）、Percentile の 1 本、嫌悪制御の 3 本（Sidman・DiscriminatedAvoidance・Escape）、試行型の 2 本（MatchingToSample・GoNoGo）は、一部の step で `meta`（`current_component`・`alternative_winner`・`cod_suppressed`・`timeout_active`・`in_probe`・`net_magnitude`・`phase`・`aversive_event`・`trial_outcome` など）も求めます。どの step にどのキーが要るかは fixture ファイルに書いてあります。

`contingency-conformance replay-python` は本パッケージの実行系に対する同じ役目のアダプタで、`contingency-conformance run -- contingency-conformance replay-python` は 39 本すべてに合格します。

## 乱数の種を持つ fixture

8 本（種付きの VR・VI・VT・RR・RI・RT・MatchingToSample・GoNoGo）は Python の `random.Random(seed)` で固定されています。`--stochastic` で判定のしかたを選びます。

- `count`（既定）: 強化の回数が記録値の 0.5〜1.5 倍に入り、最初の強化が最初の step より前でないこと。独自の乱数生成器を持つ実装向けです。
- `strict`: 決定的な fixture と同じ判定です。Python の乱数生成器を再現した実装向けです。
- `skip`: 判定しません。

試行型の 2 本では、どの比較刺激が正解か（MatchingToSample）、どの試行が Go か（GoNoGo）を乱数が決めます。生成器が違うと、同じ固定の反応で得る強化の回数が変わり、正しい実装でも `count` で落ちることがあります。Python の生成器を再現しない限り、この 2 本は `--stochastic skip` で流すか、生成器を移植してから `strict` で確かめてください。

## 出力と終了コード

出力は fixture ごとに `PASS` / `FAIL` / `ERROR` / `SKIP` の 1 行と最初の不一致（step・項目・期待値・観測値）、最後に集計です。`--json` で同じ内容を JSON で出します。終了コードは、不合格が無ければ 0、不合格かアダプタの異常終了があれば 1、入力の誤り（fixture が見つからない・未知の fixture・読めないファイル）なら 2 です。

# contingency-py

:gb: [English README](README.md)

オペラント随伴性のための Python 製強化スケジュールエンジン。

[`contingency-dsl`](../contingency-dsl/) の AST ノードを実行可能な `Schedule` ランタイムオブジェクトへ橋渡しする。セッション / コンテキスト / 再燃の扱いは含まない純粋ライブラリ — それらは [`experiment-core`](../../experiment/experiment-core/) に置かれる。

## スコープ

- 原子的強化スケジュール: FR, VR, RR, FI, VI, RI, FT, VT, RT, CRF, EXT
- 複合スケジュール: Concurrent (+ COD), Multiple, Chained, Tandem, Alternative
- 分化強化: DRO (resetting, momentary), DRL, DRH
- 累進比率 (Progressive Ratio)（arithmetic, geometric, Hantula ステップ関数）
- Limited Hold ラッパ
- Fleshler-Hoffman VI/VR 分布生成器 (Fleshler & Hoffman, 1962; Hantula, 1991)
- DSL ブリッジ: `contingency.bridge.from_dsl(ast)` → `Schedule`
- HAL (`contingency.hw`): virtual / serial / HIL bridge バックエンド

## インストール

```bash
pip install contingency-py
pip install "contingency-py[gym]"   # 任意: Gymnasium 環境
```

開発用:

```
mise install                                # Python 3.11
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

## 30 秒で動く例

スケジュールを [contingency-dsl](https://github.com/OperantKit/contingency-dsl) で書き、`from_dsl` でコンパイルし、反応を与えて進める:

```python
from contingency_dsl import parse
from contingency_dsl.annotations import TemporalExtension
from contingency_dsl.extensions import ExtensionRegistry
from contingency import ResponseEvent, from_dsl

registry = ExtensionRegistry.build(TemporalExtension())
source = '@algorithm("fleshler-hoffman", n=12, seed=42)\nConc(VI30s, VI60s, COD=2s)'
schedule = from_dsl(parse(source, extensions=registry))

# 仮想被験体: 0.5 秒ごとに 1 反応、10 秒ごとにオペランダムを切り替え、10 分間。
reinforcers = {"c0": 0, "c1": 0}
for i in range(1, 1201):
    now = i * 0.5
    operandum = "c0" if (i // 20) % 2 == 0 else "c1"
    outcome = schedule.step(now, ResponseEvent(time=now, operandum=operandum))
    if outcome.reinforced:
        reinforcers[operandum] += 1
print(reinforcers)  # {'c0': 9, 'c1': 3} — seed 固定なので毎回同じ
```

`contingency-dsl-py` は依存として一緒に入る。同じ `Schedule` オブジェクトは [session-runner](https://github.com/OperantKit/session-runner) でセッション全体の中で動かせる。

## Gymnasium 環境

`gym` extra を入れると、`contingency.envs.ScheduleEnv` が任意の `Schedule` を強化学習エージェント向けのフリーオペラントのセッションとして動かす。1 ステップで時計が `dt`（既定 1）進み、行動 `0` は無反応、行動 `i` は `i` 番目のオペランダムへの 1 反応。報酬は強化子の magnitude。観測は被験体が知りうるものだけ（直前の強化からの経過時間と反応数、直前のステップが強化されたか、直前に反応したオペランダム）。エピソードは `max_time` で打ち切る。

import すると 3 つの ID が登録される: `contingency/FR-v0`（`n`）、`contingency/FI-v0`（`interval`）、`contingency/ConcVIVI-v0`（`mean_left`・`mean_right`・`cod`。オペランダムは `left` と `right`）。`gym.make` のキーワード引数で既定値を上書きできる。

```python
import gymnasium as gym
import contingency.envs  # registers contingency/FR-v0, FI-v0 and ConcVIVI-v0

env = gym.make("contingency/ConcVIVI-v0", mean_left=30.0, mean_right=60.0, cod=2.0)
obs, info = env.reset(seed=0)
env.action_space.seed(0)
n_sr = 0
while True:  # 0 = no response, 1 = left, 2 = right; one step = 1 time unit
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    n_sr += reward > 0
    if terminated or truncated:  # truncated at max_time=600
        break
print(n_sr)  # 21
```

切替遅延（COD）による強化の抑止は `info["outcome_meta"]["cod_suppressed"]` に出る。`tests/test_envs_concurrent.py` では、既定の Conc VI 30 VI 60（COD 2）で 100 セッション学習した表形式の Q 学習が、学習に使っていない 10 セッションで 600 時間単位あたり平均 23.2 回の強化を得た。一様な無作為方策は 20.7 回、2 つのオペランダムから無作為に選んで必ず反応する方策は 21.1 回、学習前の同じ表は 21.5 回。

## 自作の 50 行では足りない理由

短い自作ループは次の点を誤りやすく、どれも報告するデータを変えてしまう:

- **変動スケジュールの値.** VI/VR の値は Fleshler-Hoffman (1962) の数列で生成する（Hantula, 1991 のプログラム版も選べる）。場当たりの `random.uniform` や指数分布の抽選ではない。`n` と `seed` を指定できるので、セッションを厳密に再現できる。
- **並立スケジュールの切替遅延 (COD).** `Conc` は各反応を対応する成分にだけ渡し、設定済みの強化子をそのオペランダムへの次の反応まで保持し、切替後 `COD` 時間単位は強化を抑止する。固定比率の切替要件 (`FRCO`) と方向別の COD にも対応する。
- **適合性 fixture.** `conformance/` の 24 個の JSON fixture が、既知の反応系列に対する各スケジュールの結果系列を厳密に固定している。Rust 版 [contingency-rs](https://github.com/OperantKit/contingency-rs) も同じ fixture を再生するので、2 つの実行系が 1 つの仕様に照らして検査される。
- **Method 節との往復変換.** スケジュールを contingency-dsl で書くので、[contingency-dsl2procedure](https://github.com/OperantKit/contingency-dsl2procedure) で Method 節の段落に変換でき、[contingency-procedure2dsl](https://github.com/OperantKit/contingency-procedure2dsl) で論文の Method 本文からスケジュールを抽出できる。
- **入力の検査.** 時刻の逆行・不正なパラメータ・未知のオペランダムは明示的なエラーになり、誤った強化時刻を黙って出すことはない。

[スケジュールの用語](docs/ja/schedule-terms.md) では、これらの点の背景にある問い（並立 VI の切替遅延 (COD)、Fleshler-Hoffman の数列、DRL と反応間時間、リセット型 DRO、VI の間隔がいつ始まるか）に数文ずつ答え、それぞれの書き方を示している。

**AI エージェントが書いたコードでの実測.** スケジュールを組む 20 件の依頼を Claude Code（claude-opus-5-5）と Codex CLI（gpt-6.1-sol）に 1 回ずつ与え、書かれたスケジュールのコードを contingency-dsl のスケジュール定義に照らして検査した（静的検査と、別モデルによる判定。判定は引用したコード行が実在するときだけ数える）:

- VI の間隔を Fleshler-Hoffman の数列で作ったのは 14 本中 7 本（Claude 7 本中 7 本、Codex 7 本中 0 本）。残りの 7 本は間隔を 1 つずつ指数分布か一様分布から抽選していた。
- 並立スケジュール 6 本はすべて COD を持っていたが、判定で 2 本に COD の誤りが見つかった（強化子の提示中の切替を記録しない、セッション最初の反応を切替として扱う）。
- 間隔スケジュールで、設定済みの強化子を次の反応まで保持し、次の間隔を提示から数えたのは 19 本中 16 本。残りの 3 本は、成分の切替で利用可能な強化子を捨てる、間隔の経過前の反応を強化しうる、このパッケージの `FI` を呼ぶだけで判定器が中身を見られない、の 3 つ。

## 自作の実装を fixture で検算する

`pip install contingency-py` で、適合性 fixture と `contingency-conformance` コマンドも入る。どの言語の実装でも、同じ 24 個の fixture に照らして検算できる。

```bash
pip install contingency-py
contingency-conformance list                    # fixture・種別・スケジュール型・step 数
contingency-conformance run --only 'atomic/fr_*' -- python my_adapter.py
```

`contingency-conformance path` は fixture のディレクトリを表示するので、他言語の検査器はそこから JSON を直接読める。`CONTINGENCY_CONFORMANCE_DIR` か `--dir` で別の写しを指定できる。

**観測 JSON.** 実装は fixture ごとに、fixture の step 1 つにつき 1 要素を返す。

```json
{"fixture": "atomic/fr_basic.json",
 "steps": [{"reinforced": false},
           {"reinforced": true, "reinforcer": {"time": 5.0, "magnitude": 1.0, "label": "SR+"}}]}
```

`reinforced` は完全一致。強化された step では `reinforcer.time` と `magnitude` の差が 1e-9 以内で、`label` が等しいこと。fixture の `meta` にあるキーはすべて同じ値で存在すること（観測側に余分なキーがあってもよい）。step 数も一致すること。このファイルを自分で書いて `contingency-conformance check observed/*.json` で比べるか、`run` にアダプタを起動させる。

**`run -- CMD` 用のアダプタ.** `run` は fixture ごとに CMD を 1 回起動し、fixture の JSON を標準入力に渡して、標準出力の観測 JSON を受け取る（`fixture` は省略してよい）。

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

このアダプタは FR だけを実装しているので、合格するのは `atomic/fr_basic.json` だけで、`--only` を外すと、他の 22 本はアダプタがスケジュールを組み立てられず `ERROR` になる。他も通すには `observe` で各スケジュール `type` を組み立てる。合成スケジュールの 6 本（Alternative・Chained・Concurrent・Multiple・Tandem）は一部の step で `meta`（`current_component`・`alternative_winner`・`cod_suppressed` など）も求める。どの step にどのキーが要るかは fixture ファイルに書いてある。

`contingency-conformance replay-python` は本パッケージの実行系に対する同じアダプタで、`contingency-conformance run -- contingency-conformance replay-python` は 23 本すべてに合格する。

**乱数の種を持つ fixture.** 6 本（種付きの VR・VI・VT・RR・RI・RT）は Python の `random.Random(seed)` で固定されている。`--stochastic` で判定のしかたを選ぶ。

- `count`（既定）: 強化の回数が記録値の 0.5〜1.5 倍に入り、最初の強化が最初の step より前でないこと。独自の乱数生成器を持つ実装向け。
- `strict`: 決定的 fixture と同じ判定。Python の乱数生成器を再現した実装向け。
- `skip`: 判定しない。

出力は fixture ごとに `PASS` / `FAIL` / `ERROR` / `SKIP` の 1 行と最初の不一致（step・項目・期待値・観測値）、最後に集計。`--json` で同じ内容を JSON で出す。終了コードは、不合格が無ければ 0、不合格かアダプタの異常終了があれば 1、入力の誤り（fixture が見つからない・未知の fixture・読めないファイル）なら 2。

## Swift 原典との違い

本パッケージは Swift のライブラリ [OperantKit](https://github.com/YutoMizutani/OperantKit) の移植である。`tests/test_legacy_swift_parity.py` は手元の Swift 原典をビルドし、小さなドライバ（`tools/legacy-swift/`）で上の fixture に通す。Swift のツールチェーンが要り、原典の場所を渡さない限り skip される。

```bash
OPERANTKIT_LEGACY_DIR=/path/to/OperantKit .venv/bin/pytest -q tests/test_legacy_swift_parity.py
```

Swift 原典にあるのは FR・CRF・EXT・FI・FT・VI・VR・VT・Alternative・Concurrent（切り替え遅延なし）。乱数生成器に種を与えられないので、VI・VR・VT は本パッケージが引く値の列をテストが注入する。FR・CRF・EXT・FI・FT は fixture と step ごとに一致する。既知の違い:

- **VI・VR・VT**: Swift 原典は値の列の最初の値を繰り返す。`Parameter.nextOrder()` が `mutating` でないため、列の位置が進まない。この 1 関数だけを `mutating` にすると（`tools/legacy-swift/nextorder-control.patch`、一時コピーにだけ当てる）3 本とも完全に一致する。
- **Alternative(FR 5, FT 10 s)**: 5 秒に FR が強化した後、Swift は次の FT の間隔を強化の時点（5 秒）から測って 15 秒に強化する。本パッケージは FT を次の step（6 秒）から測り直す。
- **Hantula (1991) の生成器**: 最小値が 0 に丸まるとき（たとえばテストした `v = 10, n = 12`）、Swift の `hantula1991` は 0 を残す。`generate_intervals_hantula1991` はそれを 1 に置き換えるので、合計が 1 多い。
- **Fleshler-Hoffman の生成器**（VR 5・`n = 12` と、VI の `(10, 12)`・`(60, 20)`・`(30000, 12)` で確認）: 比率の列は多重集合として一致する。Swift は丸めの過不足を無作為な位置に足すので、他のパラメータでは多重集合が変わりうる。Swift の間隔の生成器は整数を返す。Fleshler-Hoffman の数列を丸め、丸めの余りを無作為な位置の 1 要素に足したものである。`generate_intervals` は小数のまま返す。

これらは記録であって修正ではない。どちらの実装も相手に合わせて変えていない。

## 公開 API

```python
from contingency import Schedule, Observation, Reinforcer
from contingency.schedules import FR, VR, FI, VI, FT, VT, CRF, EXT, DRO
from contingency.schedules import Concurrent, Multiple, Chained, Tandem, Alternative
from contingency.builder import ScheduleBuilder
from contingency.bridge import from_dsl
```

## サポート

- 不具合の報告・機能の要望・使い方の質問は [issue tracker](https://github.com/OperantKit/contingency-py/issues) に書いてください。不具合の報告には、スケジュールの式（またはスケジュールを作るコード）、再現する反応・イベントの列、期待した結果と実際の結果、Python・contingency-py・contingency-dsl-py の版を添えてください。
- 開発者 1 名が手の空いたときに対応しています。返信の期日は約束できませんが、issue はすべて読みます。
- 変更の送り方は [CONTRIBUTING.md](CONTRIBUTING.md) を、行動規範は [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) を参照してください。

## 参考文献

- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Hantula, D. A. (1991). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 56(2), 391-392.
- Catania, A. C. (1966). Concurrent operants. In W. K. Honig (Ed.), *Operant behavior: Areas of research and application* (pp. 213-270). Appleton-Century-Crofts.
- Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*. Appleton-Century-Crofts.

## ライセンス

MIT. [LICENSE](LICENSE) を参照。

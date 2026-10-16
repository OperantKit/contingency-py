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

## 自作の 50 行では足りない理由

短い自作ループは次の点を誤りやすく、どれも報告するデータを変えてしまう:

- **変動スケジュールの値.** VI/VR の値は Fleshler-Hoffman (1962) の数列で生成する（Hantula, 1991 のプログラム版も選べる）。場当たりの `random.uniform` や指数分布の抽選ではない。`n` と `seed` を指定できるので、セッションを厳密に再現できる。
- **並立スケジュールの切替遅延 (COD).** `Conc` は各反応を対応する成分にだけ渡し、設定済みの強化子をそのオペランダムへの次の反応まで保持し、切替後 `COD` 時間単位は強化を抑止する。固定比率の切替要件 (`FRCO`) と方向別の COD にも対応する。
- **適合性 fixture.** `conformance/` の 22 個の JSON fixture が、既知の反応系列に対する各スケジュールの結果系列を厳密に固定している。Rust 版 [contingency-rs](https://github.com/OperantKit/contingency-rs) も同じ fixture を再生するので、2 つの実行系が 1 つの仕様に照らして検査される。
- **Method 節との往復変換.** スケジュールを contingency-dsl で書くので、[contingency-dsl2procedure](https://github.com/OperantKit/contingency-dsl2procedure) で Method 節の段落に変換でき、[contingency-procedure2dsl](https://github.com/OperantKit/contingency-procedure2dsl) で論文の Method 本文からスケジュールを抽出できる。
- **入力の検査.** 時刻の逆行・不正なパラメータ・未知のオペランダムは明示的なエラーになり、誤った強化時刻を黙って出すことはない。

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

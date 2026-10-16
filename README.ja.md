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

## 公開 API

```python
from contingency import Schedule, Observation, Reinforcer
from contingency.schedules import FR, VR, FI, VI, FT, VT, CRF, EXT, DRO
from contingency.schedules import Concurrent, Multiple, Chained, Tandem, Alternative
from contingency.builder import ScheduleBuilder
from contingency.bridge import from_dsl
```

## 参考文献

- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Hantula, D. A. (1991). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 56(2), 391-392.
- Catania, A. C. (1966). Concurrent operants. In W. K. Honig (Ed.), *Operant behavior: Areas of research and application* (pp. 213-270). Appleton-Century-Crofts.
- Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*. Appleton-Century-Crofts.

## ライセンス

MIT. [LICENSE](LICENSE) を参照。

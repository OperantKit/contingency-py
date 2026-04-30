# Swift レガシー・ソース対応表

各 `contingency-py` スケジュールを、アーカイブされた上流 `OperantKit`
Swift パッケージ内のオリジナル Swift ソースに対応づけるリファレンス表。
以下のパスは当該 Swift パッケージのリポジトリルートからの相対パス。

| スケジュール | Swift ファイル | 行 | 備考 |
|---|---|---|---|
| FR | `Sources/Common/Schedules/FR.swift` | 21-24 | 述語 `numOfResponses >= value`。 |
| CRF | `Sources/Common/Schedules/CRF.swift` | 18-20 | `fixedRatio(1)` に委譲。 |
| VR | `Sources/Common/Schedules/VR.swift` | 16-26 | ランタイム・サンプル値で `FR` に委譲。 |
| RR | `Sources/Common/Schedules/RR.swift` | 16-26 | `FR` に委譲（legacy は RR を構造的に FR として扱った）。 |
| FI | `Sources/Common/Schedules/FI.swift` | 14-17 | 述語 `numOfResponses > previous && fixedTime(value)`。 |
| VI | `Sources/Common/Schedules/VI.swift` | 18-30 | `FI` に委譲。 |
| RI | `Sources/Common/Schedules/RI.swift` | 18-30 | `FI` に委譲。指数サンプリングは外部。 |
| FT | `Sources/Common/Schedules/FT.swift` | 15-17 | 述語 `milliseconds >= value`。 |
| VT | `Sources/Common/Schedules/VT.swift` | 18-30 | `FT` に委譲。 |
| RT | `Sources/Common/Schedules/RT.swift` | 18-30 | `FT` に委譲。 |
| EXT | `Sources/Common/Schedules/EXT.swift` | 16-18 | 常に `false` を返す。 |
| Fleshler-Hoffman | `Sources/Common/Helpers/FleshlerHoffman.swift` | 11-95 | `generatedInterval` (15-52) と `generatedRatio` (55-95)。Hantula バリアントは 100-129。 |
| Concurrent | — | — | レガシー・ソースなし。`contingency-py` で新規。 |
| Alternative | — | — | レガシー・ソースなし。 |
| Multiple / Chained / Tandem | — | — | レガシー・ソースなし。 |
| LimitedHold | — | — | レガシー・ソースなし。 |
| DRO / DRL / DRH | — | — | レガシー・ソースなし。 |
| ProgressiveRatio | — | — | レガシー・ソースなし。 |

## 意味論的同値に関する注記

- Swift パッケージは `(numOfResponses, milliseconds)` を運ぶ
  `ResponseEntity` を用いるリアクティブ (RxSwift) パイプラインを使用
  していた。各スケジュールはその entity に対する純粋な述語であり、
  状態は周辺のストリーム・コンビネータが所有していた。
- `contingency-py` は状態所有を反転させる: 各スケジュールは明示的な
  `step(now, event)` 呼び出しで駆動される状態保持オブジェクトである。
  述語は保持されている（`FR >= value`、`FI > previous && elapsed >=
  interval`、`FT elapsed >= value` を参照）が、周辺のアンカー /
  カウンタ / シーケンス状態は現在は内部状態である。
- Swift ソースのランダム系スケジュール (RR, RI, RT) は外部供給の
  可変値を伴う固定版相当に委譲していた。Python 移植版はランダム・
  サンプリングを明示的かつ内部的にする（RR は `Random.random()`、
  RI と RT は `Random.expovariate`）。
- Fleshler-Hoffman の Swift 実装は整数ミリ秒で動作していた。Python
  移植版は単位非依存（呼び出し側が宣言するクロック上の float）で、
  数値安定性のため `math.fsum` を使用する。

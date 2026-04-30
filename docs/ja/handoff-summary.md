# `contingency-rs` 引き継ぎサマリ

このドキュメントは、`contingency-py` を `contingency-rs` に移植する
Rust 実装者のための 1 ページ分オンボーディング資料である。何が存在し、
何が延期されており、どこから読み始めるべきかをまとめる。

## `contingency-py` に存在するもの

### ランタイムスケジュール（本番利用可能）

| ファミリ | クラス |
|---|---|
| 比率 (Ratio) | `FR`, `VR`, `RR`, `CRF` (factory -> `FR(1)`) |
| 間隔 (Interval) | `FI`, `VI`, `RI`, `LimitedHold` |
| 時間ベース (Time-based) | `FT`, `VT`, `RT`, `EXT` |
| 分化強化 (Differential) | `DRO` (resetting/momentary), `DRL`, `DRH` |
| 複合 (Compound) | `Concurrent` (COD + COR), `Multiple`, `Chained`, `Tandem`, `Alternative` |
| 累進 (Progressive) | `ProgressiveRatio`（`arithmetic`, `geometric`, `richardson_roberts` ステップ関数付） |

すべて `Schedule` Protocol (`step(now, event) -> Outcome`,
`reset() -> None`) を実装する。完全な契約は [protocol.md](protocol.md)
を参照。

### 値オブジェクト

- `ResponseEvent(time, operandum="main")`
- `Observation(time, response_count=0)`
- `Reinforcer(time, magnitude=1.0, label="SR+")`
- `Outcome(reinforced=False, reinforcer=None, meta={})`

不変条件: `reinforced <=> reinforcer.is_some()`。構築時に強制される。

### エラー分類体系

```
ContingencyError
├── ScheduleConfigError
├── ScheduleStateError
└── HardwareError
    └── NotConnectedError
```

Rust 側では 4 バリアントの `enum Error` にマップされる
（[protocol.md](protocol.md) 参照）。

### ヘルパー

- `contingency.helpers.fleshler_hoffman.generate_intervals`,
  `generate_ratios`, `generate_intervals_hantula1991` — `seed`
  パラメータ付きの決定論的プール生成器。

### DSL ブリッジ

`contingency.bridge.from_dsl(ast_node, time_unit_seconds=1.0) -> Schedule`
は `contingency-dsl` AST ノードをランタイムスケジュールに変換する。
完全な変換表は当該モジュールの docstring を参照。

### ハードウェア抽象化層

`contingency.hw` は `Apparatus` Protocol の背後に 3 つのバックエンド
(`VirtualApparatus`, `SerialBackend`, `HilBridgeApparatus`) を提供する。
Rust 移植版は PyO3 / KMP フロントエンドが決定した時点で同じ形状を
反映させる想定。Rust 実装はおそらく `contingency-rs` の隣に置く
新規の低レベル `hal` crate になる可能性が高いため、本パッケージでは
意図的に HAL の詳細は記述しない。

## 延期されているもの

ブリッジ (`src/contingency/bridge/__init__.py`) より:

- `SecondOrder` — 未実装。
- `IdentifierRef` — let 束縛を置換するために DSL アナライザパスが必要。
- `AdjustingSchedule`, `AversiveSchedule`, `InterlockingSchedule`,
  `ResponseCostWrapped`, `TimeoutWrapped`, `TrialBased` — すべて延期。
- コンビネータ `CONJ`, `MIX`, `OVERLAY`, `INTERPOLATE`, `INTERP` — 延期。
- `Modifier` kind `Pctl`（パーセンタイル・スケジュール） — 延期。
- `Concurrent` パラメータ `PUNISH` — 延期。
- `Concurrent` の方向性 COD (`cod_asymmetric`) — 延期。

Rust 移植版は、機能が実装されるまで延期されたすべての構造に対して
同じ `Error::Config` を返すべきである。

## Swift オリジナルとの既知の意味的な差異

ファイル別の対応表は [swift-legacy-map.md](swift-legacy-map.md)
を参照。Rust 移植版が守らなければならない主要な意味的差異
（これらは **Python** の挙動であり、Swift の挙動ではない）:

1. **アンカリング (Anchoring)**。時間ベース・スケジュール
   (`FT`, `VT`, `RT`, `DRO`) はクロックを **最初の `step()` 呼び出し**
   でアンカーし、`t=0` ではない。`FI`, `VI`, `RI` は構築時に
   アンカーする (`arm_time = interval` または
   `arm_time = sequence[0]`)。
2. **ステップごとに 1 回発火**。呼び出し側が `k > 1` の間隔だけ
   遅れてステップした場合、発生する強化子は **1 つだけ**。
   呼び出し側はスケジュールを遵守するのに十分な頻度で
   ステップする必要がある。
3. **ランダム系サンプリングは内部処理**。`RR`, `RI`, `RT` は
   引き（ベルヌーイ / 指数）を内部でサンプリングする。Swift の
   `RI`/`RT`/`RR` ファイルは外部供給値を伴う固定版 (`FI`/`FT`/`FR`)
   に委譲していた。
4. **単位非依存の時間**。すべての時刻は単位非依存の float。単位
   契約は外部（`contingency-dsl` が宣言、`experiment-io` が強制）。
   Swift はミリ秒型だった。
5. **Outcome 不変条件は構築時に強制**。強化子なしに強化された
   Outcome、あるいはその逆を許容してはならない。

## Rust 移植時の落とし穴

1. **`TIME_TOL = 1e-9`** は指定通り、すべての比較（単調時間、
   イベント時刻マッチング、経過時間、スライディングウィンドウの
   eviction、LimitedHold の失効）で正確に適用すること。
2. **`Concurrent` はすべてのステップで全コンポーネントを進める**。
   イベントを受け取らなかったコンポーネントであっても進める。
   これは他の operandum に反応として到着した tick で時間ベース
   コンポーネント (FT/VT/RT) を進めるために必須。
3. **`Concurrent` のゲーティング順序**: チェンジオーバー反応自体は
   それが開く COD によってカバーされる。COD チェックの *前に*
   チェンジオーバーを記録すること。COD は他コンポーネントからの
   tick 由来の強化をゲートしない。
4. **`Chained` / `Tandem` は非活性コンポーネントをステップしない**。
   時間ベースの終端リンクは遷移時点ではなく、活性化後の最初の
   ステップでアンカーする。
5. **`Alternative` はすべてのステップで `(now, event)` を両方の
   コンポーネントに転送する**。両方のコンポーネントは独自の状態を
   維持する。ラッパは表層に出す Outcome のみをゲートし、勝利時に
   両方をリセットする。
6. **`DRO` momentary** は入ってきたイベントを記録する **前に** 境界を
   評価する（半開ウィンドウ `[anchor, now)`）。resetting バリアント
   はその逆: イベントがタイマーをリセットし、そのステップで発生
   する可能性のある強化を先取りして無効化する。
7. **`DRH` のウィンドウは強化時に空にならない**。持続する高頻度
   応答列は発火し続ける。ブレークポイント的な挙動は外部で追加
   する必要がある。
8. **`ProgressiveRatio` の `step_fn` は遅延呼び出し**。無効な戻り値
   は構築時ではなく、関数を参照する最初の反応時に `Config`
   エラーとして表面化する。
9. **`reset()` はシード未指定スケジュールでランダムパラメータを
   再サンプリングしない**。`VR(mean, seed=None).reset()` は
   `seed=None` から再シードし、新しい RNG を生成する。ビット単位
   の再現性のためには、常にシード付きで構築すること。
10. **`RR`, `RI`, `RT` は構築時に RNG 状態をスナップショットする**。
    呼び出し側が共有 `Random` を渡した場合、スナップショットは
    即座に取得される — `reset()` ごとではない。つまり共有 RNG が
    他所で進められていても、`reset()` はスケジュールのプライベート
    な引き列を初期状態に戻す。
11. **VR/VT/VI のサブシード定義域**: VR/VT は
    `rng.randrange(0, 2**31 - 1)`、VI は `rng.getrandbits(64)`。
    Rust は Python とのビット一致を目指す場合のみ同じ範囲を使う
    べきで、そうでない場合は conformance fixture で列を固定する。
12. **`Outcome.meta` キーの観測値** は全ファミリで以下:
    `"current_component"` (`Multiple`/`Chained` では str、
    `Tandem` では int)、`"chain_transition"` (bool)、
    `"cod_suppressed"` (bool)、`"operandum"` (str)、
    `"alternative_winner"` (`"first"` | `"second"`)。

## 推奨 Rust モジュール構造

```
contingency-rs/
├── src/
│   ├── lib.rs                   // re-exports
│   ├── entities.rs              // ResponseEvent, Reinforcer, Outcome, ...
│   ├── errors.rs                // Error enum
│   ├── schedule.rs              // Schedule trait
│   ├── schedules/
│   │   ├── mod.rs
│   │   ├── ratio.rs             // FR, VR, RR
│   │   ├── interval.rs          // FI, VI, RI, LimitedHold
│   │   ├── time_based.rs        // FT, VT, RT, EXT
│   │   ├── differential.rs      // DRO, DRL, DRH
│   │   ├── sequence.rs          // Multiple, Chained, Tandem
│   │   ├── concurrent.rs        // Concurrent
│   │   ├── alternative.rs       // Alternative
│   │   └── progressive.rs       // ProgressiveRatio + step fns
│   ├── helpers/
│   │   └── fleshler_hoffman.rs  // generate_intervals / generate_ratios
│   └── builder.rs               // ScheduleBuilder facade
└── tests/
    └── conformance.rs           // Loads ../../contingency-py/conformance/*.json
```

推奨 crate 機能:

- `serde` — conformance fixture 読み込み用。
- `pyo3` — `Schedule` を Python 型として公開する Python バインディング用。
- `wasm` — ブラウザデモ用。
- `c-ffi` — HIL ブリッジバイナリ用。

## どこから読み始めるか

1. [`protocol.md`](protocol.md) — 公開 API 形状の唯一の真実の源泉。
2. [`algorithms/`](algorithms/) — スケジュール 1 つにつき 1 ファイル。
   実装決定の 80% がここにある。
3. [`../conformance/README.md`](../../conformance/README.md) — Rust
   テストから fixture を消費する方法。
4. [`swift-legacy-map.md`](swift-legacy-map.md) — 11 原子的スケジュール
   のみ必要。新しい複合スケジュールに対応する Swift ソースは存在しない。

## `contingency-py` のテスト範囲

Python パッケージは 613 テスト（100% 緑）、ライン・カバレッジ 99%。
ソースを読むよりもテストを読む方が速いことが多い。各スケジュールには
[`../tests/`](../../tests/) 配下に専用ファイルがある:

- `test_ratio_schedules.py`
- `test_interval_schedules.py`
- `test_time_schedules.py`
- `test_differential_schedules.py`
- `test_concurrent.py`
- `test_sequence_schedules.py`
- `test_alternative.py`
- `test_progressive.py`
- `test_fleshler_hoffman.py`
- `test_bridge.py` (DSL 変換)
- `test_builder.py`
- `test_conformance.py` (コーパスに対する回帰チェック)

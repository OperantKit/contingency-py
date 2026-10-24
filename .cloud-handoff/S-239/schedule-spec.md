# schedule spec — 中間の形の調査（S-239）

調査結果。コードは直していない。読んだのは次の 4 か所と DSL の schema である。

| 場所 | 役割 | 分岐数 |
|---|---|---|
| contingency-py `src/contingency/conformance/spec.py` (`build_schedule_from_spec`) | 適合テストの `"schedule"` を Python の Schedule にする。生成器と `replay-python` が共用 | 36 type（35 クラス + `CRF`） |
| contingency-py `src/contingency/bridge/__init__.py` (`from_dsl`) | contingency-dsl の AST を Python の Schedule にする | 全 35 クラス（spec を経由しない） |
| contingency-rs `tests/conformance.rs` (`build_schedule`) | 同じ spec を Rust の Schedule にする。39 本の fixture を再生 | 36 type（`LimitedHold` の内側は FI/VI/RI の 3 分岐、合計 39 分岐） |
| contingency-rs `src/builder.rs` (`ScheduleBuilder`) と `src/python.rs` (PyO3 の classmethod) | Rust 内・PyO3 向けの組み立て口 | builder 26 メソッド = 20 クラス、PyO3 21 classmethod = 19 クラス |
| contingency-rs `src/bin/hil.rs` (`build_schedule`) | `--schedule` の JSON を Rust の Schedule にする | 8 type |

前提（決定済み）: Rust のエンジンが正本、仕様の拠り所は適合テスト。したがって spec とは「適合テストの `"schedule"` オブジェクト」そのものであり、本書はその形を 35 種類ぶん一覧にし、DSL の AST からの対応と、spec で表せないものを列挙する。

## 1. spec の共通形

```json
{"type": "<type 名>", "params": { ... }}
```

- `type` は文字列。大文字小文字を区別する（`FR`, `LimitedHold`, `MatchingToSample` など、下表の綴り）。
- `params` はオブジェクト。省略可（`CRF`, `EXT` は `{}`）。Rust 側は `#[serde(default)]` で欠落を許す。
- 入れ子のスケジュールは同じ `{"type", "params"}` を値として埋め込む（`inner`, `first`, `second`, `base`, `probe`, `punisher`, `overall`, `unit`, `components[]`, `components{}`, `consequence`, `incorrect`, `correct_go_schedule`, `correct_nogo_schedule`, `false_alarm`）。
- 省略可能な入れ子は「キー無し」でも `null` でもよい。Python `_optional_spec` と Rust `optional_schedule` はどちらも None 扱い。fixture は `null` を明示している（`trial/mts_n2_seed7.json`, `trial/gonogo_p05_seed1.json`）。
- 時間は単位なしの実数で、エンジンのクロック単位（既定は秒）。spec に単位フィールドは無い。単位変換は DSL→spec の段階で済ませる（§3）。
- 整数パラメータ（`n`, `n_intervals`, `seed`, `response_count`, `frco`, `rank`, `window`, `initial_ratio`, `n_comparisons`, `start`/`step_size` of PR）は JSON の整数で書く。Python は `int(...)` で `5.0` も受けるが、Rust は `as_u64` なので `5.0` は欠落扱いで panic する。正本が Rust なので「整数は整数で書く」が規則になる。
- 乱数の `seed` は整数。Python は `random.Random(seed)`、Rust は CPython 互換の `PyRandom` で同じ列を引く。`seed` を省くと非決定になり、fixture には使えない。
- 既定値は spec.py と conformance.rs が同じ値を持っている（下表の「既定」列）。builder.rs / python.rs は既定値を持たない引数もある（`n_intervals` は PyO3 側で 12）。

### 4 か所の対応状況（type ごと）

| type | spec.py | conformance.rs | builder.rs / python.rs | hil.rs |
|---|---|---|---|---|
| FR, CRF, EXT, FI, FT | ○ | ○ | ○ / ○ | ○ |
| DRO, DRL, DRH | ○ | ○ | ○ / ○ | ○（DRO のキー名が `mode`。他は `dro_type`。食い違い） |
| VR, RR, VI, RI, VT, RT | ○ | ○ | ○ / ○ | × |
| LimitedHold | ○（内側は `_withdraw_and_rearm` を持つ FI/VI/RI） | ○（内側 FI/VI/RI） | ○（`limited_hold_fi/vi/ri`）/ ○（`limited_hold(PyArmableSchedule)`） | × |
| Concurrent, Alternative, Multiple, Chained, Tandem | ○ | ○ | ○ / ○ | × |
| ProgressiveRatio | ○ | ○ | ○（3 段階関数）/ × | × |
| Timeout, ResponseCost, Percentile, Conjunctive, Mixed, Interpolate, Overlay, SecondOrder, InterlockingSchedule, AdjustingSchedule, Sidman, DiscriminatedAvoidance, Escape, MatchingToSample, GoNoGo | ○ | ○ | × / × | × |

builder.rs にあって spec に無いもの: なし（builder は spec の部分集合）。spec にあって Rust のエンジンに無いもの: なし（35 クラスとも Rust にある）。Python にあって spec に無いもの: `PhasedConcurrent`（`schedules/phased.py`。Concurrent を時刻で切り替える包み。fixture も Rust 実装も無い）。

## 2. 35 種類の params 表

列の意味: 名前 | JSON 型 | 単位 | 必須か既定値 | 備考。「秒」はエンジンのクロック単位。「○」は必須。

### 2.1 原子（11 type、10 クラス）

| type | params | 備考 |
|---|---|---|
| `FR` | `n`: 整数 ≥1 / なし / ○ | fixture `{"n": 5}` |
| `CRF` | なし | `FR(1)` の別名。spec では独立した type として残す（fixture `atomic/crf_basic.json` が使う） |
| `EXT` | なし | |
| `VR` | `mean`: 実数 >0 / 応答数 / ○ ・ `n_intervals`: 整数 ≥1 / なし / 既定 12 ・ `seed`: 整数 / 既定なし | 値列は `fleshler-hoffman` 進行。`mean` の名前は VI 系の `mean_interval` と揃っていない |
| `RR` | `probability`: 実数 (0,1] / なし / ○ ・ `seed` | DSL の RR は「平均応答数」なので bridge は `1/value` に変換する（§3） |
| `FI` | `interval`: 実数 >0 / 秒 / ○ | |
| `VI` | `mean_interval`: 実数 >0 / 秒 / ○ ・ `n_intervals` 既定 12 ・ `seed` | |
| `RI` | `mean_interval` / 秒 / ○ ・ `seed` | 指数分布 |
| `FT` | `interval` / 秒 / ○ | 応答非依存 |
| `VT` | `mean_interval` / 秒 / ○ ・ `n_intervals` 既定 12 ・ `seed` | |
| `RT` | `mean_interval` / 秒 / ○ ・ `seed` | |

`seed` の扱いの差: Python は `"seed" in params` で `random.Random(params["seed"])` を作るので `"seed": null` は非決定の乱数になる。Rust は `as_u64` が失敗して None になり、同じく非決定。実害は無いが、正本の規則は「決定的にしたければ整数を書く、そうでなければキーを書かない」とする。

### 2.2 包み（3 type、3 クラス）

| type | params | 備考 |
|---|---|---|
| `LimitedHold` | `inner`: spec / ○ ・ `hold`: 実数 >0 / 秒 / ○ | 内側は armable（FI, VI, RI）のみ。Rust は型レベルの制約（`LimitedHold<S: ArmableSchedule>`）、Python は `_withdraw_and_rearm` の有無で検査。fixture は FI のみ |
| `Timeout` | `inner`: spec / ○ ・ `duration`: 実数 >0 / 秒 / ○ ・ `reset_on_response`: 真偽 / 既定 false | |
| `ResponseCost` | `inner` / ○ ・ `amount`: 実数 / 強化子の個数 / ○ ・ `unit`: 文字列 / 既定 `"token"` ・ `initial_balance`: 実数 または null / 既定 null | `initial_balance` は DSL に無い（DSL は残高を持たない） |

### 2.3 分化強化（4 type、4 クラス）

| type | params | 備考 |
|---|---|---|
| `DRO` | `interval`: 実数 >0 / 秒 / ○ ・ `dro_type`: `"resetting"` か `"momentary"` / 既定 `"resetting"` | hil.rs だけキーが `mode` |
| `DRL` | `interval` / 秒 / ○ | IRT ≥ interval で強化 |
| `DRH` | `response_count`: 整数 ≥1 / ○ ・ `time_window`: 実数 >0 / 秒 / ○ | 「窓内に N 応答」 |
| `Percentile` | `target`: 文字列 / ○ ・ `rank`: 整数 0–100 / ○ ・ `window`: 整数 ≥1 / ○ ・ `direction`: `"above"` か `"below"` / 既定 `"above"` | Python は `target` に `irt`/`IRT`/`force`/`latency`/`duration`/`rate`/`location`/`amplitude` を受けるが実行できるのは `irt`/`IRT` だけ（他は step 時に例外）。Rust は `"irt"` のみで `"IRT"` は panic。`rank` は Rust では u8。`direction` の既定が DSL（`below`）と逆（§4） |

### 2.4 複合（10 type、10 クラス）

| type | params | 備考 |
|---|---|---|
| `Concurrent` | `components`: オブジェクト {操作体名: spec}、2 個以上 / ○ ・ `cod`: 実数 ≥0 / 秒 / 既定 0 ・ `frco`: 整数 ≥0 / 既定 0 | キーの順序を保つ（Python dict、Rust IndexMap）。キーが応答の `operandum` と一致する。両エンジンにある `cod_directional`（{(from,to): 秒}）と `punish`（{操作体名: Schedule}）は spec に無い（§4） |
| `Alternative` | `first`: spec / ○ ・ `second`: spec / ○ | 2 項のみ。3 項以上は左結合で入れ子 |
| `Conjunctive` | `first` / ○ ・ `second` / ○ | 同上 |
| `Multiple` | `components`: 配列 [spec]、1 個以上 / ○ ・ `stimuli`: 配列 [文字列]、要素数は components と同じ / 省略可 | 省略時の既定名は Python `comp_{i}`。fixture は `["s0","s1"]` を明示 |
| `Mixed` | `components`: 配列 / ○ | 弁別刺激なし |
| `Chained` | `components`: 配列 / ○ ・ `stimuli` / 省略可 | |
| `Tandem` | `components`: 配列 / ○ | |
| `Interpolate` | `base`: spec / ○ ・ `probe`: spec / ○ ・ `interval`: 実数 >0 / 秒 / ○ ・ `probe_duration`: 実数 >0 / 秒 / ○ ・ `first_probe_at`: 実数 または null / 秒 / 既定 null（= interval） | DSL の `count`/`onset` とは別の幾何（§3） |
| `Overlay` | `base`: spec / ○ ・ `punisher`: spec / ○ | 罰子側は負の magnitude、label `SR-` |
| `SecondOrder` | `overall`: spec / ○ ・ `unit`: spec / ○ | |

### 2.5 累進・調整（3 type、3 クラス）

| type | params | 備考 |
|---|---|---|
| `ProgressiveRatio` | `step`: `"arithmetic"` / `"geometric"` / `"richardson_roberts"` / ○。`arithmetic`: `start` 整数 既定 1、`step_size` 整数 既定 1。`geometric`: `start` 整数 既定 1、`ratio` 実数 既定 2.0。`richardson_roberts`: 追加パラメータなし | `richardson_roberts` は DSL に書けない |
| `InterlockingSchedule` | `initial_ratio`: 整数 ≥1 / ○ ・ `decay_time`: 実数 >0 / 秒 / ○ | |
| `AdjustingSchedule` | `target`: `"ratio"` か `"interval"` / ○ ・ `start`: 実数 / ○ ・ `step`: 実数 / ○ ・ `minimum`: 実数 または null ・ `maximum`: 実数 または null | 単位は target が `interval` のとき秒。Python は `delay`/`amount` を受理して構築時に未対応エラー、Rust の enum にも `Delay`/`Amount` はあるが conformance.rs は `ratio`/`interval` 以外 panic |

### 2.6 嫌悪統制（3 type、3 クラス）

| type | params | 備考 |
|---|---|---|
| `Sidman` | `shock_shock_interval`: 実数 >0 / 秒 / ○ ・ `response_shock_interval`: 実数 >0 / 秒 / ○ ・ `shock_magnitude`: 実数 / 既定 1.0 | 電撃は `reinforced: true`, magnitude 負, label `SR-` として出る |
| `DiscriminatedAvoidance` | `warning_duration`: 実数 >0 / 秒 / ○ ・ `iti`: 実数 >0 / 秒 / ○ ・ `shock_magnitude` 既定 1.0 | 警告刺激は応答で終了、タイムアウトで点電撃。DSL の `mode`/`ShockDuration`/`MaxShock` は反映されない |
| `Escape` | `trial_duration`: 実数 >0 / 秒 / ○ ・ `iti`: 実数 >0 / 秒 / ○ ・ `shock_magnitude` 既定 1.0 | |

### 2.7 試行型（2 type、2 クラス）

| type | params | 備考 |
|---|---|---|
| `MatchingToSample` | `n_comparisons`: 整数 ≥2 / ○ ・ `sample_duration`: 実数 ≥0 / 秒 / 既定 0.0 ・ `choice_timeout`: 実数 >0 / 秒 / 既定 5.0 ・ `consequence`: spec または null ・ `incorrect`: spec または null ・ `iti`: 実数 / 秒 / 既定 1.0 ・ `seed`: 整数 | 比較刺激の操作体名は `choice_0` … `choice_{N-1}`。正解位置は乱数 |
| `GoNoGo` | `go_probability`: 実数 [0,1] / 既定 0.5 ・ `response_window`: 実数 >0 / 秒 / 既定 5.0 ・ `iti`: 実数 / 秒 / 既定 5.0 ・ `correct_go_schedule`: spec または null ・ `correct_nogo_schedule`: spec または null ・ `false_alarm`: spec または null ・ `seed`: 整数 | 試行種別は乱数 |

### 2.8 fixture が実際に使っている spec（39 本）

```
atomic/        CRF{} EXT{} FI{interval:10} FR{n:5} FT{interval:5}
               LimitedHold{hold:2, inner:FI{interval:5}}
               RI{mean_interval:5, seed:5} RR{probability:0.2, seed:99} RT{mean_interval:3, seed:11}
               VI{mean_interval:10, n_intervals:12, seed:7} VR{mean:5, n_intervals:12, seed:42}
               VT{mean_interval:10, n_intervals:12, seed:3}
wrapper/       Timeout{inner:FR2, duration:3, reset_on_response:false}
               ResponseCost{inner:FR2, amount:1, unit:"token", initial_balance:3}
differential/  DRO{interval:5, dro_type:resetting|momentary} DRL{interval:3}
               DRH{response_count:3, time_window:2}
               Percentile{target:"irt", rank:75, window:3, direction:"above"}
compound/      Concurrent{components:{left:FR3, right:FR3}, cod:1, frco:0}
               Concurrent{components:{a:FT5, b:FI3}, cod:2, frco:0}
               Alternative{first:FR5, second:FT10} Conjunctive{first:FR3, second:FI5}
               Multiple{components:[FR2,FR3], stimuli:[s0,s1]} Mixed{components:[FR2,FR3]}
               Chained{components:[FR2,FR3|FI5], stimuli:[s0,s1]} Tandem{components:[FR2,FR3]}
               Interpolate{base:FR2, probe:EXT, interval:10, probe_duration:4, first_probe_at:5}
               Overlay{base:FR2, punisher:FR3} SecondOrder{overall:FR2, unit:FR3}
progressive/   ProgressiveRatio{step:arithmetic, start:1, step_size:1}
               InterlockingSchedule{initial_ratio:4, decay_time:8}
               AdjustingSchedule{target:ratio, start:2, step:1, maximum:3}
aversive/      Sidman{shock_shock_interval:5, response_shock_interval:3}
               DiscriminatedAvoidance{warning_duration:3, iti:4} Escape{trial_duration:3, iti:2}
trial/         MatchingToSample{n_comparisons:2, sample_duration:1, choice_timeout:3, iti:1, seed:7, consequence:null, incorrect:null}
               GoNoGo{go_probability:0.5, response_window:2, iti:1, seed:1, correct_go_schedule:null, correct_nogo_schedule:null, false_alarm:null}
```

fixture が触っていない引数: `LimitedHold` の内側 VI/RI、`Percentile` の `below`、`ProgressiveRatio` の `geometric`/`richardson_roberts`、`AdjustingSchedule` の `interval`/`minimum`、`shock_magnitude` ≠ 1、`Concurrent` の `frco` > 0、`MatchingToSample`/`GoNoGo` の入れ子 spec。spec を正本化するなら、これらにも fixture を足すのが筋（S-239 の範囲外、引き継ぎに記す）。

## 3. DSL の構文木（AST）→ spec の対応表

出典: contingency-dsl `schema/operant/ast.schema.json`（+ `stateful/`, `trial-based/`）と contingency-py の `bridge/__init__.py`。現状の bridge は AST から直接 Python のオブジェクトを作っており spec を経由しないので、下表は「bridge がやっている変換を spec の語で書き直したもの」である。bridge を「AST→spec→エンジン」に組み替えるときの仕様になる。

共通規則:

- 時間値 `value` と `time_unit`（`s`/`ms`/`min`、省略時は 1 倍）→ `value × 係数 × time_unit_seconds`（係数 s=1, ms=0.001, min=60。`time_unit_seconds` は呼び出し側のクロック換算、既定 1）。spec の時間は全部この結果の実数。
- 入れ子の `ScheduleExpr` → 再帰して spec。
- 1 始まりの成分参照（`from`/`to`/`component` の整数 k）→ Concurrent のキー `c{k-1}`。識別子なら `let` 名をそのままキーに使う。
- AST は解析後（`ast-resolved.schema.json`）であること。`IdentifierRef` が残っていれば変換できない。

### 3.1 Program と注釈

| AST | spec | 備考 |
|---|---|---|
| `Program.schedule` | その spec | Program 自体に対応する type は無い |
| `Program.program_annotations[@algorithm(name, n=, seed=)]` / 各ノードの `annotations[@algorithm]` | `VI`/`VR`/`VT` の `n_intervals` と `seed`、`RI`/`RR`/`RT`/`MatchingToSample`/`GoNoGo` の `seed` | `seed=S` は構築順（左→右、試行型は入れ子の後）に `S, S+1, …` を配る。`name` は `fleshler-hoffman` のみ可、`arithmetic` は VI/VR/VT で エラー。内側の注釈は設定した項目だけ上書き |
| `Program.param_decls` (`LH`, `COD`, `FRCO`, `BO`, `RD`) | なし | bridge は読まない。`LH` は DSL の解析 Phase 3 で各ノードの `limitedHold` に展開されるので結果的に反映される。`COD`/`FRCO`/`BO`/`RD` は落ちる（§4） |
| `Program.bindings` (`let`) | なし | 解析で置換済みであること |
| 上記以外の注釈（`@reinforcer`, `@sd`, `@species`, …） | なし | 実行には影響しない |

### 3.2 原子・特殊

| AST | spec |
|---|---|
| `Atomic{dist:F, domain:R, value}` | `FR{n: int(value)}`（value が正の整数でなければエラー） |
| `Atomic{F, I}` | `FI{interval: t(value)}` |
| `Atomic{F, T}` | `FT{interval: t(value)}` |
| `Atomic{V, R}` | `VR{mean: value, n_intervals: @algorithm.n, seed: 次の seed}` |
| `Atomic{V, I}` | `VI{mean_interval: t(value), n_intervals, seed}` |
| `Atomic{V, T}` | `VT{mean_interval: t(value), n_intervals, seed}` |
| `Atomic{R, R}` | `RR{probability: 1/value, seed}`（value ≤ 0 はエラー） |
| `Atomic{R, I}` | `RI{mean_interval: t(value), seed}` |
| `Atomic{R, T}` | `RT{mean_interval: t(value), seed}` |
| `Special{kind:EXT}` | `EXT{}` |
| `Special{kind:CRF}` | `CRF{}`（エンジン上は FR 1） |

`t(value)` = 共通規則の時間変換。

### 3.3 ノードに付く後置（LH / TO / ResponseCost）

schema ではこれらは `Atomic`/`Special`/`DRModifier`/`SecondOrder`/`AversiveSchedule` のプロパティ（`limitedHold`+`limitedHoldUnit`, `timeout`, `responseCost`）で、contingency-dsl-py の AST では包むノード（`LimitedHold`, `TimeoutWrapped`, `ResponseCostWrapped`）になる。spec では外側から包む。

| AST | spec |
|---|---|
| `limitedHold`, `limitedHoldUnit` | `LimitedHold{inner: <本体>, hold: t(limitedHold)}`。本体が FI/VI/RI 以外ならエンジンがエラー |
| `timeout{duration, durationUnit, resetOnResponse}` | `Timeout{inner, duration: t(duration), reset_on_response}` |
| `timeout.contingentResponse` | なし（§4） |
| `responseCost{amount, unit}` | `ResponseCost{inner, amount, unit}`（`initial_balance` は null） |

適用順は DSL の語順どおり LH → TO → ResponseCost（外側ほど後）。

### 3.4 複合

| AST (`Compound.combinator`, `params`) | spec |
|---|---|
| `Conc(a, b, …)` | `Concurrent{components: {"c0": a, "c1": b, …}, cod, frco}` |
| `Conc … COD={value,time_unit}` | `cod: t(value)` |
| `Conc … COD={base, directional:[{from,to,value,time_unit}]}` | `cod: t(base)`（base 無しは 0）。`directional` は エンジンの `cod_directional` {("c{from-1}","c{to-1}"): t(value)} に渡るが spec にキーが無い（§4） |
| `Conc … FRCO={value}` | `frco: int(value)` |
| `Conc … PUNISH.component[{component, schedule}]` | エンジンの `punish` {"c{k-1}": spec} に渡るが spec にキーが無い（§4） |
| `Conc … PUNISH.changeover` / `PUNISH.directional` | bridge がエラー。spec にも無い |
| `Alt(a, b)` / `Alt(a, b, c)` | `Alternative{first: a, second: b}` / `Alternative{first: Alternative{a,b}, second: c}` |
| `Conj(…)` | `Conjunctive{first, second}`、同じ左結合 |
| `Mult(a, b, …)` | `Multiple{components: [a, b, …], stimuli: ["s0", "s1", …]}` |
| `Mix(…)` | `Mixed{components}` |
| `Chain(…)` | `Chained{components, stimuli: ["s0", …]}` |
| `Tand(…)` | `Tandem{components}` |
| `Mult`/`Mix … BO={value,time_unit}` | なし（bridge は読まない。§4） |
| `Overlay(base, punisher)` | `Overlay{base, punisher}` |
| `Overlay … target="changeover"` | なし（bridge は読まない。§4） |
| `Interpolate(bg, block, count=, onset=)` | `Interpolate{base: bg, probe: block, interval: t(onset) または 1.0, probe_duration: interval/2, first_probe_at: t(onset) または null}`。`count` は使われない。これは bridge の暫定の当て嵌めで、DSL の意味（block 中に count 回強化したら戻る）とは別物（§4） |

### 3.5 修飾子

| AST (`Modifier.modifier`) | spec |
|---|---|
| `DRL{value}` | `DRL{interval: t(value)}` |
| `DRH{value}` | `DRH{response_count: 2, time_window: t(value)}`。DSL の DRH は IRT の閾値 1 個、エンジンは「窓内 N 応答」なので N=2 に固定 |
| `DRO{value}` | `DRO{interval: t(value), dro_type: "resetting"}`。DSL に momentary を書く手段が無い |
| `PR{pr_step: linear/hodos, pr_start, pr_increment}` | `ProgressiveRatio{step: "arithmetic", start: pr_start or 1, step_size: pr_increment or 1}` |
| `PR{pr_step: geometric/exponential, pr_start, pr_ratio}` | `ProgressiveRatio{step: "geometric", start: pr_start or 1, ratio: pr_ratio or 2.0}` |
| `Repeat{value: N, inner}`（解析前のみ） | `Chained{components: [inner × N], stimuli: [s0…]}`。解析後は `Tand` に脱糖されているので `Tandem` になる。同じ入力で結果の型が変わる点に注意 |
| `Lag{value, length}` | なし（エンジンにクラスが無い。§4） |
| `Pctl{pctl_target, pctl_rank, pctl_window, pctl_dir}` | `Percentile{target: pctl_target, rank, window: pctl_window or 20, direction: pctl_dir or "below"}`。DSL の target は `IRT`（大文字）で、Rust は `"irt"` しか受けないので、spec に書くときは小文字に正規化する必要がある |

### 3.6 高次・状態依存・嫌悪・試行

| AST | spec |
|---|---|
| `SecondOrder{overall: Atomic, unit: Atomic}` | `SecondOrder{overall, unit}` |
| `AdjustingSchedule{adj_target, adj_start, adj_step, adj_min, adj_max}` | `AdjustingSchedule{target: adj_target, start, step, minimum, maximum}`。値は `{value, time_unit}` で、target が `delay` のとき時間変換、`ratio`/`amount` は素の数。DSL の target は `delay`/`ratio`/`amount`、エンジンが実行できるのは `ratio`/`interval`。共通部分は `ratio` だけ（§4） |
| `InterlockingSchedule{interlock_R0, interlock_T}` | `InterlockingSchedule{initial_ratio: R0, decay_time: t(T)}` |
| `AversiveSchedule{kind: Sidman, params: {SSI, RSI}}` | `Sidman{shock_shock_interval: t(SSI), response_shock_interval: t(RSI), shock_magnitude: 1.0}`。bridge は `params.magnitude` も読むが schema に無い |
| `AversiveSchedule{kind: DiscrimAv, params: {CSUSInterval, ITI, mode, ShockDuration, MaxShock}}` | `DiscriminatedAvoidance{warning_duration: t(CSUSInterval), iti: t(ITI), shock_magnitude: 1.0}`。`mode`/`ShockDuration`/`MaxShock` は落ちる |
| `AversiveSchedule{kind: Escape, params: {SafeDuration, MaxShock}}` | `Escape{trial_duration: t(MaxShock) または 1e9, iti: t(SafeDuration), shock_magnitude: 1.0}` |
| `TrialBased{trial_type: MTS, comparisons, consequence, incorrect, ITI, ITI_unit, mts_type, limitedHold, limitedHoldUnit, delay}` | `MatchingToSample{n_comparisons: comparisons, sample_duration: t(delay) または 0.0, choice_timeout: t(limitedHold) または 5.0, consequence, incorrect, iti: t(ITI) または 0.0, seed}`。`mts_type`, `correction` は落ちる。bridge は `delay`/`delay_unit` を読むが schema の MTS 定義に `delay` は無い（文法にはある）。`iti` の既定が bridge（0.0）と spec（1.0）で違う |
| `TrialBased{trial_type: GoNoGo, responseWindow, consequence, incorrect, falseAlarm, ITI, limitedHold}` | `GoNoGo{go_probability: 0.5, response_window: t(responseWindow), iti: t(ITI) または 0.0, correct_go_schedule: consequence, correct_nogo_schedule: incorrect, false_alarm: falseAlarm, seed}`。`go_probability` は DSL に書く手段が無く常に 0.5。`limitedHold` は落ちる。`iti` の既定が bridge（0.0）と spec（5.0）で違う |

### 3.7 逆向き（spec にあって DSL に書けないもの）

- `ProgressiveRatio{step: "richardson_roberts"}`
- `DRO{dro_type: "momentary"}`
- `DRH{response_count ≠ 2}`
- `Percentile{direction: "above"}` は書ける（`percentile_dir`）が、`target` は `IRT` のみ
- `AdjustingSchedule{target: "interval"}`
- `Concurrent` のキー名が `c0, c1, …` 以外（fixture の `left`/`right`, `a`/`b`）
- `Multiple`/`Chained` の `stimuli` が `s0, s1, …` 以外
- `ResponseCost{initial_balance}`、`Interpolate` の `interval`/`probe_duration`/`first_probe_at` の自由な組、`GoNoGo{go_probability}`、`Sidman`/`DiscriminatedAvoidance`/`Escape` の `shock_magnitude`、`MatchingToSample{consequence: null}`（DSL は `consequence` 必須）
- `seed`/`n_intervals` は `@algorithm` で書ける

spec は DSL より広く、DSL は spec より広い（§4）。どちらも他方の部分集合ではないので、「DSL → spec」は非可逆の写像、「spec → DSL」は別途の整形が要る。

## 4. spec で表せない構文の一覧

| # | 構文 | 今の扱い | 起点 | 対処の方向（案） |
|---|---|---|---|---|
| 1 | `let` 束縛・`IdentifierRef` | 解析で置換。残っていれば bridge がエラー | DSL | spec は木であり共有を持たない。解析済み AST だけを受けると決める |
| 2 | `Repeat(N, s)` | 解析で `Tand` に脱糖。解析前なら `Chained` | DSL | 解析済みを前提にすれば `Tandem` に一本化 |
| 3 | 注釈（`@algorithm` 以外全部） | 無視 | DSL | 実行に関係ないので spec の外。`@algorithm` は `seed`/`n_intervals` に畳まれ、注釈としては残らない |
| 4 | `param_decls` の `COD`/`FRCO`/`BO`/`RD`（Program 既定値） | 無視 | DSL | 解析 Phase 3 が `LH` と同様に各ノードへ展開するのが筋。spec 側は何も足さない |
| 5 | `Mult`/`Mix` の `BO`（ブラックアウト） | 無視 | DSL | エンジンにも無い。足すなら両エンジン + spec キー `blackout` |
| 6 | `Overlay(target="changeover")` | 無視 | DSL | エンジンにも無い |
| 7 | `Conc` の方向別 `COD(1->2)` | エンジンには渡る（`cod_directional`）が spec に無い | spec | spec に `cod_directional: [{"from": "c0", "to": "c1", "cod": 秒}]` を足す（fixture も） |
| 8 | `Conc` の `PUNISH(k)`（成分別） | エンジンには渡る（`punish`）が spec に無い | spec | spec に `punish: {"c0": spec}` を足す |
| 9 | `Conc` の `PUNISH(changeover)`/`PUNISH(1->2)` | bridge がエラー、エンジンにも無い | 両方 | 保留 |
| 10 | `Timeout(contingent_response=…)` | 無視 | 両方 | 保留 |
| 11 | `Interpolate(count=, onset=)` | `count` 無視、`onset` から幾何を当て推量 | 両方 | エンジンの Interpolate を「count 回強化で戻る」意味に合わせるか、spec に `count`/`onset` を足して bridge の当て推量をやめるか、どちらかを決める |
| 12 | `Lag N`（`length=`） | bridge がエラー | 両方 | エンジンにクラスが無い |
| 13 | `Pctl` の `IRT` 以外の target（`latency`, `duration`, `force`, `rate`） | Python は構築可・step で例外、Rust は構築時 panic | 両方 | spec の `target` は `"irt"` のみと明記 |
| 14 | `Adj(delay, …)`, `Adj(amount, …)` | Python は構築時エラー、Rust も conformance.rs で panic | 両方 | spec の `target` は `"ratio"`/`"interval"`。DSL に `interval` が無いのは DSL 側の課題 |
| 15 | `DiscrimAv` の `mode`/`ShockDuration`/`MaxShock` | 無視 | 両方 | エンジンは応答終了型・点電撃に固定 |
| 16 | `Sidman` の電撃強度 | DSL に無い（bridge は schema 外の `magnitude` を読む） | DSL | `shock_magnitude` は spec 側だけの値 |
| 17 | `MTS` の `type=identity|arbitrary`, `correction=`, `delay=`（schema 未記載） | 無視（`delay` は bridge が読む） | 両方 | DSL schema に `delay` を足す。`mts_type`/`correction` はエンジン未対応 |
| 18 | `GoNoGo` の `LH=`、`go_probability` の指定手段 | `LH` 無視、確率は 0.5 固定 | 両方 | DSL に go 確率の書き方が無い |
| 19 | `PR 5` の短縮形（linear, start=5, increment=5） | 文法のみ。AST では `linear` に展開されるはず | DSL | 対応は PR 行と同じ |
| 20 | `DRH10/60s` 形（文法の例に出るが schema の DRModifier は `value` 1 個） | 不明（DSL conformance ではエラー例として出る） | DSL | `DRH{response_count, time_window}` に写せる唯一の形なので、DSL に正式に入れるなら spec と一致する |
| 21 | レスポンデント層（`Pair.*`, `CSOnly`, …）、`Experiment`/`Paper`、`PhaseSequence` | 対応物なし | 両方 | spec は 1 本のオペラント スケジュールの形。相の列は Python の `PhasedConcurrent` が近いが spec も Rust も無い |
| 22 | 時間の単位 | spec は単位なし | 仕様 | 単位は DSL→spec の段階で秒に潰す。spec に `time_unit` を足さない（fixture が全部秒で、Rust のクロックも単位なしの f64） |

「起点」列: DSL = DSL にだけある構文、spec = エンジンにはあって spec に無い、両方 = DSL とエンジンの意味がずれている。

## 5. 決めること（候補と推し）

1. **`Percentile.direction` の既定**: spec/エンジンは `above`、DSL と bridge は `below`。推し: spec の既定は `above` のまま、DSL→spec の変換は `direction` を必ず明示して書く（既定に頼らない）。
2. **`Percentile.target` と `AdjustingSchedule.target` の正規化**: spec は小文字 `irt` / `ratio` / `interval` のみ。Rust の受理集合に合わせる。
3. **`DRO` のキー名**: `dro_type` に統一し、hil.rs の `mode` を直す（S-215 の作業に含める）。
4. **整数の表記**: JSON 整数で書く（Rust `as_u64` 準拠）。
5. **省略可の入れ子**: 「キー無し」と `null` を同義とする（両エンジンとも既にそう）。
6. **`Concurrent` の拡張キー**: `cod_directional` と `punish` を spec に足し、fixture を 1 本ずつ追加する（両エンジンに実装はある）。
7. **`Interpolate`**: 案 11 のどちらかを決めるまで、DSL→spec は「`count`/`onset` を落として今の当て推量を使う」と明記する。
8. **`MatchingToSample`/`GoNoGo` の `iti` 既定**: spec（1.0 / 5.0）と bridge（0.0）で違う。DSL は `ITI` 必須なので実害は無いが、spec の既定を正とする。
9. **組み立て関数の置き場**: contingency-rs に `ScheduleSpec`（serde）と `build_from_spec` を 1 つ置き（`src/builder.rs` の隣か `src/spec.rs`）、`tests/conformance.rs`・`src/bin/hil.rs`・PyO3（`Schedule.from_spec(json)`）がそれを呼ぶ。contingency-py の `spec.py` は Rust 正本化後にその薄い包みになる。`LimitedHold` の内側の型分岐（FI/VI/RI）はこの関数の中に閉じる。

## 6. S-215（HIL の通信形式）と重なる点

contingency-rs `src/bin/hil.rs` の現状:

- 起動時に `--schedule '<JSON>'` または `--schedule-file` で spec を受け取り、`build_schedule` が自前の 8 分岐（`CRF`, `EXT`, `FR`, `FI`, `FT`, `DRO`, `DRL`, `DRH`）で組み立てる。形は適合テストの `"schedule"` と同じで、`DRO` のキーだけ `mode`（spec は `dro_type`）。
- 通信は TCP の JSONL で、`type` キーのメッセージ: 受信 `response{time, operandum}` / `stimulus` / `reinforcer`（無視）/ `disconnect`、送信 `reinforcer{time, channel, magnitude, label}`。contingency-py の `hw/hil_bridge.py`（`HilBridgeApparatus`）はこの形の対になっており、`tests/hil_integration.rs` は `{"type":"FR","params":{"n":3}}` で起動して確かめている。
- S-215 の注記にある experiment-io の `HilBridge`（標準入出力、`cmd`/`event`）は本調査の 3 repo には無く、ここでは比べていない。

重なり:

1. **spec はそのまま「スケジュール設定」のペイロードになる。** どちらの通信形式に寄せるにせよ、接続後に送る設定メッセージ（例: `{"type": "configure", "schedule": <spec>}` あるいは `cmd: configure`）の中身は本書の spec でよい。hil.rs の `--schedule` は起動引数で同じものを受けている。
2. **操作体名の一致。** spec の `Concurrent.components` のキー（DSL 由来なら `c0, c1, …`、単一操作体なら既定 `main`）と、`response` メッセージの `operandum` が同じ文字列でなければならない。hil.rs は `operandum` 省略時に `main` を入れる。試行型は `choice_0 …` を使う。この対応表は通信仕様の一部として S-215 に載せる。
3. **時間の基準。** spec の時間は単位なしの実数、`response.time` も f64 で、同じクロックを前提にしている。単位変換は DSL→spec の段階で済んでおり、通信層では変換しない。
4. **組み立て関数の一本化。** hil.rs の 8 分岐は §5-9 の `build_from_spec` に置き換えれば 35 種類に広がる。S-215 で通信形式を直すとき、同時に hil.rs の `build_schedule` を削って共通関数を呼ぶ形にするのがよい。
5. **spec の外側の情報。** 強化子の `channel`（hil.rs は `--channel`、既定 `food`）と弁別刺激の出力先は spec に無い。これらは通信形式（S-215）側の設定項目であり、spec に入れない。

## 7. 参照した場所

- contingency-py: `src/contingency/conformance/spec.py`, `src/contingency/bridge/__init__.py`, `src/contingency/schedules/*.py`（コンストラクタの既定値）, `conformance/README.md`, `conformance/**/*.json`（39 本）, `conformance-index.json`, `src/contingency/hw/hil_bridge.py`
- contingency-rs: `tests/conformance.rs`, `src/builder.rs`, `src/python.rs`, `src/bin/hil.rs`, `src/schedules/*.rs`（`new` の引数と enum）, `tests/hil_integration.rs`
- contingency-dsl: `schema/operant/ast.schema.json`, `ast-parsed.schema.json`, `ast-resolved.schema.json`, `schema/operant/stateful/ast.schema.json`, `schema/operant/trial-based/ast.schema.json`, `schema/operant/grammar.ebnf`, `schema/operant/{stateful,trial-based}/grammar.ebnf`, `conformance/**/*.json`（入力例）

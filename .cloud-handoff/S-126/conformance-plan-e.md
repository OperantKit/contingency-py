# (e) 拡張 — fixture の無い 15 クラスの棚卸しと spec 対応表

取り込み先: 手元の `conformance-plan.md` の「(e) 拡張」節。fixture の形式（(a)）は変えない前提で書いた。
すべて contingency-py / contingency-rs の main（py `bc29fd8`、rs `d96582a`）を実測した結果。

## 1. 前提の実測（TODO の文面との差分）

- **36 クラスの内訳**: `src/contingency/schedules/__init__.py` の `__all__` は 36 名だが、`grep '^class '` で見つかるスケジュールは 35 クラス（`interval.py` の `_ArmableSchedule` は Protocol で除外）。残り 1 名の `CRF` は `FR(1)` を返す関数。fixture のある 21 型（FR / CRF / EXT / FI / FT / VR / VI / VT / RR / RI / RT / LimitedHold / Concurrent / Chained / Multiple / Tandem / Alternative / DRO / DRL / DRH / ProgressiveRatio）を引いた残り 15 が本節の対象で、TODO の列挙と一致する。
- **builder の実体**: `build_schedule_from_spec` は `scripts/generate_conformance.py` ではなく `src/contingency/conformance/spec.py` にある（generator と `contingency-conformance replay-python` が同じ関数を import する）。S-127〜S-129 の 範囲 はこのパスに直した。
- **型名の規約**: spec の `type` は Python のクラス名そのまま（`LimitedHold`、`ProgressiveRatio`、`Concurrent`）。`params` のキーはコンストラクタのキーワード名そのまま、入れ子のスケジュールは `{"type":…,"params":…}` をそのキーワード名（`inner` / `first` / `second` / `components`）の下に置く。本節の 15 クラスもこの規約に従う。
- **確率的の判定が py と rs で違う**: py の `is_stochastic` は spec の木のどこかに `seed` param があるか。rs の `tests/conformance.rs` の `schedule_is_stochastic` は型名が `STOCHASTIC_TYPES = [VR, VI, RI, RR, VT, RT]` に含まれるか。MatchingToSample / GoNoGo は `seed` を持つので py は stochastic、rs は deterministic と見なす。rs 側のリストに 2 型を足さないと `conformance_corpus_completeness` が `run_fixture` を要求して落ちる（S-129 の分岐）。
- **rs の completeness guard は py の `conformance/` を直接走査する**: `../contingency-py/conformance/**/*.json` のうち deterministic なものすべてに `run_fixture("…")` 呼び出しが要る。したがって py に deterministic fixture を 1 本足すたびに、rs の `build_schedule` の分岐と `run_fixture` のテスト 1 本が同時に要る（足さないと rs のテストが赤になる）。S-127〜S-129 は py と rs の両 repo を触る作業として書き換えた。
- **rs の `run_fixture` は `meta` を比べない**（`reinforced` と `reinforcer` の 3 欄のみ）。本節の試作では meta も比べ、15 クラスのうち決定的な 13 クラスで一致した（後述）。
- **モジュール数**: rs `src/schedules/` は `mod.rs` を除いて 20 モジュール、py `src/contingency/schedules/` も `__init__.py` を除いて 20。ファイル名は 1 対 1 で対応し、15 クラスすべてに rs の同名実装がある。

## 2. 対応表 (a)(b)(c): spec の形・決定性・step 形式での表現

| クラス | (a) spec `type` と `params`（py コンストラクタ） | (b) 決定性 | (c) 既存の `steps` 形式（`now` + 反応事象 or `null`）で表せるか |
|---|---|---|---|
| Mixed | `Mixed` / `components: [spec, …]`（2 本以上） | 決定的（成分による） | 表せる。`Multiple` と同じ手順。meta `current_component` は int。 |
| Timeout | `Timeout` / `inner: spec`, `duration`, `reset_on_response`(省略時 false) | 決定的（inner による） | 表せる。TO 中の反応は捨てられ meta `timeout_active`、強化の step は meta `timeout_started`。 |
| Conjunctive | `Conjunctive` / `first: spec`, `second: spec` | 決定的（成分による） | 表せる。両成分が同じ step で強化したときだけ強化、meta `conjunctive`。注意: FI は最初の step ではなく絶対時刻 `interval` で arm され、reset 後は `arm_time = interval` に戻る（py / rs とも同じ）。`Conjunctive(FR3, FI5)` に 1 秒刻みで反応し続けると FR は 3,6,9…、FI は 5,10… で噛み合わず一度も強化しない。試作では反応を 1,2,5,6,…,13 に置き、5 と 11 で同時強化を得た。 |
| InterlockingSchedule | `InterlockingSchedule` / `initial_ratio`(int), `decay_time` | 決定的 | 表せる。要求数は `int(R0 * (1 - t/T))`、反応がある step だけ強化。最初の step（tick でよい）が anchor。 |
| ResponseCost | `ResponseCost` / `inner: spec`, `amount`, `unit`(省略時 "token"), `initial_balance`(省略時 null = 無制限) | 決定的（inner による） | 表せる。残高不足の反応は meta `cost_insufficient` と `unit`。`initial_balance` を省くと何も起きないので fixture では有限値を与える。 |
| AdjustingSchedule | `AdjustingSchedule` / `target`("ratio" or "interval"), `start`, `step`, `minimum`?, `maximum`? | 決定的 | 表せる。`delay` / `amount` は py が構築時に例外（rs は enum に残るが実行不可）なので fixture には使わない。 |
| Interpolate | `Interpolate` / `base: spec`, `probe: spec`, `interval`, `probe_duration`, `first_probe_at`?(省略時 interval) | 決定的（成分による） | 表せる。毎 step に meta `in_probe`（bool）。最初の step が anchor。 |
| Overlay | `Overlay` / `base: spec`, `punisher: spec` | 決定的（成分による） | 表せる。punisher 側は magnitude が負で label `SR-`、両方同時は差額で `SR+`/`SR-`/`SR0` と meta `net_magnitude`（float、9 桁丸め）。 |
| Percentile | `Percentile` / `target`("irt"), `rank`(int), `window`(int), `direction`("above" or "below") | 決定的（乱数なし。IRT の分位点比較のみ） | 表せる。最初の `window+1` 反応は基準づくりで強化しない。meta に `percentile_threshold` / `percentile_irt`（float）/ `percentile_rank` / `percentile_direction`。注意: rs の `rank` は `u8` で 1..=100、py は 0..=100。fixture は 1〜100 を使う。`target` は両港とも irt のみ実行可。 |
| SecondOrder | `SecondOrder` / `overall: spec`, `unit: spec` | 決定的（成分による） | 表せる。unit 完了の step は meta `unit_completed`、overall が強化する step は meta `second_order`。内部の合成反応（operandum `__unit_completion__`）は fixture には現れない。 |
| Sidman | `Sidman` / `shock_shock_interval`, `response_shock_interval`, `shock_magnitude`(省略時 1.0) | 決定的 | 表せる。電撃は**出力**（`reinforced: true`、`reinforcer.magnitude` が負、label `SR-`、meta `aversive_event: "shock"`）なので、入力側に電撃事象は要らない。tick だけで電撃が出る。 |
| DiscriminatedAvoidance | `DiscriminatedAvoidance` / `warning_duration`, `iti`, `shock_magnitude`(省略時 1.0) | 決定的 | 表せる。警告刺激の提示は内部の相（meta `phase: "iti"/"warning"`）として出力され、入力事象は不要。回避は meta `aversive_event: "avoidance"` と `trial_outcome: "avoid"`、電撃は SR-。注意: 最初の ITI は最初の step ではなく絶対時刻 0 から数える（`phase_end = iti`）ので fixture は `now = 0` から始める。 |
| Escape | `Escape` / `trial_duration`, `iti`, `shock_magnitude`(省略時 1.0) | 決定的 | 表せる。電撃中は tick ごとに SR-（最初の tick は meta `aversive_event: "shock_onset"`）、反応で終了（`shock_termination`、SR- なし）、時間切れは `trial_end`。最初の試行も絶対時刻 0 から始まるので `now = 0` から。 |
| MatchingToSample | `MatchingToSample` / `n_comparisons`(int ≥ 2), `sample_duration`, `choice_timeout`, `consequence`?: spec, `incorrect`?: spec, `iti`, `seed` | **確率的**（正しい比較刺激を試行ごとに `random.Random(seed)` で抽選） | 表せる。見本・比較刺激の提示は内部の相（meta `phase: "sample"/"choice"/"iti"`）として時間で進み、入力は `operandum: "choice_k"` の反応だけ。選択の結果は meta `trial_outcome`（"correct"/"incorrect"）。meta の `trial_outcome: null` と `inner_meta`（dict）は serializer が落とすので fixture には出ない。seed 付きで py は再現、rs は PRNG が違うので緩い検査（`run_stochastic_fixture`）。 |
| GoNoGo | `GoNoGo` / `go_probability`, `response_window`, `iti`, `correct_go_schedule`?: spec, `correct_nogo_schedule`?: spec, `false_alarm`?: spec, `seed` | **確率的**（Go/NoGo を試行ごとに抽選） | 表せる。相は meta `phase: "go"/"nogo"/"iti"`、結果は `trial_outcome`（"correct"/"incorrect"/"false_alarm"/"correct_withhold"）。seed 付きで py は再現、rs は緩い検査。 |

結論: **形式拡張が要るクラスは 0**。試行型も嫌悪制御も、刺激提示・電撃・試行開始は内部状態の出力（meta と負の強化子）であり、入力は反応事象と tick だけで足りる。S-129 の blocked 条件（形式拡張の可否）は発生しない見込み。ただし次の 2 点は S-129 の 引き継ぎ に書く価値がある（形式の拡張ではなく任意の改善）:

1. MatchingToSample では「正しい比較刺激」が fixture から観測できないので、「正答を選ぶ」系列は書けない（固定の `choice_0` を選び続けて、当たり外れは seed に従う）。もし「正答を選んだときの帰結」を言語間で厳密に比べたければ、クラス側に `correct_operandum` を meta へ出す変更（fixture 形式は不変）か、rs 側で py の Mersenne Twister を再現する変更が要る。
2. rs の `run_stochastic_fixture` は強化子数が期待の 50%〜150% に入るかしか見ないので、試行が 5 回程度だと帯域が自明に通る（試作では期待 1〜2 本）。S-129 の fixture は試行 20 回以上にする。

## 3. 対応表 (d): contingency-rs の実装と `build_schedule` に足す分岐

| クラス | rs モジュール / 型 | rs コンストラクタ（`contingency::schedules`） | `tests/conformance.rs` の `build_schedule` に足す分岐（試作で動作確認済み） |
|---|---|---|---|
| Mixed | `mixed.rs` / `Mixed` | `Mixed::new(Vec<Box<dyn Schedule>>) -> Result` | `"Mixed" => Mixed::new(build_component_list(params))` |
| Timeout | `timeout.rs` / `Timeout` | `Timeout::new(inner, duration: f64, reset_on_response: bool) -> Result` | `"Timeout" => Timeout::new(build(inner_spec("inner")), require_f64("duration"), params["reset_on_response"].as_bool().unwrap_or(false))` |
| Conjunctive | `conjunctive.rs` / `Conjunctive` | `Conjunctive::new(first, second) -> Result` | `"Conjunctive" => Conjunctive::new(build(first), build(second))`（`Alternative` と同形） |
| InterlockingSchedule | `interlocking.rs` / `InterlockingSchedule` | `InterlockingSchedule::new(initial_ratio: u64, decay_time: f64) -> Result` | `"InterlockingSchedule" => InterlockingSchedule::new(require_u64("initial_ratio"), require_f64("decay_time"))` |
| ResponseCost | `response_cost.rs` / `ResponseCost` | `ResponseCost::new(inner, amount: f64, unit: String, initial_balance: Option<f64>) -> Result` | `"ResponseCost" => ResponseCost::new(build(inner), require_f64("amount"), str("unit").unwrap_or("token").to_string(), optional_f64("initial_balance"))` |
| AdjustingSchedule | `adjusting.rs` / `AdjustingSchedule`, `AdjustingTarget{Ratio, Interval, Delay, Amount}` | `AdjustingSchedule::new(target, start: f64, step: f64, minimum: Option<f64>, maximum: Option<f64>) -> Result` | `"AdjustingSchedule" => …`（`target` 文字列を enum に写す。"ratio"→Ratio、"interval"→Interval、他は panic） |
| Interpolate | `interpolate.rs` / `Interpolate` | `Interpolate::new(base, probe, interval: f64, probe_duration: f64, first_probe_at: Option<f64>) -> Result` | `"Interpolate" => Interpolate::new(build(base), build(probe), require_f64("interval"), require_f64("probe_duration"), optional_f64("first_probe_at"))` |
| Overlay | `overlay.rs` / `Overlay` | `Overlay::new(base, punisher) -> Result` | `"Overlay" => Overlay::new(build(base), build(punisher))` |
| Percentile | `percentile.rs` / `Percentile`, `PercentileTarget{Irt}`, `PercentileDirection{Above, Below}` | `Percentile::new(target, rank: u8, window: usize, direction) -> Result` | `"Percentile" => Percentile::new(PercentileTarget::Irt, require_u64("rank") as u8, require_u64("window") as usize, dir)`（"above"/"below" を enum に写す） |
| SecondOrder | `second_order.rs` / `SecondOrder` | `SecondOrder::new(overall, unit) -> Self`（Result ではない） | `"SecondOrder" => SecondOrder::new(build(overall), build(unit))` |
| Sidman | `aversive.rs` / `Sidman` | `Sidman::new(ssi: f64, rsi: f64, shock_magnitude: f64) -> Result` | `"Sidman" => Sidman::new(require_f64("shock_shock_interval"), require_f64("response_shock_interval"), optional_f64("shock_magnitude").unwrap_or(1.0))` |
| DiscriminatedAvoidance | `aversive.rs` / `DiscriminatedAvoidance` | `DiscriminatedAvoidance::new(warning_duration, iti, shock_magnitude) -> Result` | 同形（`warning_duration`, `iti`, `shock_magnitude` 省略時 1.0） |
| Escape | `aversive.rs` / `Escape` | `Escape::new(trial_duration, iti, shock_magnitude) -> Result` | 同形（`trial_duration`, `iti`, `shock_magnitude` 省略時 1.0） |
| MatchingToSample | `trial_based.rs` / `MatchingToSample`（`SmallRng::seed_from_u64`） | `MatchingToSample::new(n_comparisons: u32, sample_duration, choice_timeout, consequence: Option<Box>, incorrect: Option<Box>, iti, seed: Option<u64>) -> Result` | `"MatchingToSample" => …`（省略時の既定値は py と同じ: sample_duration 0.0 / choice_timeout 5.0 / iti 1.0）。加えて **`STOCHASTIC_TYPES` に `"MatchingToSample"` を足す**（足さないと completeness guard が `run_fixture` を要求する）。テストは `run_stochastic_fixture` + `#[ignore]`。 |
| GoNoGo | `trial_based.rs` / `GoNoGo`（`SmallRng::seed_from_u64`） | `GoNoGo::new(go_probability, response_window, iti, correct_go_schedule: Option<Box>, correct_nogo_schedule: Option<Box>, false_alarm: Option<Box>, seed: Option<u64>) -> Result` | `"GoNoGo" => …`（既定値 0.5 / 5.0 / 5.0）。**`STOCHASTIC_TYPES` に `"GoNoGo"` を足す**。テストは `run_stochastic_fixture` + `#[ignore]`。 |

補足:
- rs 側の分岐の実物は `.cloud-handoff/S-126/proto/rsproto/src/main.rs` の `build` にある（既存ヘルパ `require_f64` 等と同じ流儀で書き直せば `tests/conformance.rs` に移せる）。
- 入れ子の `Option<Box<dyn Schedule>>`（MTS の `consequence` など）は、spec で省略 or `null` のとき `None`。
- 13 の決定的クラスは `run_fixture`（厳密）で py と一致した。rs の `run_fixture` は meta を見ないが、試作では meta も比べて一致を確認した（`Timeout` の `timeout_active`、`Overlay` の `net_magnitude`、`Percentile` の float 2 つ、嫌悪制御の `phase` / `aversive_event` / `trial_outcome` を含む）。

## 4. 群分けと順序

| 群 | ストーリー | クラス（実名） | 根拠 |
|---|---|---|---|
| 第 1 群 | S-127 | Mixed / Timeout / Conjunctive / InterlockingSchedule / ResponseCost | 5 クラスとも決定的で、入れ子は既存型（FR / FI）だけ。rs の分岐が 1〜2 行。TODO の候補どおり。 |
| 第 2 群 | S-128 | AdjustingSchedule / Interpolate / Overlay / Percentile / SecondOrder | 5 クラスとも**決定的**（TODO の候補は「乱数を含む」想定だったが、15 クラスで乱数を持つのは MTS / GoNoGo の 2 つだけ）。状態が多く meta に float を含む（Overlay `net_magnitude`、Percentile の 2 値）ので 9 桁丸めの検算が要る。rs 側は enum の写し（AdjustingTarget / PercentileTarget / PercentileDirection）が要る。S-128 の「seed 付き・stochastic 記載」の条項は成分に seeded VR/VI 等を入れたときだけ意味を持つ。 |
| 第 3 群 | S-129 | Sidman / DiscriminatedAvoidance / Escape / MatchingToSample / GoNoGo | 嫌悪制御 3 つは決定的で既存形式で表せる（負の強化子で表す）。試行型 2 つは seed 付きの確率的。形式拡張は不要なので blocked にはならない見込み。rs の `STOCHASTIC_TYPES` 拡張と `run_stochastic_fixture` の新規テスト 2 本が要る。 |

順序は S-127 → S-128 → S-129。各群は独立（依存なし）だが、README の本数の書き換えが重なるので直列に進める。

fixture の置き場所（案）: 既存の 4 ディレクトリ（atomic / compound / differential / progressive）に加え、`wrapper/`（Timeout / ResponseCost）、`aversive/`（3 つ）、`trial/`（MTS / GoNoGo）を増やす。rs の guard と py の `rglob` はどちらも再帰で拾うので、ディレクトリ名は自由。`LimitedHold` は `atomic/` にあるので、wrapper を `atomic/` に置く選択もある（S-127 で決める）。

## 5. README の本数が書かれている場所（S-127〜S-129 で更新）

- contingency-py `README.md`: 68 行目（"22 JSON fixtures"）、74 行目（"the same 22 fixtures"）、129 行目（"the other 21 fixtures"、"The 5 compound fixtures"）、131 行目（"passes all 22 fixtures"）、133 行目（"Six fixtures (seeded VR, VI, VT, RR, RI, RT)" — S-129 で MTS / GoNoGo が増えると 8）。`README.ja.md` は同じ行番号に同じ数。
- contingency-py `conformance/README.md`: Layout の木（既に multiple_fr_fr / tandem_fr_fr が抜けている）、"Supported types" の箇条書き、"Stochastic fixtures (seeded VR, VI, VT, RR, RI, RT)" の文。
- contingency-rs `README.md`: 5 行目（"22 conformance fixtures"）、77 行目（"22 JSON fixtures"、"16 deterministic"、"6 seeded"）。`README.ja.md` も同じ構成。
- contingency-rs `tests/conformance.rs` 冒頭の doc comment（決定的 / 確率的の型の列挙）。

## 6. 試作の証跡

試作は `.cloud-handoff/S-126/proto/` に置いた（tracked のソース・テスト・fixture は未変更）。

- `proto/proto_gen.py`: `build_schedule_from_spec` を 15 型ぶん広げ、`scripts/generate_conformance.py` の `_run_fixture` / `_write` をそのまま使って候補 fixture 15 本を `proto/fixtures/` に生成する。同じ spec を 2 回走らせて JSON が一致すること（決定性）も確認する。
- `proto/fixtures/**/*.json`: 候補 fixture 15 本（既存形式そのまま。float 9 桁丸め、meta は bool/int/str/float のみ）。S-127〜S-129 はこれを出発点にしてよい（時刻の系列と `description` は見直し可）。
- `proto/rsproto/`: contingency-rs を path 依存で読む捨てクレート。`build` に 15 分岐を持ち、候補 fixture を厳密（reinforced / reinforcer）と meta の両方で比べる。`Cargo.toml` の path は手元に合わせて直す。

Python 側（`.venv/bin/python .cloud-handoff/S-126/proto/proto_gen.py`）:

```
fixture                                      type                     steps reinf kind          same meta
compound/mixed_fr_fr.json                    Mixed                        7     3 deterministic True  current_component
wrapper/timeout_fr.json                      Timeout                      9     3 deterministic True  timeout_active,timeout_started
compound/conjunctive_fr_fi.json              Conjunctive                 11     2 deterministic True  conjunctive
atomic/interlocking_basic.json               InterlockingSchedule        12     3 deterministic True
wrapper/response_cost_fr.json                ResponseCost                 6     1 deterministic True  cost_insufficient,unit
progressive/adjusting_ratio.json             AdjustingSchedule           15     4 deterministic True
compound/interpolate_fr_ext.json             Interpolate                 21     7 deterministic True  in_probe
compound/overlay_fr_fr.json                  Overlay                      9     6 deterministic True  base_fired,net_magnitude,punisher_fired
differential/percentile_irt.json             Percentile                  13     3 deterministic True  percentile_direction,percentile_irt,percentile_rank,percentile_threshold
compound/second_order_fr_fr.json             SecondOrder                  9     1 deterministic True  second_order,unit_completed
aversive/sidman_basic.json                   Sidman                       8     3 deterministic True  aversive_event
aversive/discriminated_avoidance_basic.json  DiscriminatedAvoidance       8     1 deterministic True  aversive_event,phase,trial_outcome
aversive/escape_basic.json                   Escape                       7     5 deterministic True  aversive_event,phase,trial_outcome
trial/mts_seeded_1.json                      MatchingToSample            15     2 stochastic    True  phase,trial_index,trial_outcome
trial/gonogo_seeded_3.json                   GoNoGo                      16     1 stochastic    True  phase,trial_index,trial_outcome
```

Rust 側（`cargo run -q --manifest-path proto/rsproto/Cargo.toml -- proto/fixtures`）:

```
fixture                                        exp  rust strict   meta first mismatch
atomic/interlocking_basic.json                   3     3   true   true
aversive/discriminated_avoidance_basic.json      1     1   true   true
aversive/escape_basic.json                       5     5   true   true
aversive/sidman_basic.json                       3     3   true   true
compound/conjunctive_fr_fi.json                  2     2   true   true
compound/interpolate_fr_ext.json                 7     7   true   true
compound/mixed_fr_fr.json                        3     3   true   true
compound/overlay_fr_fr.json                      6     6   true   true
compound/second_order_fr_fr.json                 1     1   true   true
differential/percentile_irt.json                 3     3   true   true
progressive/adjusting_ratio.json                 4     4   true   true
trial/gonogo_seeded_3.json                       1     1   true  false step 1: meta[phase] exp="iti" got=Some(String("go"))
trial/mts_seeded_1.json                          2     2  false  false step 2 (now=2): reinforced exp=true got=false ...
wrapper/response_cost_fr.json                    1     1   true   true
wrapper/timeout_fr.json                          3     3   true   true
```

読み方: 決定的 13 本は rs が py と同じ `reinforced` / `reinforcer` / meta を返す。seed 付き 2 本は PRNG の違いで軌跡が違う（期待どおり）。既存のベースラインは py `pytest tests/test_conformance.py tests/test_conformance_cli.py` が 53 passed、`contingency-conformance run -- contingency-conformance replay-python` が 22 passed、rs `cargo test --test conformance` が 17 passed / 6 ignored。

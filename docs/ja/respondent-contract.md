# レスポンデントの実行の契約

:gb: [English](../en/respondent-contract.md)

contingency-dsl は、レスポンデント（パヴロフ型、CS–US）の手続きを 2 つの段に分けている。Tier A は DSL のレスポンデント層にある少数の基本要素である（一覧は contingency-dsl の `spec/en/respondent/primitives.md` を正とし、このページには写さない）。Tier B は、それらを試行や段階をまたいで組み合わせた手続き（阻止、復元、潜在制止など）で、レスポンデントの拡張点（contingency-dsl `design-philosophy.md` §5.4）を通して姉妹パッケージ contingency-respondent-dsl が定義する。

実行の側も同じ分け方に従う:

| 段 | 構文の置き場 | 展開の置き場 |
|---|---|---|
| A | contingency-dsl（レスポンデント層） | contingency-rs（このパッケージが公開する） |
| B | contingency-respondent-dsl | contingency-respondent-py（プログラムごとに登録する） |

このページは「Tier A の式を展開する」と何が返るかを固定する。contingency-rs・このパッケージ・ほかの実装は、`conformance/respondent/` の同じファイルで照らし合わせられる。

## 拡張の 5 つの性質を実行の側に写した形

contingency-dsl §5.4 は、レスポンデントの拡張点の性質を 5 つ挙げている。それぞれの実行の側での読み方:

| DSL の性質 | 実行の側 |
|---|---|
| 1. レスポンデントの文法は変えない。拡張は生成規則を足すだけ。 | Tier A の展開はこの契約で固定する。拡張は Tier A の式の展開を変えず、自分の基本要素の展開を足すだけである。 |
| 2. プログラムごとに閉じる。どの拡張の基本要素を認めるかは、各プログラムの登録表が決める。 | Tier B の展開は、プログラムごとに作る登録表で足す（DSL の `ExtensionRegistry` と同じ作法）。全体への登録は無い。プログラムの登録表に無い名前の式は誤りとして返し、黙って何もしないことはしない。 |
| 3. Tier A は完全に静的に検証できる。拡張は自分の規則を持つ。 | Tier A の展開は、走っているセッションの状態を要らない。Tier B の展開は、中に入れ子になった式に Tier A の展開を呼んでよく、自分の引数は自分で検証する。 |
| 4. 拡張は実行時の状態を使ってよいが、レスポンデント層そのものはチューリング完全性を持ち込まない。 | Tier A の展開は、式とプログラムの設定から計算する有限の列である。セッション中の反応に応じて変わるものは、拡張かセッションの実行器が受け持つ。 |
| 5. 拡張の構成要素どうしの同値の判定は拡張の責任。 | 2 つの Tier B の式が同値かは、それを定義した拡張が判定する。Tier A の fixture が固定するのは Tier A の出力だけである。 |

## Tier A の展開の結果

Tier A の式 1 つを展開すると、試行の列が返る。各試行は、次のキーをちょうど持つ JSON のオブジェクトである:

| キー | 型 | 意味 |
|---|---|---|
| `stage` | 文字列 | 単独の Tier A の式では `"main"`。段階の名前は、外側の段階か Tier B の手続きが付ける。 |
| `cs` | 文字列の列（整列済み） | その試行で提示する CS のラベル。US だけの試行・刺激の無い区間では空。 |
| `us` | 文字列か null | US のラベル。US を出さないときは null。 |
| `context` | 文字列か null | Tier A では常に null（文脈を名指す Tier A の基本要素は無い）。 |
| `order` | 文字列の列か null | `Serial` の CS の提示順。それ以外は null。 |
| `time` | オブジェクト | 試行の中の時刻。試行の始めからの秒（下記）。 |
| `iti_s` | 数か null | この試行の前の間隔（秒）。`ITI` だけが入れる。 |

`time` のキーは `cs_on`・`cs_off`・`us_on`・`us_off`・`gap`。式が決めない時刻は null にする（たとえば痕跡条件づけの対の CS の長さは装置に任されている）。`gap` は null か `{"from", "to", "s", "at_least"}` のオブジェクトで、2 つの時刻の絶対値が決まらないときに、その間の長さを固定する。たとえば 5 秒の痕跡は `{"from": "cs_off", "to": "us_on", "s": 5.0, "at_least": false}`。`at_least: true` は「これより短くしない」の意味。`from: "any_cs"` は `ExplicitlyUnpaired` だけが使い、この試行の US の始めと、列の中のすべての CS との距離（前後どちらも、試行をまたいで）を縛る。長さは秒に直す。単位は `ms`・`s`・`min`・`h` のどれかでなければならない。

### 要素ごと

試行の数は、要素が自分の `trials` 引数を持つときはその値、無いときはプログラムの設定の `trials`。

| 要素 | 試行の数 | 各試行（または 2 試行の組） |
|---|---|---|
| `Pair.ForwardDelay` | n | CS（ラベルか同時の `Compound`）と US。`cs_on` 0、`cs_off` = `cs_duration`、`us_on` = `isi`。 |
| `Pair.ForwardTrace` | n | CS と US。`cs_on` 0、`gap` は CS の終わり → US の始め = `trace_interval`。`cs_duration` があれば `cs_off` と `us_on` が決まり、無ければ null。 |
| `Pair.Simultaneous` | n | CS と US。`cs_on` = `us_on` = 0。 |
| `Pair.Backward` | n | CS と US。`us_on` 0、`gap` は US の終わり → CS の始め = `isi`。 |
| `Extinction`・`CSOnly` | n | CS だけ、US 無し。`cs_on` 0。 |
| `USOnly` | n | CS 無し、US だけ。`us_on` 0。 |
| `Compound` | n | `cs_list` の CS を全部同時に、US 無し。`cs_on` 0。Tier A は `mode=Simultaneous` だけ。 |
| `Serial` | n | CS を全部、US 無し。`cs_on` 0（最初の CS）、`order` = `cs_list`、`gap` は CS の終わり → 次の CS の始め = `isi`。 |
| `Contingency(p, q)` | 2n | CS の区間 n 個と CS の無い区間 n 個を、CS から交互に。区間ごとにこの順で 1 回ずつ `u` を引き、CS の区間では `u < p`、CS の無い区間では `u < q` のとき US を出す。CS と US のラベルはプログラムの設定から取る。 |
| `TrulyRandom(cs, us, p)` | 2n | 名指したラベルでの `Contingency(p, p)`。`p` が無いときは `unresolved` で失敗する。 |
| `ExplicitlyUnpaired` | 2n | CS だけの試行、続いて US だけの区間、を n 回。`min_separation` があれば、US の区間ごとに `gap` `{"from": "any_cs", "to": "us_on", "at_least": true}` を付ける。 |
| `Differential` | 2n | US 付きの CS+、続いて US 無しの CS−、を n 回。短い形は US をプログラムの設定から取り、無ければ `unresolved` で失敗する。 |
| `ITI(distribution, mean)` | n | 刺激の無い項目で `iti_s` を持つ。`fixed` = mean、`uniform` = 2·mean·u、`exponential` = −mean·ln(1 − u)。 |

### 乱数の引き方

展開のたびに整数の `seed` から生成器を作り直し、`u` はどれも CPython の `random.Random(seed).random()` の次の値である。ほかの実装が再現できる形で書くと:

- MT19937 を、`|seed|` の 32 ビットの語（下位の語から）で `init_by_array` を使って初期化する（`-7` と `7` は同じ列。`0` は語 `0` 1 つ）。
- `u = ((a >> 5) * 2^26 + (b >> 6)) / 2^53`。`a`・`b` は続けて出る 32 ビットの出力。
- 契約に入るのは整数の seed だけ。

ほかの実装はこの生成器で同じ列を出し、自前の生成器は使わない。確率が 0 や 1 の区間でも 1 回引くので、列の位置は試行の番号だけで決まる。

### プログラムに任される値

式は、実行に要る値を全部は持たない。プログラムが展開の設定で渡す（セッションでは `@cs`・`@us`・`@iti` の注釈か段階から来る）:

| 設定 | 使う要素 |
|---|---|
| `trials` | 自分の `trials` を持たない全要素 |
| `seed` | `Contingency`・`TrulyRandom`・`ITI(uniform/exponential)` |
| `labels.cs`・`labels.us` | `Contingency`。`Differential` の短い形は `labels.us` |

要素が要る値が無いときは誤り（`{"error": "unresolved", "missing": "<名前>"}`。`<名前>` は上の設定の名前か、`p` のような要素自身の引数）にし、実装が既定値を選んで埋めることはしない。

## fixture

`conformance/respondent/<name>.json` に、1 ファイル 1 件で置く:

```json
{
  "name": "pair_forward_trace",
  "element": "Pair.ForwardTrace",
  "description": "R2. ... (Ellison, 1964).",
  "spec": {"type": "PairForwardTrace", "cs": "tone", "us": "food",
           "trace_interval": {"value": 5.0, "unit": "s"}},
  "options": {"trials": 2},
  "expected": {"trials": [ ... ]}
}
```

`spec` は contingency-dsl のレスポンデントの schema の Tier A の AST の節点、`expected` は `{"trials": [...]}` か上の `unresolved` の誤りである。Tier A の要素はどれも 1 件以上ある。ファイルは `scripts/generate_respondent_conformance.py` が書き、これがこのページの参照の読み方である。`python scripts/generate_respondent_conformance.py --check` は、書くはずの中身とファイルが違うと失敗する。スケジュールの照合器（`contingency-conformance`）はこのディレクトリを読まない（スケジュールの fixture ではないため）。

## README の範囲の文の案

README は今「強化子をいつ提示するかを決める Python のライブラリです」で始まる。Tier A の展開を公開した後の案:

> contingency-py は、行動実験の随伴性を扱う Python のライブラリである。オペラントのスケジュール（`FR 5`・`VI 60-s` など）では、反応が起きるたびに強化子がいつ出るかを決める。DSL の Tier A のレスポンデント（パヴロフ型）の手続き（`Pair.ForwardDelay(tone, shock, isi=10-s, cs_duration=15-s)` など）では、試行の列 — どの CS を、どの US を、いつ — を返す。

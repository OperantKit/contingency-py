# contingency-py

:gb: [English README](README.md)

contingency-py は、強化子をいつ提示するかを決める Python のライブラリです。`FR 5`・`VI 60-s`・`Conc(VI 30-s, VI 60-s, COD=2-s)` のような強化スケジュールと、起きた反応を順に渡すと、ステップごとに強化子を提示するかどうかを返します。実験箱の制御プログラム、シミュレーション、授業でのデモ、別の実装を検査するときの基準として使えます。[OperantKit](https://github.com/OperantKit) の Python 版スケジュールエンジンで、セッション全体の実行（開始・終了の条件、記録）はこの上に載る [session-runner](https://github.com/OperantKit/session-runner) が受け持ちます。

- **変動スケジュールの値は Fleshler–Hoffman (1962) の数列で作ります。** 値の個数と乱数の種を指定できるので、セッションを厳密に再現できます。
- **並立スケジュールは切替遅延（COD）を扱います。** COD は、オペランダム（レバーやキー）を切り替えた直後に反応を強化しない時間のことです。設定済みの強化子は、そのオペランダムへの次の反応まで保持します。
- **39 個の適合性 fixture が、各スケジュールの結果を既知の反応系列に対して固定しています。** Rust 版エンジン [contingency-rs](https://github.com/OperantKit/contingency-rs) も同じ fixture を再生します。自作の実装をこれに照らして検算することもできます。

[スケジュールの用語](docs/ja/schedule-terms.md) では、これらの点（COD、Fleshler–Hoffman の数列、DRL と反応間時間、リセット型 DRO、VI の間隔がいつ始まるか）を数文ずつ説明しています。AI のコーディングエージェント向けの案内（AI アシスタントが書いたスケジュールのコードとの比較を含む）は [AGENTS.md](AGENTS.md) にあります。

## インストール

Python 3.11 以上が必要です。contingency-py はまだ PyPI に登録されていません。本体と、式を読む [contingency-dsl-py](https://github.com/OperantKit/contingency-dsl-py) を、依存の解決をせずに GitHub から入れます。

```bash
G=git+https://github.com/OperantKit
pip install --no-deps $G/contingency-dsl-py $G/contingency-py
pip install gymnasium numpy     # 任意: Gymnasium 環境を使うときだけ
```

`pip check` はこれらの開発版のあいだの版の不一致を報告しますが、以下の例には影響しません。

## 例

スケジュールを [contingency-dsl](https://github.com/OperantKit/contingency-dsl) の式で書き、`from_dsl` で組み立て、反応を渡します。

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
print(reinforcers)  # {'c0': 16, 'c1': 9} — 種を固定しているので毎回同じ
```

並立スケジュールの成分は、書いた順に `c0`、`c1`、… と名付けられます。

## 含まれるもの

| 分野 | 内容 |
|---|---|
| 単一スケジュール | FR、VR、RR、FI、VI、RI、FT、VT、RT、CRF、EXT、リミテッドホールド |
| 複合スケジュール | 並立（COD、固定比率の切替要件、方向別の COD つき）、多元、混合、連鎖、連接、選択、連言、挿入、重畳、二次スケジュール |
| 分化強化と累進 | DRO（リセット型・瞬時型）、DRL、DRH、パーセンタイル、累進比率（等差・等比・Richardson–Roberts）、調整、連動 |
| その他の手続き | タイムアウト、反応コスト、Sidman 型回避、弁別回避、逃避、見本合わせ、Go/No-Go |
| 値の生成 | Fleshler–Hoffman（Fleshler & Hoffman, 1962）と Hantula (1991) の変法 |
| スケジュールの組み立て | `contingency.bridge.from_dsl`（contingency-dsl の式から）、`contingency.builder.ScheduleBuilder`（Python の呼び出しから） |
| ハードウェア抽象化層（`contingency.hw`） | `Apparatus` インタフェースと、仮想の装置、シリアル接続（`pyserial` が必要）、ハードウェア・イン・ザ・ループ（HIL）試験装置向けの TCP 接続 |
| 適合性検査 | `conformance/` の 39 個の JSON fixture と `contingency-conformance` コマンド |
| Gymnasium | `contingency.envs.ScheduleEnv` と、登録済みの 3 つの環境（任意） |

```python
from contingency import Schedule, Observation, Reinforcer
from contingency.schedules import FR, VR, FI, VI, FT, VT, CRF, EXT, DRO
from contingency.schedules import Concurrent, Multiple, Chained, Tandem, Alternative
from contingency.builder import ScheduleBuilder
from contingency.bridge import from_dsl
```

### 自作の実装を fixture で検算する

`contingency-conformance` コマンドは、どの言語で書いたスケジュールの実装でも、39 個の fixture に照らして検算します。自作のプログラムは fixture を JSON で受け取り、どのステップが強化されたかを返します。

```bash
contingency-conformance list                    # fixture・種別・スケジュール型・step 数
contingency-conformance run --only 'atomic/fr_*' -- python my_adapter.py
```

JSON の形式、動くアダプタの全文、乱数の種を持つ fixture の判定のしかたは [自作の実装を fixture で検算する](docs/ja/conformance-checker.md) にあります。

### Gymnasium 環境

`contingency.envs.ScheduleEnv` は、任意の `Schedule` を強化学習エージェント向けのフリーオペラントのセッションとして動かします。1 ステップで時計が `dt`（既定 1）進み、行動 `0` は反応しないこと、行動 `i` は `i` 番目のオペランダムへの 1 反応です。`step` が返す `reward` の値は強化子の大きさで、反応したステップでは `response_cost`（既定 0）を引きます。観測は被験体が知りうるものだけです（直前の強化からの経過時間と反応数、直前のステップが強化されたか、直前に反応したオペランダム）。import すると 4 つの ID が登録されます: `contingency/FR-v0`（`n`）、`contingency/FI-v0`（`interval`）、`contingency/ConcVIVI-v0`（`mean_left`・`mean_right`・`cod`。オペランダムは `left` と `right`）、`contingency/ConcPhases-v0`（`phases`・`cod`）。

`ConcPhases-v0` は、リサージェンス（resurgence）などの手続きのために、1 つのセッションで相を順に切り替えます。各相は `(left, right, steps)` で、各オペランダムの VI の平均（秒。`None` は消去）と、その相が続くステップ数を表します。既定は、`left` に VI 15 s を 18000 ステップ、次に `left` を消去して `right` に VI 15 s を 12000 ステップ、最後に両方を消去して 3000 ステップです。相は観測に入らず、`info["phase"]` と `info["phase_step"]` に出ます。最後の相が終わるとエピソードは終了（terminated）します。

```python
import gymnasium as gym
import contingency.envs  # registers contingency/FR-v0, FI-v0, ConcVIVI-v0, ConcPhases-v0

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

`contingency.envs.tabular` は ε-greedy の表形式 Q 学習（`train_q`、`run_greedy`）を持ちます。`tests/test_envs_concurrent.py` では、既定の Conc VI 30 VI 60（COD 2）で 100 セッション学習した表が、学習に使っていない 10 セッションで 600 時間単位あたり平均 23.2 回の強化を得ました。一様な無作為方策は 20.7 回、2 つのオペランダムから無作為に選ぶ方策（必ず反応する）は 21.1 回、学習前の同じ表は 21.5 回でした。

## 制限

- まだ PyPI に登録されておらず、API は変わることがあります（開発段階: alpha）。
- これはライブラリであって、セッションを動かすプログラムではありません。自前の時計を持たず、呼び出す側のプログラムが現在時刻を付けて `step` を呼びます。時間の精度はそのプログラムとハードウェアで決まります。
- 8 本の fixture（種付きの VR・VI・VT・RR・RI・RT・見本合わせ・Go/No-Go）は Python の `random.Random(seed)` に合わせて固定されています。別の乱数生成器を使う実装は、これらと近似的にしか一致しません。
- 仮想被験体と Gymnasium の被験体はモデルです。書いたスケジュールどおりにプログラムが強化子を提示することは示しますが、動物がどう振る舞うかは示しません。

## MED-PC の記録で再生した結果

`tests/test_apparatus_replay.py` は、公開されたセッションで MED-PC の装置が記録した反応を `DRL` と `ProgressiveRatio` にそのまま渡し、どの反応が強化されるかを比べます。データ全体 — DRL 10 s の 928 セッション・強化 30,578 件（Sosa et al., 2026）と、比率が 1, 4, 7, … と増える累進比率の 273 セッション・強化 3,979 件（McCutcheon, 2022）— で、装置のプログラムと同じく IRT（反応間時間）の計時を DRL の開始時点から始め、強化後 4 s（累進比率では 5.1 s）の反応を除くと、記録された強化のうち DRL で 99.30%、累進比率で 99.75% を、このパッケージも同じ反応で強化しました。食い違いはすべて計時の分解能によるもので、DRL では 10 s の境界から ±70 ms 以内の反応 192 件とそれに続く反応 93 件、累進比率では 5.1 s のタイムアウトの境界にあった反応とそれに続く反応の計 20 件です。テストはこのうち 4 セッションを fixture として持っています。

## Swift 原典との違い

本パッケージは Swift のライブラリ [OperantKit](https://github.com/YutoMizutani/OperantKit) の移植です。`tests/test_legacy_swift_parity.py` は手元の Swift 原典をビルドし、小さなドライバ（`tools/legacy-swift/`）で上の fixture に通します。Swift のツールチェーンが必要で、原典の場所を渡さない限り skip されます。

```bash
OPERANTKIT_LEGACY_DIR=/path/to/OperantKit .venv/bin/pytest -q tests/test_legacy_swift_parity.py
```

Swift 原典にあるのは FR・CRF・EXT・FI・FT・VI・VR・VT・Alternative・Concurrent（切替遅延なし）です。乱数生成器に種を与えられないので、VI・VR・VT は本パッケージが引く値の列をテストが注入します。FR・CRF・EXT・FI・FT は fixture と step ごとに一致します。既知の違い:

- **VI・VR・VT**: Swift 原典は値の列の最初の値を繰り返します。`Parameter.nextOrder()` が `mutating` でないため、列の位置が進みません。この 1 関数だけを `mutating` にすると（`tools/legacy-swift/nextorder-control.patch`、一時コピーにだけ当てます）3 本とも完全に一致します。
- **Alternative(FR 5, FT 10 s)**: 5 秒に FR が強化した後、Swift は次の FT の間隔を強化の時点（5 秒）から測って 15 秒に強化します。本パッケージは FT を次の step（6 秒）から測り直します。
- **Hantula (1991) の生成器**: 最小値が 0 に丸まるとき（たとえばテストした `v = 10, n = 12`）、Swift の `hantula1991` は 0 を残します。`generate_intervals_hantula1991` はそれを 1 に置き換えるので、合計が 1 多くなります。
- **Fleshler-Hoffman の生成器**（VR 5・`n = 12` と、VI の `(10, 12)`・`(60, 20)`・`(30000, 12)` で確認）: 比率の列は多重集合として一致します。Swift は丸めの過不足を無作為な位置に足すので、他のパラメータでは多重集合が変わることがあります。Swift の間隔の生成器は整数を返します。Fleshler-Hoffman の数列を丸め、丸めの余りを無作為な位置の 1 要素に足したものです。`generate_intervals` は小数のまま返します。

これらは記録であって修正ではありません。どちらの実装も相手に合わせて変えていません。

## 開発

```bash
git clone https://github.com/OperantKit/contingency-py && cd contingency-py
python -m venv .venv
.venv/bin/python -m pip install --no-deps git+https://github.com/OperantKit/contingency-dsl-py -e .
.venv/bin/python -m pip install pytest pytest-cov hypothesis gymnasium numpy
.venv/bin/pytest
```

変更の送り方は [CONTRIBUTING.md](CONTRIBUTING.md) を、行動規範は [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) を参照してください。

## サポート

不具合の報告・機能の要望・使い方の質問は [issue tracker](https://github.com/OperantKit/contingency-py/issues) に書いてください。不具合の報告には、スケジュールの式（またはスケジュールを作るコード）、再現する反応の列、期待した結果と実際の結果、Python・contingency-py・contingency-dsl-py の版を添えてください。開発者 1 名が手の空いたときに対応しています。返信の期日は約束できませんが、issue はすべて読みます。

## 参考文献

- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Hantula, D. A. (1991). A simple BASIC program to generate values for variable-interval schedules of reinforcement. *Journal of Applied Behavior Analysis*, 24(4), 799-801. https://doi.org/10.1901/jaba.1991.24-799
- McCutcheon, J. (2022). *Progressive ratio in protein-restricted vs. non-restricted rats* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5409201
- Sosa, R., Mata-Luévanos, J., Tapia-de-Jesús, A., Saavedra, P., & Buenrostro-Jáuregui, M. (2026). *Longitudinal association between different indices of behavioral inhibition* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19164959

## ライセンス

MIT です。[LICENSE](LICENSE) を参照してください。

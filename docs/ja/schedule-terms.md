# スケジュールの用語: 切替遅延、Fleshler-Hoffman、DRL、リセット型 DRO、VI

:gb: [English](../en/schedule-terms.md)

強化スケジュールを手で書くときに出てくる問いへの短い答えです。各節の最後に、contingency-py がそのまま実行する contingency-dsl の式を書きます。ページ末尾の例は、ここに書いた COD・DRL・DRO・VI の挙動を実行し、どれかが変わると失敗します。

## Changeover delay (COD) in concurrent variable-interval schedules（並立 VI の切替遅延）

並立スケジュールでは、2 つ以上のスケジュールが別々のオペランダムで同時に進み、被験体はその間を切り替えられます。切替遅延（changeover delay, COD）は、切り替えのたびに一定時間だけ強化を止め、切り替えた直後の反応が切り替え先のスケジュールで強化されないようにします。COD の間に切り替え先で強化子が利用可能になっても取り消されず、保持されて、COD が明けた後の最初の反応で提示されます。COD は切り替えそのものの直後に強化が続かないようにするもので、マッチング法則の元になった研究の手続きに含まれています（Herrnstein, 1961; Catania, 1963）。

operantkit での書き方: `Conc(VI30s, VI60s, COD=2s)`（固定比率の切替要件は `FRCO=n` を足す）。

## Fleshler-Hoffman progression for variable-interval schedules（VI の Fleshler-Hoffman 数列）

変動間隔（VI）スケジュールには、平均が名目値（VI 60 秒なら 60 秒）に等しい間隔の一覧が要ります。Fleshler and Hoffman（1962）は、直前の強化子からの経過時間によらず強化の確率がおおむね一定になるような一覧を作る数列を示しました。間隔を一様分布や指数分布から 1 つずつ引くと、間隔の分布が変わり、論文の手続きを再現しません。同じ数列は変動比率（VR）の要求数にも使い、反応数として整数に丸めます。

operantkit での書き方: `VI60s` だけで間隔は Fleshler-Hoffman の数列から作られる（既定は 12 個）。前に `@algorithm("fleshler-hoffman", n=12, seed=42)` を置くと、個数と seed が固定され、セッションを再現できる。同じ値の一覧は `contingency.helpers.fleshler_hoffman.generate_intervals(60, n=12, seed=42)` が返す。

## DRL: each response resets the interresponse-time timer（DRL では反応ごとに反応間時間の計時が戻る）

低反応率分化強化（DRL）は、直前の反応からの時間、つまり反応間時間（IRT）がスケジュール値以上のときだけ反応を強化します。直前の反応が無いセッション最初の反応は強化されます。強化されたかどうかに関わらず反応のたびに計時がやり直しになるので、早すぎる反応は次に強化されうる時点を先へ延ばします。反応せずに待つだけでは強化子は出ず、反応が要ります（Kramer & Rilling, 1970）。

operantkit での書き方: `DRL20s`。

## Resetting DRO: every response resets the interval（リセット型 DRO では反応ごとに間隔が戻る）

他行動分化強化（DRO）は、標的反応が間隔の全体を通して起きなかったときに強化子を提示します。リセット型では、標的反応のたびに間隔がやり直しになり、強化子の後は次の間隔がすぐに始まります。強化子は反応ではなく時間の経過で提示されるので、何も起きていないときにも計時を確かめる必要があります。DRO の手続きは、間隔が固定か変動か、間隔の全体を見るか終わりの瞬間だけを見るかでも分かれます（Lindberg et al., 1999）。

operantkit での書き方: `DRO10s`（リセット型・間隔全体・固定）。

## Variable-interval schedule: the reinforcer is held, and the next interval starts after reinforcement（VI では強化子が保持され、次の間隔は強化の後から始まる）

VI スケジュールでは、現在の間隔が経過した後の最初の反応が強化されます。それより前の反応には予定された結果が無く、間隔をやり直させもしません。間隔が経過すると、どれだけ時間がかかっても、強化子は次の反応まで利用可能なまま残ります。次の間隔は、その強化子の提示から数えます（Ferster & Skinner, 1957; Catania & Reynolds, 1968）。

operantkit での書き方: `VI60s`。

## 実行できる確認

```python
from contingency_dsl import parse
from contingency_dsl.annotations import TemporalExtension
from contingency_dsl.extensions import ExtensionRegistry
from contingency import ResponseEvent, from_dsl

registry = ExtensionRegistry.build(TemporalExtension())


def compile_(source):
    return from_dsl(parse(source, extensions=registry))


def respond(schedule, now, operandum="main"):
    return schedule.step(now, ResponseEvent(time=now, operandum=operandum))


# 切替遅延: 4 秒の切り替えで c1 に 2 秒の COD が始まる。c1 の FI 3 秒は
# 既に経過しているので、強化子は保持され 6.5 秒に提示される。
conc = compile_("Conc(FI3s, FI3s, COD=2s)")
assert not respond(conc, 1.0, "c0").reinforced
assert respond(conc, 4.0, "c1").meta["cod_suppressed"]
assert respond(conc, 5.0, "c1").meta["cod_suppressed"]
assert respond(conc, 6.5, "c1").reinforced

# DRL 20 秒: 10 秒と 50 秒の反応は早すぎ、計時をやり直させる。
drl = compile_("DRL20s")
times = [0.0, 10.0, 35.0, 50.0, 75.0]
assert [t for t in times if respond(drl, t).reinforced] == [0.0, 35.0, 75.0]

# リセット型 DRO 10 秒: 4 秒の反応で最初の強化子が 14 秒に移る。
dro = compile_("DRO10s")
delivered = []
for t in range(31):
    event = ResponseEvent(time=float(t), operandum="main") if t == 4 else None
    if dro.step(float(t), event).reinforced:
        delivered.append(t)
assert delivered == [14, 24]

# VI 60 秒: 強化子は最初の反応が 500 秒でも 700 秒でもそれまで待ち、次の間隔は
# その提示から数えるので、2 回目の強化子も同じ 200 秒だけずれる。
def second_reinforcer(first_response):
    vi = compile_('@algorithm("fleshler-hoffman", n=12, seed=42)\nVI60s')
    assert respond(vi, float(first_response)).reinforced
    t = first_response + 1
    while not respond(vi, float(t)).reinforced:
        t += 1
    return t


assert second_reinforcer(700) - second_reinforcer(500) == 200
```

## 参考文献

- Catania, A. C. (1963). Concurrent performances: Reinforcement interaction and response independence. *Journal of the Experimental Analysis of Behavior*, 6(2), 253-263. https://doi.org/10.1901/jeab.1963.6-253
- Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis of the responding maintained by interval schedules of reinforcement. *Journal of the Experimental Analysis of Behavior*, 11(3, Pt. 2), 327-383. https://doi.org/10.1901/jeab.1968.11-s327
- Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*. Appleton-Century-Crofts.
- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating variable-interval schedules. *Journal of the Experimental Analysis of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Herrnstein, R. J. (1961). Relative and absolute strength of response as a function of frequency of reinforcement. *Journal of the Experimental Analysis of Behavior*, 4(3), 267-272. https://doi.org/10.1901/jeab.1961.4-267
- Kramer, T. J., & Rilling, M. (1970). Differential reinforcement of low rates: A selective critique. *Psychological Bulletin*, 74(4), 225-254. https://doi.org/10.1037/h0029813
- Lindberg, J. S., Iwata, B. A., Kahng, S., & DeLeon, I. G. (1999). DRO contingencies: An analysis of variable-momentary schedules. *Journal of Applied Behavior Analysis*, 32(2), 123-136. https://doi.org/10.1901/jaba.1999.32-123

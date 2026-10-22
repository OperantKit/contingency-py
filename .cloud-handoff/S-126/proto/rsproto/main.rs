//! S-126 prototype: replay scratch fixtures for the 15 uncovered classes
//! against contingency-rs with the proposed `build_schedule` branches.
use std::path::Path;
use serde::Deserialize;
use contingency::schedules::*;
use contingency::{Outcome, Reinforcer, ResponseEvent, Schedule};

#[derive(Deserialize)]
struct Fixture { schedule: Spec, steps: Vec<Step> }
#[derive(Deserialize, Clone)]
struct Spec { #[serde(rename = "type")] ty: String, #[serde(default)] params: serde_json::Value }
#[derive(Deserialize)]
struct Step { now: f64, event: Option<ResponseEvent>, expect: Expect }
#[derive(Deserialize)]
struct Expect { reinforced: bool, #[serde(default)] reinforcer: Option<Reinforcer>, #[serde(default)] meta: serde_json::Map<String, serde_json::Value> }

fn f(p: &serde_json::Value, k: &str) -> f64 { p.get(k).and_then(|v| v.as_f64()).unwrap_or_else(|| panic!("param {k}")) }
fn of(p: &serde_json::Value, k: &str) -> Option<f64> { p.get(k).and_then(|v| v.as_f64()) }
fn u(p: &serde_json::Value, k: &str) -> u64 { p.get(k).and_then(|v| v.as_u64()).unwrap_or_else(|| panic!("param {k}")) }
fn s<'a>(p: &'a serde_json::Value, k: &str, d: &'a str) -> &'a str { p.get(k).and_then(|v| v.as_str()).unwrap_or(d) }
fn sub(p: &serde_json::Value, k: &str) -> Box<dyn Schedule> { build(&serde_json::from_value(p[k].clone()).expect("spec")) }
fn osub(p: &serde_json::Value, k: &str) -> Option<Box<dyn Schedule>> { p.get(k).filter(|v| !v.is_null()).map(|v| build(&serde_json::from_value(v.clone()).expect("spec"))) }

// Existing branches (subset needed by the prototypes) + the 15 proposed ones.
fn build(spec: &Spec) -> Box<dyn Schedule> {
    let p = &spec.params;
    match spec.ty.as_str() {
        "FR" => Box::new(FR::new(u(p, "n")).unwrap()),
        "CRF" => Box::new(crf()),
        "EXT" => Box::new(EXT::new()),
        "FI" => Box::new(FI::new(f(p, "interval")).unwrap()),
        "FT" => Box::new(FT::new(f(p, "interval")).unwrap()),
        // ---- proposed ----
        "AdjustingSchedule" => {
            let target = match s(p, "target", "ratio") { "ratio" => AdjustingTarget::Ratio, "interval" => AdjustingTarget::Interval, o => panic!("target {o}") };
            Box::new(AdjustingSchedule::new(target, f(p, "start"), f(p, "step"), of(p, "minimum"), of(p, "maximum")).unwrap())
        }
        "Sidman" => Box::new(Sidman::new(f(p, "shock_shock_interval"), f(p, "response_shock_interval"), of(p, "shock_magnitude").unwrap_or(1.0)).unwrap()),
        "DiscriminatedAvoidance" => Box::new(DiscriminatedAvoidance::new(f(p, "warning_duration"), f(p, "iti"), of(p, "shock_magnitude").unwrap_or(1.0)).unwrap()),
        "Escape" => Box::new(Escape::new(f(p, "trial_duration"), f(p, "iti"), of(p, "shock_magnitude").unwrap_or(1.0)).unwrap()),
        "Conjunctive" => Box::new(Conjunctive::new(sub(p, "first"), sub(p, "second")).unwrap()),
        "InterlockingSchedule" => Box::new(InterlockingSchedule::new(u(p, "initial_ratio"), f(p, "decay_time")).unwrap()),
        "Interpolate" => Box::new(Interpolate::new(sub(p, "base"), sub(p, "probe"), f(p, "interval"), f(p, "probe_duration"), of(p, "first_probe_at")).unwrap()),
        "Mixed" => Box::new(Mixed::new(p["components"].as_array().unwrap().iter().map(|v| build(&serde_json::from_value(v.clone()).unwrap())).collect()).unwrap()),
        "Overlay" => Box::new(Overlay::new(sub(p, "base"), sub(p, "punisher")).unwrap()),
        "Percentile" => {
            let dir = match s(p, "direction", "above") { "above" => PercentileDirection::Above, "below" => PercentileDirection::Below, o => panic!("dir {o}") };
            Box::new(Percentile::new(PercentileTarget::Irt, u(p, "rank") as u8, u(p, "window") as usize, dir).unwrap())
        }
        "ResponseCost" => Box::new(ResponseCost::new(sub(p, "inner"), f(p, "amount"), s(p, "unit", "token").to_string(), of(p, "initial_balance")).unwrap()),
        "SecondOrder" => Box::new(SecondOrder::new(sub(p, "overall"), sub(p, "unit"))),
        "Timeout" => Box::new(Timeout::new(sub(p, "inner"), f(p, "duration"), p.get("reset_on_response").and_then(|v| v.as_bool()).unwrap_or(false)).unwrap()),
        "MatchingToSample" => Box::new(MatchingToSample::new(u(p, "n_comparisons") as u32, of(p, "sample_duration").unwrap_or(0.0), of(p, "choice_timeout").unwrap_or(5.0), osub(p, "consequence"), osub(p, "incorrect"), of(p, "iti").unwrap_or(1.0), p.get("seed").and_then(|v| v.as_u64())).unwrap()),
        "GoNoGo" => Box::new(GoNoGo::new(of(p, "go_probability").unwrap_or(0.5), of(p, "response_window").unwrap_or(5.0), of(p, "iti").unwrap_or(5.0), osub(p, "correct_go_schedule"), osub(p, "correct_nogo_schedule"), osub(p, "false_alarm"), p.get("seed").and_then(|v| v.as_u64())).unwrap()),
        o => panic!("unsupported {o}"),
    }
}

fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() { let p = e.unwrap().path(); if p.is_dir() { walk(&p, out) } else if p.extension().map(|x| x == "json").unwrap_or(false) { out.push(p) } }
}

fn main() {
    let root = std::env::args().nth(1).expect("fixture dir");
    let mut files = Vec::new(); walk(Path::new(&root), &mut files); files.sort();
    println!("{:44} {:>5} {:>5} {:>6} {:>6} {}", "fixture", "exp", "rust", "strict", "meta", "first mismatch");
    for path in files {
        let rel = path.strip_prefix(&root).unwrap().to_string_lossy().to_string();
        let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut sch = build(&fx.schedule);
        let (mut exp_n, mut rs_n) = (0, 0);
        let mut strict = true; let mut meta_ok = true; let mut first = String::new();
        for (i, st) in fx.steps.iter().enumerate() {
            let o: Outcome = match sch.step(st.now, st.event.as_ref()) { Ok(o) => o, Err(e) => { strict = false; if first.is_empty() { first = format!("step {i}: error {e}"); } break; } };
            if st.expect.reinforced { exp_n += 1 } if o.reinforced { rs_n += 1 }
            let mut ok = o.reinforced == st.expect.reinforced;
            if ok { if let (Some(a), Some(b)) = (&o.reinforcer, &st.expect.reinforcer) { ok = (a.time - b.time).abs() <= 1e-9 && (a.magnitude - b.magnitude).abs() <= 1e-9 && a.label == b.label; } else if o.reinforcer.is_some() != st.expect.reinforcer.is_some() { ok = false } }
            if !ok { strict = false; if first.is_empty() { first = format!("step {i} (now={}): reinforced exp={} got={} rf exp={:?} got={:?}", st.now, st.expect.reinforced, o.reinforced, st.expect.reinforcer, o.reinforcer); } }
            let got_meta = serde_json::to_value(&o).unwrap().get("meta").cloned().unwrap_or(serde_json::Value::Object(Default::default()));
            for (k, v) in &st.expect.meta {
                let g = got_meta.get(k);
                let same = match (g, v) { (Some(serde_json::Value::Number(a)), serde_json::Value::Number(b)) => (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1e-9, (Some(a), b) => a == b, (None, _) => false };
                if !same { meta_ok = false; if first.is_empty() { first = format!("step {i}: meta[{k}] exp={v} got={:?}", g); } }
            }
        }
        println!("{:44} {:>5} {:>5} {:>6} {:>6} {}", rel, exp_n, rs_n, strict, meta_ok, first);
    }
}

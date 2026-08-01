"""Seed-aware analysis for the axis-4 grid. Prints a table and emits JSON for the report."""
import json, sys, statistics as st

GRID = sys.argv[1] if len(sys.argv) > 1 else 'results_gpu_sweep/axis4_eagle_code/grid.json'
ARMS = ['static-cap-hi','ratio-cap','predictive-cap','predictive-kv','predictive-full','predictive-shed']
METRICS = ['slo_attainment','slo_attainment_offered','ttft_p50','ttft_p99','tpot_p50',
           'e2e_p95','goodput_tok_s','mean_gamma','mean_kv_frac','shed_frac','n_finished',
           'mean_accept_rate','accept_rate_cv']

d = json.load(open(GRID))
ok = [v for v in d.values() if 'error' not in v]
rates = sorted({v['rate'] for v in ok})

def agg(rows, m):
    vals = [r[m] for r in rows if m in r and r[m] == r[m]]
    if not vals: return None
    return {'mean': sum(vals)/len(vals),
            'sd': st.stdev(vals) if len(vals) > 1 else 0.0,
            'n': len(vals), 'vals': vals}

out = {}
for arm in ARMS:
    for rate in rates:
        rows = [v for v in ok if v.get('arm') == arm and v.get('rate') == rate]
        if not rows: continue
        out[f'{arm}@{rate}'] = {'arm': arm, 'rate': rate,
                                **{m: agg(rows, m) for m in METRICS}}

# significance vs baseline: |delta| > 2 * pooled SD
def cmp_to(base_arm, arm, rate, metric='slo_attainment'):
    a = out.get(f'{arm}@{rate}'); b = out.get(f'{base_arm}@{rate}')
    if not a or not b or not a.get(metric) or not b.get(metric): return None
    da, db = a[metric], b[metric]
    pooled = ((da['sd']**2 + db['sd']**2) / 2) ** 0.5
    delta = da['mean'] - db['mean']
    return {'delta': delta, 'pooled_sd': pooled,
            'significant': abs(delta) > 2*pooled if pooled > 0 else False,
            'ratio': (abs(delta)/pooled) if pooled > 0 else float('inf')}

print(f"{'arm':18s}{'rate':>5s}{'n':>3s}{'slo':>16s}{'slo_offer':>16s}{'ttft_p99':>16s}{'tpot_p50':>10s}{'gamma':>7s}{'shed':>7s}{'e2e_p95':>10s}")
for arm in ARMS:
    for rate in rates:
        k = f'{arm}@{rate}'
        if k not in out: continue
        r = out[k]
        def s(m, prec=3):
            v = r.get(m)
            return 'n/a'.rjust(10) if not v else f"{v['mean']:.{prec}f}±{v['sd']:.{prec}f}"
        print(f"{arm:18s}{rate:5.0f}{r['slo_attainment']['n']:3d}"
              f"{s('slo_attainment'):>16s}{s('slo_attainment_offered'):>16s}"
              f"{s('ttft_p99',2):>16s}{r['tpot_p50']['mean']:10.4f}"
              f"{r['mean_gamma']['mean']:7.2f}{(r['shed_frac']['mean'] if r.get('shed_frac') else 0):7.3f}"
              f"{r['e2e_p95']['mean']:10.2f}")

print("\n=== vs static-cap-hi (2-SD test on slo_attainment) ===")
sig = {}
for arm in ARMS[1:]:
    for rate in rates:
        c = cmp_to('static-cap-hi', arm, rate)
        if not c: continue
        sig[f'{arm}@{rate}'] = c
        verdict = 'SIGNIFICANT' if c['significant'] else f"tie ({c['ratio']:.1f} SD)"
        print(f"  {arm:18s} rate={rate:.0f}: delta={c['delta']:+.4f} pooled_sd={c['pooled_sd']:.4f}  {verdict}")

print("\n=== TTFT p99 vs static-cap-hi ===")
for arm in ARMS[1:]:
    for rate in rates:
        c = cmp_to('static-cap-hi', arm, rate, 'ttft_p99')
        if not c: continue
        verdict = 'SIGNIFICANT' if c['significant'] else f"tie ({c['ratio']:.1f} SD)"
        print(f"  {arm:18s} rate={rate:.0f}: delta={c['delta']:+8.2f}s pooled_sd={c['pooled_sd']:6.2f}  {verdict}")

json.dump({'agg': out, 'sig': sig}, open('axis4_analysis.json','w'), indent=2)
print(f"\n{len(ok)}/{len(d)} cells ok -> axis4_analysis.json")

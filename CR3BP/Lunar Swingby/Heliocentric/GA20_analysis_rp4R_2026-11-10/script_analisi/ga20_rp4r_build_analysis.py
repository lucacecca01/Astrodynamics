"""Package the rp=4R catalogue census and bounded dynamical analysis."""
import os,json,csv,shutil,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/ga20_rp4r_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pykep as pk

BASE=Path('/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Heliocentric')
TMP=Path('/tmp');OUT=BASE/'GA20_analysis_rp4R_2026-11-10'
OUT.mkdir(exist_ok=True)
DETAIL=OUT/'dettagli';DETAIL.mkdir(exist_ok=True)
def read(name):return json.loads((TMP/name).read_text())
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
coverage=read('ga20_all_rp4r_coverage.json')
itokawa=read('ga20_rp4r_itokawa_report.json')
ccshort=read('ga20_rp4r_cc_three_turns_report.json')
fgshort=read('ga20_rp4r_fg3_return_coasts.json')
physical=json.loads((BASE/'asteroids_sbdb_physical.json').read_text())['asteroids']
rows=[];dynamics={};calls=branches=0
for tag,index,spkid in [('itokawa',40,'20025143'),('cc',66,'20136617'),('fg3',125,'20175706')]:
    out=read(f'ga20_rp4r_{tag}_outbound.json')
    calls+=out['lambert_problem_calls'];branches+=out['lambert_branches_checked']
    bc={str(b):sum(m['outbound_branch']==b for m in out['outbound_matches']) for b in range(5)}
    p=physical[spkid];R=p['diameter_km']/2
    candidate=None;audit=None
    if tag!='itokawa':
        candidate=read(f'ga20_rp4r_{tag}_extended_candidate.json')
        audit=read('ga20_rp4r_cc_extended_soi_validation.json' if tag=='cc' else 'ga20_rp4r_fg3_surface_soi_validation.json')
    row=dict(catalogue_index=index,spkid=spkid,name=out['body'],mass_kg=p['mass_kg'],mass_scope=p.get('mass_scope','asteroid'),radius_km=R,min_periapsis_km=4*R,min_altitude_km=3*R,lambert_problem_calls=out['lambert_problem_calls'],lambert_branches_checked=out['lambert_branches_checked'],solver_errors=out['solver_errors'],accepted_outbound_count=len(out['outbound_matches']),outbound_branch_counts=bc,current_complete_candidates_found=0,current_status='no_acceptable_outbound_on_daily_grid_plus_local_refinement' if tag=='itokawa' else 'no_close_earth_return_for_accepted_daily_outbound_arcs_at_three_tested_turns',extended_future_candidate_verified=bool(audit and audit['no_intermediate_soi_contacts']),extended_return_days=None if candidate is None else candidate['tof_return_days'],extended_return_revolutions=None if candidate is None else candidate['return_revolutions'])
    rows.append(row)
    dynamics[tag]=dict(summary=row,outbound_metadata={k:v for k,v in out.items() if k not in ('outbound_matches','best_outbound')},short_return_analysis=itokawa if tag=='itokawa' else (ccshort if tag=='cc' else fgshort),extended_candidate=candidate,extended_physical_first_entry_audit=audit,physical_data_provenance=p)

source=BASE/'CR3BP_prova_GA_20.py'
main_hash=hashlib.sha256(source.read_bytes()).hexdigest()
recorded_hash=coverage['sources'][str(source)]['sha256']
limitations=[
    'Only three objects have an observational mass and diameter in the local physical JSON. The remaining13052 are not declared infeasible: they are not dynamically evaluated with a measured mass. FE14 has an assumed-density mass and is excluded.',
    'All13055 catalogue entries are censused. Complete outbound enumeration covers the stated daily departure/encounter grid,763 statistical velocity disks and all feasible Lambert branches up to2revolutions for the three eligible objects.',
    'The short-return calculation propagates physically unpowered turns at minus maximum,zero and plus maximum. It is not an exhaustive search over every continuous departure/encounter date and every intermediate turn angle.',
    'Itokawa received50 local continuous refinements within3days of the best daily seeds. No accepted outbound seed emerged.',
    'SOI position centroids are projected radially onto the SOI sphere, preserving the catalogue velocity means,velocity radii and ETD phase conventions.',
    'Long-return CC/FG3 examples and the historical Itokawa example are explicitly excluded from the current200..499day-return results.',
    'Nominal observational masses and catalogue radii are used. Planar heliocentric two-body arcs and point-mass asteroid flybys do not include asteroid satellites,three-dimensional effects or reconstruction of an individual lunar catalogue member.'
]
report=dict(generated_at_local=datetime.now(ZoneInfo('Pacific/Auckland')).isoformat(),timezone='Pacific/Auckland',objective='Catalogue analysis with minimum asteroid-centred periapsis4R and observational masses',catalogue_census=coverage,analysis_configuration=dict(jd_departure_and_encounter_start=2461326.5,jd_departure_and_encounter_end=2462056.5,departure_step_days=1,encounter_step_days=1,corona_au=[.6,1.4],cluster_count=763,n_sigma=2.5,outbound_maximum_revolutions=2,current_return_days=[200,499],current_return_maximum_revolutions=3,periapsis_rule='rp_min=4R; minimum altitude=3R',soi_positions_normalized=True,velocity_speed_matching_tolerance_km_s=.001),total_lambert_problem_calls=calls,total_lambert_branches_checked=branches,total_accepted_outbound_arcs=sum(r['accepted_outbound_count'] for r in rows),current_complete_candidates_found=0,known_future_long_return_candidates_verified=2,results=dynamics,limitations=limitations,ga20_source_changed_during_analysis=(main_hash!=recorded_hash),ga20_source_sha256=main_hash)
write('risultati.json',report)
with (OUT/'riepilogo_bersagli.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
    for row in rows:w.writerow({**row,'outbound_branch_counts':json.dumps(row['outbound_branch_counts'])})
with (TMP/'ga20_all_rp4r_coverage.csv').open(newline='') as f:allrows=list(csv.DictReader(f))
extra=['dynamical_search_status','accepted_outbound_count_daily','complete_current_candidates_found']
lookup={str(r['catalogue_index']):r for r in rows}
with (OUT/'catalogo_completo_13055.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0])+extra);w.writeheader()
    for r in allrows:
        item=lookup.get(str(r['catalogue_index']))
        if item:r.update(dynamical_search_status=item['current_status'],accepted_outbound_count_daily=item['accepted_outbound_count'],complete_current_candidates_found=0)
        else:r.update(dynamical_search_status='not_evaluated_observational_mass_or_radius_missing',accepted_outbound_count_daily='',complete_current_candidates_found='')
        w.writerow(r)

# Retain raw numerical results and provenance; do not copy7GB of old archives.
for name in ['ga20_all_rp4r_coverage.json','ga20_rp4r_itokawa_report.json','ga20_rp4r_cc_three_turns_report.json','ga20_rp4r_fg3_return_coasts.json','ga20_rp4r_cc_current_return_report.json','ga20_rp4r_cc_extended_candidate.json','ga20_rp4r_cc_extended_soi_validation.json','ga20_rp4r_fg3_extended_candidate.json','ga20_rp4r_fg3_surface_soi_validation.json','ga20_rp4r_itokawa_historical_candidate.json','ga20_rp4r_itokawa_historical_soi_validation.json']:
    shutil.copy2(TMP/name,DETAIL/name)
for tag in dynamics:
    shutil.copy2(TMP/f'ga20_rp4r_{tag}_outbound.json',DETAIL/f'ga20_rp4r_{tag}_outbound.json')
for tag in ['cc','fg3']:
    shutil.copy2(TMP/f'ga20_rp4r_{tag}_extended_plot_transfer.png',OUT/f'missione_estesa_{tag}.png')
    shutil.copy2(TMP/f'ga20_rp4r_{tag}_extended_plot_validation.json',DETAIL/f'verifica_indipendente_{tag}.json')

# The formula is exactly the hyperbolic point-mass limit used by GA20.
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
speeds=np.geomspace(.05,25,500)
for row,color in zip(rows,['tab:blue','tab:orange','tab:green']):
    mu=6.67430e-20*row['mass_kg'];R=row['radius_km']
    new=2*np.arcsin(mu/(mu+4*R*speeds**2));old=2*np.arcsin(mu/(mu+(R+50)*speeds**2))
    label=row['name'];axs[0].loglog(speeds,new,color=color,label=label)
    axs[0].loglog(speeds,old,color=color,ls='--',alpha=.65)
    axs[1].loglog(speeds,2*speeds*np.sin(new/2)*1000,color=color,label=label)
axs[0].set(xlabel='Velocità relativa all’asteroide [km/s]',ylabel='Deviazione massima [rad]',title='Pericentro minimo:4R (continuo), R+50km (tratteggio)')
axs[1].set(xlabel='Velocità relativa all’asteroide [km/s]',ylabel='Variazione vettoriale massima [m/s]',title='Effetto massimo del flyby con pericentro4R')
for ax in axs:ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Modello puntiforme planare — masse e raggi nominali del catalogo')
fig.savefig(OUT/'effetto_nuova_distanza.png',dpi=180);plt.close(fig)

lines=[
    'ANALISI GA20 — PERICENTRO MINIMO4R',
    '',
    'Catalogo censito:13055 asteroidi;158 diametri;3 masse osservative con raggio disponibile.',
    'FE14: massa da densita assunta, esclusa come richiesto dalle preferenze precedenti.',
    'Finestra partenza/incontro:JD2461326.5–2462056.5TDB. Ritorni:200–499giorni.',
    'Tutti763cluster, dischi2.5sigma; tutti rami Lambert di andata fino a2rivoluzioni.',
    'Distanza minima dal centro4R: quota minima3R. Centroidi di posizione proiettati sullaSOI.',
    '',
    f'Problemi Lambert risolti:{calls:,}; rami esaminati:{branches:,}.',
    f'Andate compatibili:{sum(r["accepted_outbound_count"] for r in rows):,}.',
    'Nessun collegamento completo trovato nella ricerca breve descritta.',
    'L’assenza si riferisce alla griglia e agli angoli campionati, non a tutte le epoche continue.',
    ''
]
for r in rows:
    lines.append(f'{r["name"]}, indice{r["catalogue_index"]}:rp_min={r["min_periapsis_km"]:.3f}km, quota={r["min_altitude_km"]:.3f}km; andate={r["accepted_outbound_count"]}; candidati completi brevi=0.')
lines+=['','Confronto esteso, fuori200–499giorni:']
for tag in ['fg3','cc']:
    c=dynamics[tag]['extended_candidate'];a=dynamics[tag]['extended_physical_first_entry_audit']
    lines.append(f'{c["body"]}:ritorno{c["tof_return_days"]:.6f}giorni,{c["return_revolutions"]}rivoluzioni; primo ingressoSOI finale verificato={a["no_intermediate_soi_contacts"]}.')
lines+=['Itokawa storico2000–2001 resta valido, fuori dalla finestra attuale.','','Limiti e metodo:']+['- '+s for s in limitations]
lines+=['','File:risultati.json contiene parametri, conteggi e verifiche; catalogo_completo_13055.csv conserva uno stato per ogni asteroide; riepilogo_bersagli.csv contiene i tre bersagli valutabili.']
(OUT/'report.txt').write_text('\n'.join(lines)+'\n')
manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.rglob('*') if p.is_file()}
write('manifest.json',manifest)
assert len(allrows)==13055 and len({r['catalogue_index'] for r in allrows})==13055
assert len(rows)==3 and sum(r['accepted_outbound_count'] for r in rows)==11704
assert calls==357885913 and branches==541139579
assert all(dynamics[t]['extended_physical_first_entry_audit']['no_intermediate_soi_contacts'] for t in ['cc','fg3'])
print('OUTPUT',OUT)
print('TOTAL',calls,'Lambert;',branches,'branches;',sum(r['accepted_outbound_count'] for r in rows),'outbound;0current complete')
print('SOURCE_UNCHANGED',main_hash==recorded_hash)

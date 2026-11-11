"""Exact bounded outbound screening; optional CLI catalogue index (default40)."""
import sys,os,json,time
os.environ.setdefault('MPLCONFIGDIR','/tmp/ga20_rp4r_mpl')
sys.path.insert(0,'/tmp')
import ga20_diagnose_1994cc as d
import numpy as np
import pykep as pk
import spiceypy as spice
from joblib import Parallel,delayed,parallel_config

START,END=2461326.5,2462056.5

def scan(jd0,r0,vc0,ids,radius,jdas,ras,vas):
    matches=[];calls=branches=errors=0;best=[]
    for jda,ra,va in zip(jdas,ras,vas):
        tof=jda-jd0
        if tof<=0:continue
        for c in range(len(ids)):
            calls+=1
            try:lp=pk.lambert_problem(r0[c],ra,tof*d.DAY,d.MU,False,2)
            except RuntimeError:
                errors+=1;continue
            for b,(v0,v1) in enumerate(zip(lp.v0,lp.v1)):
                branches+=1
                dx=v0[0]-vc0[c,0];dy=v0[1]-vc0[c,1]
                ratio2=(dx*dx+dy*dy)/(radius[c]*radius[c])
                if ratio2<=1:
                    matches.append(dict(jd_departure=float(jd0),jd_asteroid=float(jda),departure_cluster_id=int(ids[c]),outbound_branch=b,tof_out_days=float(tof),departure_disk_ratio=float(np.sqrt(ratio2)),arrival_velocity_heliocentric=np.asarray(v1).tolist(),vinf_minus=(np.asarray(v1)-va).tolist(),vinf_speed=float(np.linalg.norm(np.asarray(v1)-va))))
                if len(best)<3 or ratio2<best[-1][0]:
                    best.append((float(ratio2),float(jd0),float(jda),int(ids[c]),b))
                    best.sort();best=best[:3]
    return calls,branches,errors,matches,best

if __name__=='__main__':
    before=time.perf_counter();spice.furnsh(pk.udpla.de440s.kernel_file())
    index=int(sys.argv[1]) if len(sys.argv)>1 else 40
    raw=json.loads((d.FOLDER/'asteroids_sbdb_all_i5.json').read_text());rec=dict(zip(raw['fields'],raw['data'][index]))
    physical=json.loads((d.FOLDER/'asteroids_sbdb_physical.json').read_text())['asteroids'].get(str(rec['spkid']),{})
    body_radius=physical.get('radius_km')
    tag={40:'itokawa',66:'cc',125:'fg3'}.get(index,str(index))
    cat=np.loadtxt(d.FOLDER.parent/'Clustering/Clusters/CHAMPION_GA_16_plots_vinf_time_K780/cluster_velocity_statistics.txt')
    cat=cat[~np.isin(cat[:,0].astype(int),np.arange(763,780))]
    ids=cat[:,0].astype(int);tau=cat[:,7];mv=cat[:,2:4]
    radius=2.5*np.linalg.norm(cat[:,4:6],axis=1)
    soi=(5.974e24/1.989e30)**.4*1.496e8
    mr=cat[:,10:12].copy();mr*=soi/np.linalg.norm(mr,axis=1)[:,None]
    dates=np.arange(START,END+1.)
    targets=[d.asteroid(jd,rec) for jd in dates]
    ras=np.array([v[0] for v in targets]);vas=np.array([v[1] for v in targets])
    radii=np.linalg.norm(ras,axis=1)/(pk.AU/1000)
    mask=(radii>=.6)&(radii<=1.4)
    jdas,ras,vas=dates[mask],ras[mask],vas[mask]
    departures={}
    for jd in dates:
        re,ve=d.earth(jd);theta=d.moon_angles(jd-tau)
        r0=np.tile(re,(len(ids),1));r0[:,:2]+=d.rotate(mr,theta)
        departures[float(jd)]=(r0,d.rotate(mv,theta)+ve[:2])
    print(tag,'daily outbound prepared',len(dates),'departure dates',len(jdas),'encounter dates',flush=True)
    summary=dict(catalogue_index=index,body=rec['full_name'].strip(),mass_kg=physical.get('mass_kg'),radius_km=body_radius,min_periapsis_km=None if body_radius is None else 4*body_radius,jd_start=START,jd_end=END,departure_step_days=1.,encounter_step_days=1.,outbound_multi_revs=2,normalized_soi_positions=True,lambert_problem_calls=0,lambert_branches_checked=0,solver_errors=0,outbound_matches=[],best_outbound=[])
    with parallel_config(backend='loky',n_jobs=4,inner_max_num_threads=1):
        results=Parallel(return_as='generator',batch_size=1)(delayed(scan)(float(jd),*departures[float(jd)],ids,radius,jdas,ras,vas) for jd in dates)
        for n,(calls,branches,errors,matches,best) in enumerate(results):
            summary['lambert_problem_calls']+=calls;summary['lambert_branches_checked']+=branches;summary['solver_errors']+=errors
            summary['outbound_matches'].extend(matches);summary['best_outbound'].extend(best)
            if n%50==0:print(tag,'outbound',n+1,'calls',summary['lambert_problem_calls'],'matches',len(summary['outbound_matches']),flush=True)
    summary['best_outbound']=sorted(summary['best_outbound'])[:100]
    summary['elapsed_seconds']=time.perf_counter()-before
    summary['limitation']='Exhaustive only on the stated daily outbound grid and branch cap; continuous departures/encounters are not excluded by an empty result.'
    out='/tmp/ga20_rp4r_'+tag+'_outbound.json';open(out,'w').write(json.dumps(summary,indent=2))
    print(tag,'done',summary['lambert_problem_calls'],'calls',summary['lambert_branches_checked'],'branches',len(summary['outbound_matches']),'matches',summary['elapsed_seconds'],'s',flush=True)
    print('best',summary['best_outbound'][:3],flush=True)

"""Independent three-angle continuation of every accepted CC daily outbound arc."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,'/tmp')
import ga20_diagnose_1994cc as d
import numpy as np
import pykep as pk
import spiceypy as spice
from scipy.optimize import minimize_scalar
from joblib import Parallel,delayed,parallel_config

def process(rows,ra,va,earths,tofs,muast,rpmin):
    best={s:None for s in (-1,0,1)}
    max_pair_separation=0.;velocity_bound=0.;checks=0
    for row in rows:
        vin=np.asarray(row['arrival_velocity_heliocentric'])-va
        speed=np.linalg.norm(vin);maximum=2*np.arcsin(muast/(muast+rpmin*speed**2))
        rzero=None
        # Zero first enables an explicit positional comparison of both extremes.
        for sign in (0,-1,1):
            angle=sign*maximum;c,s=np.cos(angle),np.sin(angle)
            rot=vin.copy();rot[:2]=[c*vin[0]-s*vin[1],s*vin[0]+c*vin[1]]
            v=va+rot
            pp=np.asarray(pk.propagate_lagrangian_grid([ra,v],np.r_[0.,tofs]*d.DAY,d.MU))[1:,0,:]
            distance=np.linalg.norm(pp-earths,axis=1);i=int(distance.argmin());checks+=len(distance)
            if sign==0:rzero=pp
            else:max_pair_separation=max(max_pair_separation,float(np.linalg.norm(pp-rzero,axis=1).max()))
            h=np.cross(ra,v);evec=np.cross(v,h)/d.MU-ra/np.linalg.norm(ra)
            ecc=np.linalg.norm(evec);peri=np.dot(h,h)/(d.MU*(1+ecc))
            vmax=np.sqrt(d.MU*(1+ecc)/peri);velocity_bound=max(velocity_bound,float(vmax))
            if best[sign] is None or distance[i]<best[sign]['distance_km']:
                best[sign]=dict(distance_km=float(distance[i]),return_tof_days=float(tofs[i]),sign=sign,turn_angle_rad=float(angle),max_turn_angle_rad=float(maximum),outbound=row)
    return best,max_pair_separation,velocity_bound,checks

if __name__=='__main__':
    start=time.perf_counter();spice.furnsh(pk.udpla.de440s.kernel_file())
    data=json.loads(Path('/tmp/ga20_rp4r_cc_outbound.json').read_text());rows=data['outbound_matches']
    raw=json.loads((d.FOLDER/'asteroids_sbdb_all_i5.json').read_text());rec=dict(zip(raw['fields'],raw['data'][66]))
    physical=json.loads((d.FOLDER/'asteroids_sbdb_physical.json').read_text())['asteroids']['20136617']
    muast=6.67430e-20*physical['mass_kg'];rpmin=4*physical['radius_km']
    tofs=np.arange(200.,500.,1.)
    encs=sorted({row['jd_asteroid'] for row in rows})
    states={jd:d.asteroid(jd,rec) for jd in encs}
    earthdates=sorted({jd+t for jd in encs for t in tofs})
    ecache={jd:d.earth(jd)[0] for jd in earthdates}
    best={s:None for s in (-1,0,1)};separation=0.;vbound=0.;checks=0
    with parallel_config(backend='loky',n_jobs=4,inner_max_num_threads=1):
        jobs=Parallel(return_as='generator',batch_size=1)(delayed(process)([row for row in rows if row['jd_asteroid']==jd],*states[jd],np.array([ecache[jd+t] for t in tofs]),tofs,muast,rpmin) for jd in encs)
        for g,(b,sep,vmax,n) in enumerate(jobs):
            separation=max(separation,sep);vbound=max(vbound,vmax);checks+=n
            for sign,item in b.items():
                if best[sign] is None or item['distance_km']<best[sign]['distance_km']:best[sign]=item
            if g%15==0:print('CC three turns',g+1,'/',len(encs),'minimum',min(x['distance_km'] for x in best.values()),flush=True)
    # Refine the global minimum of each physical rotation in continuous time.
    for sign,item in best.items():
        row=item['outbound'];jd=row['jd_asteroid'];ra,va=states[jd]
        vin=np.asarray(row['arrival_velocity_heliocentric'])-va;angle=item['turn_angle_rad'];c,s=np.cos(angle),np.sin(angle)
        vrel=vin.copy();vrel[:2]=[c*vin[0]-s*vin[1],s*vin[0]+c*vin[1]];vsolar=va+vrel
        def distance(t):
            r=np.asarray(pk.propagate_lagrangian([ra,vsolar],t*d.DAY,d.MU)[0]);return float(np.linalg.norm(r-d.earth(jd+t)[0]))
        t=item['return_tof_days'];lo=max(200.,t-1);hi=min(499.,t+1)
        opt=minimize_scalar(distance,bounds=(lo,hi),method='bounded',options={'xatol':1e-7})
        if opt.fun<item['distance_km']:item.update(refined_distance_km=float(opt.fun),refined_return_tof_days=float(opt.x))
        else:item.update(refined_distance_km=item['distance_km'],refined_return_tof_days=t)
    minimum=min(x['refined_distance_km'] for x in best.values())
    # Conservative interpolation bound for each sampled trajectory, assuming
    # heliocentric Earth speed <=50 km/s throughout these ephemeris dates.
    lower=min(x['distance_km'] for x in best.values())-(vbound+50.)*d.DAY/2
    out=dict(catalogue_index=66,body=rec['full_name'].strip(),mass_kg=physical['mass_kg'],radius_km=physical['radius_km'],minimum_periapsis_km=rpmin,outbound_arc_count=len(rows),distinct_encounter_dates=len(encs),return_tof_min_days=200.,return_tof_max_days_inclusive=499.,sample_step_days=1.,rotation_choices=['minus_delta_max','zero','plus_delta_max'],distance_evaluations=checks,propagated_trajectory_count=3*len(rows),minimum_distance_km=minimum,best_by_rotation=best,max_extreme_vs_zero_position_separation_km=separation,maximum_spacecraft_perihelion_speed_bound_km_s=vbound,assumed_earth_speed_upper_bound_km_s=50.,sample_interpolation_distance_lower_bound_km=float(lower),earth_soi_km=(5.974e24/1.989e30)**.4*1.496e8,elapsed_seconds=time.perf_counter()-start,scope='All accepted daily outbound arcs and three stated physical turn angles; not an exhaustive search over continuous departure or encounter dates or arbitrary scattering angles.')
    Path('/tmp/ga20_rp4r_cc_three_turns_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='best_by_rotation'},indent=2))

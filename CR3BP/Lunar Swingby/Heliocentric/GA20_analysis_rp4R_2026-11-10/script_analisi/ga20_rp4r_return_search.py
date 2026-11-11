"""Bounded continuous returns from all daily outbound branch matches."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,'/tmp')
import ga20_diagnose_1994cc as d
import numpy as np
import pykep as pk
import spiceypy as spice
from scipy.optimize import least_squares

if __name__=='__main__':
    index=int(sys.argv[1]);tag={66:'cc',125:'fg3',40:'itokawa'}[index]
    start=time.perf_counter();spice.furnsh(pk.udpla.de440s.kernel_file())
    outbound=json.loads(Path(f'/tmp/ga20_rp4r_{tag}_outbound.json').read_text())
    rows=outbound['outbound_matches']
    raw=json.loads((d.FOLDER/'asteroids_sbdb_all_i5.json').read_text());rec=dict(zip(raw['fields'],raw['data'][index]))
    p=json.loads((d.FOLDER/'asteroids_sbdb_physical.json').read_text())['asteroids'][str(rec['spkid'])]
    mu_ast=6.67430e-20*p['mass_kg'];rp=2*p['diameter_km']
    data=np.loadtxt(d.FOLDER.parent/'Clustering/Clusters/CHAMPION_GA_16_plots_vinf_time_K780/cluster_velocity_statistics.txt')
    data=data[~np.isin(data[:,0].astype(int),np.arange(763,780))]
    ids=data[:,0].astype(int);lookup={int(cid):k for k,cid in enumerate(ids)}
    tau=data[:,7];mv=data[:,2:4];mr=data[:,10:12].copy();soi=(5.974e24/1.989e30)**.4*1.496e8
    mr*=soi/np.linalg.norm(mr,axis=1)[:,None];radius=2.5*np.linalg.norm(data[:,4:6],axis=1)
    def endpoint(jd,k,incoming=False):
        er,ev=d.earth(jd);theta=d.moon_angles([jd+(1 if incoming else -1)*tau[k]])
        er[:2]+=d.rotate(mr[k:k+1]*([1.,-1.] if incoming else [1.,1.]),theta)[0]
        ev[:2]+=d.rotate(mv[k:k+1]*([-1.,1.] if incoming else [1.,1.]),theta)[0]
        return er,ev
    def make_turn(vm):
        delta=np.arcsin(mu_ast/(mu_ast+rp*np.dot(vm,vm)))
        vp=vm.copy();vp[:2]=d.rotate(vm[None,:2],np.array([delta]))[0]
        return vp,delta
    cap_cache={};earth_cache={};ast_cache={}
    def caps(jd):
        if jd not in cap_cache:
            er,ev=d.earth(jd);theta=d.moon_angles(jd+tau)
            rr=np.tile(er,(len(ids),1));rr[:,:2]+=d.rotate(mr*[1.,-1.],theta)
            cap_cache[jd]=(rr,d.rotate(mv*[-1.,1.],theta)+ev[:2])
        return cap_cache[jd]
    def get_vm(row,va):
        for key in ('v_inf_minus','vinf_relative'):
            if key in row:return np.asarray(row[key])
        for key in ('arrival_velocity_heliocentric','v_arrival_heliocentric'):
            if key in row:return np.asarray(row[key])-va
        return np.asarray(row['vinf_minus'])-va
    seeds=[];best_distance=np.inf;coasts=0
    grid=np.arange(200.,500.,2.);grid=np.r_[grid,499.]
    for ri,row in enumerate(rows):
        jd0=row['jd_departure'];jda=row['jd_asteroid']
        if jda not in ast_cache:ast_cache[jda]=d.asteroid(jda,rec)
        ra,va=ast_cache[jda];vm=get_vm(row,va);vp,_=make_turn(vm)
        st=np.asarray(pk.propagate_lagrangian_grid([ra,va+vp],np.r_[0.,grid]*d.DAY,d.MU))[1:]
        if jda not in earth_cache:earth_cache[jda]=np.array([d.earth(jda+t)[0] for t in grid])
        dist=np.linalg.norm(st[:,0,:]-earth_cache[jda],axis=1)
        coasts+=1;best_distance=min(best_distance,float(dist.min()))
        local=list(np.flatnonzero((dist[1:-1]<dist[:-2])&(dist[1:-1]<dist[2:]))+1)
        local+=list(np.argsort(dist)[:1])
        for m in sorted(set(local),key=lambda i:dist[i])[:2]:
            if dist[m]>8e7:continue
            rc,vc=caps(float(jda+grid[m]));dr=np.linalg.norm(rc-st[m,0],axis=1);dv=np.linalg.norm(vc-st[m,1,:2],axis=1)
            score=(dr/soi)**2+(dv/.2)**2
            for k in np.argsort(score)[:2]:
                seeds.append(dict(row_index=ri,capture_index=int(k),return_tof=float(grid[m]),score=float(score[k]),sample_distance_km=float(dist[m])))
        if ri%2000==0:print(tag,'coasts',ri+1,'/',len(rows),'seeds',len(seeds),'minEarth',best_distance,flush=True)
    seeds.sort(key=lambda z:z['score'])
    Path(f'/tmp/ga20_rp4r_{tag}_current_seeds.json').write_text(json.dumps(seeds,indent=2))
    candidates=[];roots=[];attempts=0
    # The root uses a deterministic physical rotation of half the permitted
    # maximum; possible turn angles and continuous departure dates are not
    # exhaustively searched. Negative results are explicitly bounded.
    for seed in seeds[:2500]:
        row=rows[seed['row_index']];jd0=row['jd_departure'];c=lookup[row['departure_cluster_id']];k=seed['capture_index'];b=row['outbound_branch']
        r0,vc0=endpoint(jd0,c)
        def state(x):
            t1,t2=x;ra,va=d.asteroid(jd0+t1,rec)
            lp=pk.lambert_problem(r0,ra,t1*d.DAY,d.MU,False,2)
            if b>=len(lp.v0):raise ValueError('outbound branch unavailable')
            v0=np.asarray(lp.v0[b]);vin=np.asarray(lp.v1[b]);vp,delta=make_turn(vin-va);vout=va+vp
            rf,vf=map(np.asarray,pk.propagate_lagrangian([ra,vout],t2*d.DAY,d.MU));rc,vcc=endpoint(jd0+t1+t2,k,True)
            return rf-rc,v0,vf,rc,vcc,ra,va,vin,vout,delta
        def residual(x):
            try:return state(x)[0][:2]/soi
            except (RuntimeError,ValueError):return np.full(2,1e6)
        x0=[row['tof_out_days'],seed['return_tof']]
        opt=least_squares(residual,x0,bounds=([1.,200.],[min(730.,2462056.5-jd0),499.]),diff_step=1e-6,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=70)
        attempts+=1;val=state(opt.x);err=np.linalg.norm(val[0]);r0ratio=np.linalg.norm(val[1][:2]-vc0[:2])/radius[c];r1ratio=np.linalg.norm(val[2][:2]-val[4][:2])/radius[k]
        if err<.01:
            roots.append(dict(seed=seed,times=opt.x.tolist(),position_error_km=float(err),departure_ratio=float(r0ratio),capture_ratio=float(r1ratio)))
        if err<.01 and max(r0ratio,r1ratio)<=1:
            t1,t2=opt.x;rr,vc,vf,rc,vcc,ra,va,vin,vout,delta=val
            back=pk.lambert_problem(ra,rc,t2*d.DAY,d.MU,False,3)
            br=int(np.argmin([np.linalg.norm(np.asarray(v)-vout) for v in back.v0]))
            bc=np.asarray(back.v1[br]);vpl=np.asarray(back.v0[br])-va;vmi=vin-va
            angle=np.arctan2(np.linalg.norm(np.cross(vmi,vpl)),np.dot(vmi,vpl));limit=2*np.arcsin(mu_ast/(mu_ast+rp*np.dot(vmi,vmi)))
            ok=angle<=limit and abs(np.linalg.norm(vmi)-np.linalg.norm(vpl))<=1e-3 and np.linalg.norm(bc[:2]-vcc[:2])<=radius[k]
            ca=dict(catalogue_index=index,body=rec['full_name'].strip(),mass_kg=p['mass_kg'],radius_km=p['diameter_km']/2,min_periapsis_km=rp,surface_normalized=True,jd_departure=jd0,jd_asteroid=float(jd0+t1),jd_capture=float(jd0+t1+t2),tof_out_days=float(t1),tof_return_days=float(t2),departure_cluster_id=int(ids[c]),capture_cluster_id=int(ids[k]),outbound_branch=b,return_branch=br,return_revolutions=int(np.ceil(br/2)),departure_residual=float(np.linalg.norm(vc[:2]-vc0[:2])),capture_residual=float(np.linalg.norm(bc[:2]-vcc[:2])),departure_radius=float(radius[c]),capture_radius=float(radius[k]),r0=r0.tolist(),ra=ra.tolist(),r1=rc.tolist(),v0=vc.tolist(),vin=vin.tolist(),vout=np.asarray(back.v0[br]).tolist(),v1=bc.tolist(),asteroid_velocity=va.tolist(),turn_angle_rad=float(angle),max_turn_angle_rad=float(limit),speed_residual=float(abs(np.linalg.norm(vmi)-np.linalg.norm(vpl))),two_lambert_constraints_pass=bool(ok),first_soi_entry_verified=False)
            candidates.append(ca)
            Path(f'/tmp/ga20_rp4r_{tag}_current_candidates.json').write_text(json.dumps(candidates,indent=2))
            print(tag,'CANDIDATE',ca,flush=True)
            if ok:break
        if attempts%250==0:print(tag,'roots attempted',attempts,'closed',len(roots),'best disks',min((max(z['departure_ratio'],z['capture_ratio']) for z in roots),default=None),flush=True)
    report=dict(catalogue_index=index,body=rec['full_name'].strip(),return_tof_min_days=200.,return_tof_max_days=499.,rp_min_km=rp,outbound_coasts_screened=coasts,coast_grid_step_days=2.,coast_min_earth_distance_km=float(best_distance) if coasts else None,capture_seeds=len(seeds),continuous_attempts=attempts,closed_roots=len(roots),best_roots=sorted(roots,key=lambda z:max(z['departure_ratio'],z['capture_ratio']))[:30],candidates=candidates,elapsed_seconds=time.perf_counter()-start,limitations='Daily departure dates and outbound seed grid; screening of physical half-max positive turns; bounded continuous refinement of up to2500 selected encounter/return seeds. No global nonexistence proof.')
    Path(f'/tmp/ga20_rp4r_{tag}_current_return_report.json').write_text(json.dumps(report,indent=2))
    print(tag,'RETURNS DONE',len(candidates),'candidates',time.perf_counter()-start,'seconds',flush=True)

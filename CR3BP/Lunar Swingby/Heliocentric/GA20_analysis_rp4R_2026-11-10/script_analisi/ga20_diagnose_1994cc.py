"""Read-only reproduction of GA20 rejection metrics, reusing identical arcs."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ga20_diagnostic_mpl')
import json
import time
from pathlib import Path
import numpy as np
import pykep as pk
import spiceypy as spice
from scipy.optimize import newton
from joblib import Parallel, delayed, parallel_config

FOLDER = Path('/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Heliocentric')
START, END = 2461326.5, 2462056.5
DAY = 86400.
MU = pk.MU_SUN / 1e9
TOL = 1e-3


def moon_angles(jds):
    return np.array([np.arctan2(s[1], s[0]) for s in
        (spice.spkezr('MOON', float((jd-2451545.)*DAY), 'ECLIPJ2000', 'NONE', 'EARTH')[0]
         for jd in jds)])


def earth(jd):
    s = np.asarray(spice.spkezr('EARTH', float((jd-2451545.)*DAY), 'ECLIPJ2000', 'NONE', 'SUN')[0])
    s[[2,5]] = 0.
    return s[:3], s[3:]


def rotate(rows, angles):
    c, s = np.cos(angles), np.sin(angles)
    return np.column_stack((c*rows[:,0]-s*rows[:,1],s*rows[:,0]+c*rows[:,1]))


def asteroid(jd, rec):
    a, e = float(rec['a'])*pk.AU/1000, float(rec['e'])
    m = (np.deg2rad(float(rec['ma']))+np.sqrt(MU/a**3)*(jd-float(rec['epoch']))*DAY) % (2*np.pi)
    E = newton(lambda x:x-e*np.sin(x)-m, m-e/2 if m>np.pi else m+e/2)
    nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    return tuple(np.asarray(x) for x in pk.par2ic([a,e,0.,0.,np.deg2rad((float(rec['om'])+float(rec['w']))%360),nu],MU))


def best_record(index, incoming_index, residual, mode, rows, dates, tofs, ids, sin, sout, incoming, outgoing, limits):
    # A nearest-speed exemplar can have a different incoming vector from the
    # nearest-direction exemplar, so evaluate this pair's angle explicitly.
    vminus, vplus = incoming[incoming_index], outgoing[index]
    angle = np.arctan2(np.linalg.norm(np.cross(vminus, vplus)), np.dot(vminus, vplus))
    return dict(mode=mode, departure_jd=float(rows[incoming_index,0]),
                departure_cluster=int(rows[incoming_index,1]), encounter_jd=float(rows[incoming_index,2]),
                capture_jd=float(dates[index]), return_tof=float(tofs[index]), capture_cluster=int(ids[index]),
                speed_in=float(sin[incoming_index]), speed_out=float(sout[index]),
                residual=float(residual), turn_angle=float(angle), max_turn=float(limits[incoming_index]))


def analyze_group(jd, rows, incoming, r_ast, v_ast, capture_cache, cluster_ids, radius, mu_ast, rp):
    nin = len(rows)
    tofs = np.arange(50,200,dtype=float)
    ncl = len(cluster_ids)
    vp = np.empty((len(tofs),ncl,3))
    passes = np.empty((len(tofs),ncl),dtype=bool)
    capture_distances = np.empty((len(tofs),ncl))
    capture_velocities = np.empty_like(vp)
    for t, tof in enumerate(tofs):
        captures, centers, earth_velocity = capture_cache[float(jd+tof)]
        for k in range(ncl):
            lam = pk.lambert_problem(r0=r_ast,r1=captures[k],tof=tof*DAY,mu=MU,cw=False,multi_revs=0)
            vp[t,k] = np.asarray(lam.v0[0])-v_ast
            capture_velocities[t,k] = np.asarray(lam.v1[0])
            d = centers[k]-capture_velocities[t,k,:2]
            capture_distances[t,k] = np.linalg.norm(d)
            passes[t,k] = np.dot(d,d)<=radius[k]**2
    vp = vp.reshape(-1,3)
    cap = passes.ravel()
    capture_ratios = capture_distances / radius[None,:]
    best_t,best_k = np.unravel_index(int(capture_ratios.argmin()),capture_ratios.shape)
    best_capture_jd = float(jd+tofs[best_t])
    best_captures,best_centers,best_earth_velocity = capture_cache[best_capture_jd]
    best_geo = capture_velocities[best_t,best_k]-best_earth_velocity
    sin = np.linalg.norm(incoming,axis=1)
    sout = np.linalg.norm(vp,axis=1)
    limits = 2*np.arcsin(mu_ast/(mu_ast+rp*sin**2))
    theta_in, theta_out = np.arctan2(incoming[:,1],incoming[:,0]),np.arctan2(vp[:,1],vp[:,0])
    order = np.argsort(sin)
    sorted_speed = sin[order]
    at = np.searchsorted(sorted_speed,sout)
    left,right = np.maximum(at-1,0),np.minimum(at,nin-1)
    il,ir = order[left],order[right]
    nearest = np.where(np.abs(sin[il]-sout)<=np.abs(sin[ir]-sout),il,ir)
    residuals = np.abs(sin[nearest]-sout)
    speed_candidates = np.searchsorted(sorted_speed,sout+TOL,side='right')-np.searchsorted(sorted_speed,sout-TOL,side='left')
    # Circular nearest directions: sorting is equivalent to minimizing the full
    # pair matrix, but only two adjacent candidates need exact atan2 evaluation.
    angle_order = np.argsort(theta_in)
    base = theta_in[angle_order]
    sorted_angles = np.concatenate((base-2*np.pi,base,base+2*np.pi))
    angular_ids = np.tile(angle_order,3)
    at_angle = np.searchsorted(sorted_angles,theta_out)
    left = angular_ids[np.maximum(at_angle-1,0)]
    right = angular_ids[np.minimum(at_angle,len(sorted_angles)-1)]
    def exact_angles(which):
        return np.arctan2(np.linalg.norm(np.cross(incoming[which],vp),axis=1),
                          np.einsum('ij,ij->i',incoming[which],vp))
    al,ar = exact_angles(left),exact_angles(right)
    angle_nearest = np.where(al<=ar,left,right)
    angles = np.minimum(al,ar)
    metrics = dict(encounter_jd=float(jd),incoming=nin,distinct_return_calls=len(vp),
                   total_pairs=nin*len(vp),capture_pairs=int(cap.sum())*nin,
                   speed_pairs=int(speed_candidates.sum()),speed_capture_pairs=int(speed_candidates[cap].sum()),
                   min_speed_residual=float(residuals.min()),
                   min_speed_residual_capture=float(residuals[cap].min()) if cap.any() else None,
                   min_angle=float(angles.min()),min_angle_capture=float(angles[cap].min()) if cap.any() else None,
                   max_allowed_angle=float(limits.max()),speed_in_range=[float(sin.min()),float(sin.max())],
                   speed_out_range=[float(sout.min()),float(sout.max())])
    metrics['min_capture_ratio'] = float(capture_ratios.min())
    metrics['best_capture_velocity'] = dict(encounter_jd=float(jd), capture_jd=best_capture_jd, return_tof=float(tofs[best_t]), capture_cluster=int(cluster_ids[best_k]), normalized_residual=float(capture_ratios.min()), residual_km_s=float(capture_distances[best_t,best_k]), permitted_radius_km_s=float(radius[best_k]), actual_geocentric_speed_km_s=float(np.linalg.norm(best_geo)), actual_geocentric_velocity_km_s=best_geo.tolist(), mean_catalogue_geocentric_velocity_km_s=(best_centers[best_k]-best_earth_velocity[:2]).tolist())
    flat_tof = np.repeat(tofs,ncl)
    dates = jd+flat_tof
    ids = np.tile(cluster_ids,len(tofs))
    metrics['best_speed'] = best_record(int(residuals.argmin()),int(nearest[residuals.argmin()]),residuals.min(),
        'speed',rows,dates,flat_tof,ids,sin,sout,incoming,vp,limits)
    a = int(angles.argmin()); ai = int(angle_nearest[a])
    metrics['best_angle'] = best_record(a,ai,abs(sout[a]-sin[ai]),'angle',rows,dates,flat_tof,ids,sin,sout,incoming,vp,limits)
    if cap.any():
        cap_idx = np.flatnonzero(cap)
        i = int(cap_idx[residuals[cap].argmin()])
        metrics['best_speed_capture'] = best_record(i,int(nearest[i]),residuals[i],'speed_capture',rows,dates,flat_tof,ids,sin,sout,incoming,vp,limits)
        i = int(cap_idx[angles[cap].argmin()]); ai = int(angle_nearest[i])
        metrics['best_angle_capture'] = best_record(i,ai,abs(sout[i]-sin[ai]),'angle_capture',rows,dates,flat_tof,ids,sin,sout,incoming,vp,limits)
    return metrics


if __name__ == '__main__':
    before = time.perf_counter()
    spice.furnsh(pk.udpla.de440s.kernel_file())
    raw = json.loads((FOLDER/'asteroids_sbdb_all_i5.json').read_text())
    rec = dict(zip(raw['fields'],raw['data'][66]))
    physical = json.loads((FOLDER/'asteroids_sbdb_physical.json').read_text())['asteroids']['20136617']
    mu_ast,rp = 6.67430e-20*physical['mass_kg'],physical['radius_km']+50
    data = np.loadtxt(FOLDER.parent/'Clustering/Clusters/CHAMPION_GA_16_plots_vinf_time_K780/cluster_velocity_statistics.txt',ndmin=2)
    data = data[~np.isin(data[:,0].astype(int),np.arange(763,780))]
    ids, tau, mr, mv = data[:,0].astype(int),data[:,7],data[:,10:12],data[:,2:4]
    radius = 2.5*np.linalg.norm(data[:,4:6],axis=1)
    rows = np.loadtxt(FOLDER/'GA18_real_asteroids/Asteroide_136617 (1994 CC)/transfers_rendezvous.txt',ndmin=2)
    rows = rows[(rows[:,0]>=START)&(rows[:,0]<=END)&(rows[:,2]<=END)&(rows[:,2]>rows[:,0])]
    assert rows.shape == (2342,5)
    assert len(np.unique(rows[:,:4],axis=0)) == len(rows)
    encounters = np.unique(rows[:,2])
    targets = {float(jd):asteroid(jd,rec) for jd in encounters}
    outgoing_states = {}
    for jd in np.unique(rows[:,0]):
        re,ve = earth(jd)
        theta = moon_angles(jd-tau)
        r = np.tile(re,(len(ids),1));r[:,:2] += rotate(mr,theta)
        outgoing_states[float(jd)] = r,rotate(mv,theta)+ve[:2]
    incoming = np.empty((len(rows),3))
    max_ratio_error = 0.
    for i,row in enumerate(rows):
        jd,cid,jda,tof,archive_speed = row
        c = int(np.searchsorted(ids,int(cid)))
        r,center = outgoing_states[float(jd)]
        ra,va = targets[float(jda)]
        lam = pk.lambert_problem(r0=r[c],r1=ra,tof=tof*DAY,mu=MU,cw=False,multi_revs=0)
        assert np.linalg.norm(center[c]-np.asarray(lam.v0[0])[:2]) <= radius[c]+1e-10
        incoming[i] = np.asarray(lam.v1[0])-va
        max_ratio_error = max(max_ratio_error,abs(np.linalg.norm(incoming[i])-archive_speed))
    print('Andate ricostruite:',len(rows),'errore velocita archivio max:',max_ratio_error,flush=True)
    capture_dates = np.unique((encounters[:,None]+np.arange(50,200)[None,:]).ravel())
    capture_cache = {}
    for jd in capture_dates:
        re,ve = earth(jd)
        theta = moon_angles(jd+tau)
        r = np.tile(re,(len(ids),1));r[:,:2] += rotate(mr*np.array([1.,-1.]),theta)
        capture_cache[float(jd)] = r,rotate(mv*np.array([-1.,1.]),theta)+ve[:2],ve
    print('Ingressi precalcolati:',len(capture_cache),'epoche; gruppi:',len(encounters),flush=True)
    with parallel_config(backend='loky',n_jobs=4,inner_max_num_threads=1):
        results = Parallel(return_as='generator',batch_size=1)(
            delayed(analyze_group)(float(jd),rows[rows[:,2]==jd],incoming[rows[:,2]==jd],
                *targets[float(jd)],capture_cache,ids,radius,mu_ast,rp) for jd in encounters)
        groups = []
        for result in results:
            groups.append(result)
            print('Gruppi completati:',len(groups),'/',len(encounters),flush=True)
    report = dict(groups=groups,mu_asteroid=mu_ast,minimum_radius_km=rp,
                  elapsed_seconds=time.perf_counter()-before)
    report.update({key:sum(g[key] for g in groups) for key in
        ('distinct_return_calls','total_pairs','capture_pairs','speed_pairs','speed_capture_pairs')})
    for key in ('min_speed_residual','min_speed_residual_capture','min_angle','min_angle_capture'):
        report[key] = min(g[key] for g in groups if g[key] is not None) if any(g[key] is not None for g in groups) else None
    report['max_allowed_angle'] = max(g['max_allowed_angle'] for g in groups)
    report['min_capture_ratio'] = min(g['min_capture_ratio'] for g in groups)
    report['best_capture_velocity'] = min((g['best_capture_velocity'] for g in groups),key=lambda r:r['normalized_residual'])
    for key, metric in [('best_speed','speed_residual_km_s'),('best_speed_capture','speed_residual_km_s')]:
        records = [g[key] for g in groups if key in g]
        if records:
            report[key] = min(records,key=lambda r:r['residual'])
    for key in ('best_angle','best_angle_capture'):
        records = [g[key] for g in groups if key in g]
        if records:
            report[key] = min(records,key=lambda r:r['turn_angle'])
    output = Path('/tmp/ga20_1994cc_rejection_report.json')
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('REPORT',json.dumps({k:v for k,v in report.items() if k!='groups'}),flush=True)
    print('Report salvato soltanto in',output,flush=True)

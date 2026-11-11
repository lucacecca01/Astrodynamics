"""Rebuild the two Lambert arcs and plot only the independently checked candidate."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/tmp')
import ga20_diagnose_1994cc as d
import numpy as np
import pykep as pk
import spiceypy as spice
from scipy.optimize import minimize_scalar,brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_path=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp/ga20_itokawa_candidate.json')
candidate=json.loads(input_path.read_text())
if 'tof_out_days' in candidate:
    candidate={**candidate,'name':candidate['body'],'catalogue_index':candidate.get('catalogue_index',125),
               'jd_flyby':candidate['jd_asteroid'],
               'tof_outbound_days':candidate['tof_out_days'],
               'outbound_branch':candidate.get('outbound_branch',0)}
index=candidate['catalogue_index']
prefix='/tmp/ga20_'+{125:'fg3',66:'cc',40:'itokawa',50:'didymos'}[index]+'_complete'
if 'rp4r' in input_path.name:
    prefix='/tmp/ga20_rp4r_'+{125:'fg3',66:'cc',40:'itokawa',50:'didymos'}[index]+'_extended_plot'
spice.furnsh(pk.udpla.de440s.kernel_file())
raw=json.loads((d.FOLDER/'asteroids_sbdb_all_i5.json').read_text())
rec=dict(zip(raw['fields'],raw['data'][index]))
physical=json.loads((d.FOLDER/'asteroids_sbdb_physical.json').read_text())['asteroids'][str(rec['spkid'])]
data=np.loadtxt(d.FOLDER.parent/'Clustering/Clusters/CHAMPION_GA_16_plots_vinf_time_K780/cluster_velocity_statistics.txt')
lookup={int(r[0]):r for r in data}
dep=lookup[candidate['departure_cluster_id']];cap=lookup[candidate['capture_cluster_id']]
mu_ast=6.67430e-20*physical['mass_kg'];rpmin=candidate.get('min_periapsis_km',physical['radius_km']+50.)
E_SOI=(5.974e24/1.989e30)**.4*1.496e8

def boundary(jd,row,capture=False):
    r,v=d.earth(jd)
    theta=d.moon_angles([jd+row[7] if capture else jd-row[7]])
    mr=row[10:12].copy()
    if any(candidate.get(k,False) for k in ('normalize_soi_positions','normalized_soi_positions','positions_normalized_to_soi','surface_normalized','soi_positions_normalized')):
        mr*=E_SOI/np.linalg.norm(mr)
    r[:2]+=d.rotate(mr[None,:]*([1.,-1.] if capture else [1.,1.]),theta)[0]
    v[:2]+=d.rotate(row[None,2:4]*([-1.,1.] if capture else [1.,1.]),theta)[0]
    return r,v

jd0=candidate['jd_departure'];t1=candidate['tof_outbound_days'];t2=candidate['tof_return_days']
jda=jd0+t1;jdc=jda+t2
r0,vc0=boundary(jd0,dep);ra,va=d.asteroid(jda,rec);rc,vcc=boundary(jdc,cap,True)
la=pk.lambert_problem(r0,ra,t1*d.DAY,d.MU,False,2)
lr=pk.lambert_problem(ra,rc,t2*d.DAY,d.MU,False,int(np.ceil(candidate['return_branch']/2)))
v0=np.asarray(la.v0[candidate['outbound_branch']]);vm=np.asarray(la.v1[candidate['outbound_branch']])
vp=np.asarray(lr.v0[candidate['return_branch']]);vc=np.asarray(lr.v1[candidate['return_branch']])
vin=vm-va;vout=vp-va
sm,sp=np.linalg.norm(vin),np.linalg.norm(vout)
angle=np.arctan2(np.linalg.norm(np.cross(vin,vout)),np.dot(vin,vout))
max_angle=2*np.arcsin(mu_ast/(mu_ast+rpmin*sm**2))
rp_effective=mu_ast/sm**2*(1/np.sin(angle/2)-1)
radius0=2.5*np.linalg.norm(dep[4:6]);radiusc=2.5*np.linalg.norm(cap[4:6])
grid1=np.linspace(0.,t1*d.DAY,int(np.ceil(t1/.05))+1)
grid2=np.linspace(0.,t2*d.DAY,int(np.ceil(t2/.05))+1)
states1=np.asarray(pk.propagate_lagrangian_grid([r0,v0],grid1,d.MU))
states2=np.asarray(pk.propagate_lagrangian_grid([ra,vp],grid2,d.MU))
times=np.r_[grid1/d.DAY,t1+grid2[1:]/d.DAY]
positions=np.vstack([states1[:,0,:],states2[1:,0,:]])
earths=np.array([d.earth(jd0+t)[0] for t in times])
dist=np.linalg.norm(positions-earths,axis=1)

def state_at(t):
    if t<=t1:return tuple(map(np.asarray,pk.propagate_lagrangian([r0,v0],t*d.DAY,d.MU)))
    return tuple(map(np.asarray,pk.propagate_lagrangian([ra,vp],(t-t1)*d.DAY,d.MU)))

def earthdistance(t):return float(np.linalg.norm(state_at(t)[0]-d.earth(jd0+t)[0]))

minima=[]
for i in np.flatnonzero((dist[1:-1]<dist[:-2])&(dist[1:-1]<dist[2:]))+1:
    result=minimize_scalar(earthdistance,bounds=(times[i-1],times[i+1]),method='bounded',options={'xatol':1e-9})
    minima.append(dict(tof_days=float(result.x),distance_km=float(result.fun)))
crossings=[]
for i in np.flatnonzero((dist[:-1]-E_SOI)*(dist[1:]-E_SOI)<0):
    t=brentq(lambda x:earthdistance(x)-E_SOI,times[i],times[i+1],xtol=1e-10)
    rv=state_at(t);re,ve=d.earth(jd0+t)
    rate=float(np.dot(rv[0]-re,rv[1]-ve)/E_SOI)
    crossings.append(dict(tof_days=float(t),jd_tdb=float(jd0+t),direction='exit' if rate>0 else 'entry',radial_velocity_km_s=rate))

report=dict(candidate=candidate,verification='independent reconstruction from original catalogue and statistics',
            outbound_lambert_branch_count=len(la.v0),return_lambert_branch_count=len(lr.v0),
            departure_velocity_residual_km_s=float(np.linalg.norm(v0-vc0)),departure_velocity_radius_km_s=float(radius0),
            departure_disk_ratio=float(np.linalg.norm(v0-vc0)/radius0),
            capture_velocity_residual_km_s=float(np.linalg.norm(vc-vcc)),capture_velocity_radius_km_s=float(radiusc),
            capture_disk_ratio=float(np.linalg.norm(vc-vcc)/radiusc),
            speed_residual_km_s=float(abs(sm-sp)),turn_angle_rad=float(angle),max_turn_angle_rad=float(max_angle),
            implied_pericentre_km=float(rp_effective),minimum_pericentre_km=float(rpmin),
            outbound_position_closure_km=float(np.linalg.norm(states1[-1,0,:]-ra)),
            return_position_closure_km=float(np.linalg.norm(states2[-1,0,:]-rc)),
            earth_soi_km=float(E_SOI),departure_earth_distance_km=float(dist[0]),capture_earth_distance_km=float(dist[-1]),
            interior_local_minima=minima,soi_crossings=crossings,
            max_sampling_interval_days=float(np.max(np.diff(times))),
            mass_kg=float(physical['mass_kg']),mass_source=physical.get('mass_source_url',physical.get('mass_source')))
Path(prefix+'_validation.json').write_text(json.dumps(report,indent=2))

fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
ax=axs[0];au=pk.AU/1000
ax.plot(earths[:,0]/au,earths[:,1]/au,color='tab:red',lw=1,label='Terra durante la missione')
revout=int(np.ceil(candidate['outbound_branch']/2));revback=int(np.ceil(candidate['return_branch']/2))
ax.plot(states1[:,0,0]/au,states1[:,0,1]/au,color='tab:blue',lw=1.6,label=f'Andata: {t1:.3f} giorni, {revout} rivoluzioni')
ax.plot(states2[:,0,0]/au,states2[:,0,1]/au,'--',color='tab:green',lw=2,label=f'Ritorno: {t2:.3f} giorni, {revback} rivoluzioni')
period=2*np.pi*np.sqrt((float(rec['a'])*au)**3/d.MU)/d.DAY
ar=np.array([d.asteroid(jda+t,rec)[0] for t in np.linspace(0,period,400)])
ax.plot(ar[:,0]/au,ar[:,1]/au,color='tab:purple',alpha=.5,lw=1,label='Orbita planare asteroide')
ax.scatter(0,0,color='orange',s=100,label='Sole',zorder=6)
for pos,mark,col,label in [(r0,'o','tab:blue','Partenza SOI'),(ra,'*','black','Flyby asteroide'),(rc,'^','tab:green','Arrivo SOI')]:
    ax.scatter(pos[0]/au,pos[1]/au,marker=mark,color=col,s=90,label=label,zorder=7)
ax.set(xlabel='x eliocentrica [AU]',ylabel='y eliocentrica [AU]',title='Missione nel modello planare di GA20',aspect='equal')
ax.grid(alpha=.25);ax.legend(fontsize=8,loc='lower left')
ax=axs[1];ax.semilogy(times,dist,label='Distanza sonda–Terra',color='tab:blue')
ax.axhline(E_SOI,color='tab:red',ls='--',label='SOI terrestre')
ax.axvline(t1,color='black',ls=':',label='Flyby asteroide')
ax.set(xlabel='Giorni dalla partenza',ylabel='Distanza geocentrica [km]',title='Controllo dei ritorni intermedi')
ax.legend(fontsize=9);ax.grid(alpha=.25)
fig.suptitle(f'{rec["full_name"].strip()} — indice {index} — partenza JD {jd0:.6f} TDB\nEpoche continue; {revout} rivoluzioni all’andata, {revback} al ritorno; rp minimo {rpmin:g} km')
fig.savefig(prefix+'_transfer.png',dpi=180)
print(json.dumps(report,indent=2))
print('PLOT',prefix+'_transfer.png')

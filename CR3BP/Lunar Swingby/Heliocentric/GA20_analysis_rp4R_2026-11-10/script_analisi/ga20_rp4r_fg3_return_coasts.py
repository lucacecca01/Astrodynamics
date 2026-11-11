import sys,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/tmp');import ga20_rp4r_fg3_utils as s
start=time.perf_counter();source=Path('/tmp/ga20_rp4r_fg3_outbound.json');data=json.loads(source.read_text());matches=data['outbound_matches'];grid=np.arange(200.,500.,1.)
# Numerical screening of the physical unpowered departure cone at its two
# endpoints and its center. This is not a proof over continuous grid epochs.
checks=[];near=[];propagations=0
all_epochs=np.unique(np.array([z['jd_asteroid'] for z in matches])[:,None]+grid) if matches else np.array([])
earth=s.core.earth_state(all_epochs)[0] if len(all_epochs) else np.empty((0,3))
for i,z in enumerate(matches):
 jd0=z['jd_departure'];jda=z['jd_asteroid'];c=s.id_to_row[z['departure_cluster_id']];ra,va=s.ast_state(jda);vi=np.array(z['vinf_minus']);sm=np.linalg.norm(vi);dmax=2*np.arcsin(s.MU_AST/(s.MU_AST+s.RP*sm*sm));er=earth[np.searchsorted(all_epochs,jda+grid)]
 for sign in (-1.,0.,1.):
  vout=va.copy();vout[:2]+=s.core.rotate_xy(vi[:2],sign*dmax)
  pp=np.array(s.pk.propagate_lagrangian_grid(rv=[ra,vout],tofs=np.r_[0.,grid]*s.DAY,mu=s.MU))[1:,0,:];propagations+=1
  dist=np.linalg.norm(pp-er,axis=1);picks=np.flatnonzero((dist[1:-1]<dist[:-2])&(dist[1:-1]<=dist[2:]))+1;picks=np.unique(np.r_[picks,0,len(grid)-1])
  best=(float(dist.min()),float(grid[np.argmin(dist)]))
  for q in picks:
   if q in (0,len(grid)-1):continue
   def d2(t):
    p=s.pk.propagate_lagrangian([ra,vout],t*s.DAY,s.MU)[0];e=s.core.earth_state(jda+t)[0];return float(np.sum((np.array(p)-e)**2))
   op=minimize_scalar(d2,bounds=(grid[q-1],grid[q+1]),method='bounded',options={'xatol':1e-7})
   if np.sqrt(op.fun)<best[0]:best=(float(np.sqrt(op.fun)),float(op.x))
  zz=dict(outbound_index=i,jd_departure=jd0,jd_asteroid=jda,departure_cluster_id=z['departure_cluster_id'],outbound_branch=z['outbound_branch'],flyby_turn_fraction=sign,max_turn_rad=float(dmax),minimum_earth_distance_km=best[0],return_days_at_minimum=best[1]);checks.append(zz)
  if best[0]<10*s.RSOI:near.append(zz)
checks.sort(key=lambda z:z['minimum_earth_distance_km']);out=dict(body='175706 (1996 FG3)',source_outbound=str(source),mass_kg=s.MASS,radius_km=s.R_AST,min_periapsis_km=s.RP,return_grid_days=[200,499],grid_step_days=1.,outbound_matches=len(matches),physical_coasts_propagated=propagations,turn_fractions=[-1,0,1],coasts_within_10soi=len(near),best_coasts=checks[:30],all_close_coasts=near,elapsed_seconds=time.perf_counter()-start,limitations='Fixed daily outbound epochs; three allowable flyby angles sampled. Local minima refined in time. This does not exclude continuous outbound dates or prove all intermediate flyby angles, although their deflections are tiny.')
Path('/tmp/ga20_rp4r_fg3_return_coasts.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

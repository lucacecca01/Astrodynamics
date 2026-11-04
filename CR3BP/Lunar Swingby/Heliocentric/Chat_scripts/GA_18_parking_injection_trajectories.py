"""Ricolora le 56 tavole 05–12 con il DV circolare -> periapside osculatore.

Le selezioni, i ranghi, le epoche e gli archi sono quelli gia salvati. Non
esegue integrazioni, clustering o Lambert. I primi tre disegni generano i
livelli grafici condivisi dalle selezioni individuali di tutte le mappe.
Con --from-leo-altitude-km aggiunge i due impulsi di Hohmann dalla quota data.
"""

import argparse
import gc
import hashlib
import inspect
import json
from multiprocessing import get_context
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages

import GA_18_plots as plots
import GA_18_catalog_epoch_trajectories as catalog_driver
import GA_18_phi_trajectories as phi_driver
from GA_18_combined_trajectories import draw_combined


BASE = plots.OUTPUT_DIR
SOURCES = dict(catalog=BASE/'traiettorie_05_10_J2000_catalogo_100x100', phi=BASE/'traiettorie_phi')
FOOTER = ('ΔV al periapside dell’orbita osculatrice; circolare complanare e stesso verso. '
          'Costo impulsivo analitico geocentrico, non ΔV di rendez-vous.')
_STATE = None


def parking_delta_v(eps, ecc):
    """Stable elliptic same-direction tangential impulse at osculating rp."""
    eps, ecc = np.broadcast_arrays(np.asarray(eps, dtype=float), np.asarray(ecc, dtype=float))
    if not (np.isfinite(eps).all() and np.isfinite(ecc).all()
            and np.all(eps < 0) and np.all((ecc >= 0) & (ecc < 1))):
        raise ValueError('Servono orbite osculatrici legate con 0 <= e < 1.')
    rp = -plots.MU_EARTH*(1-ecc)/(2*eps)
    if not np.isfinite(rp).all() or np.any(rp <= plots.R_EARTH_KM):
        raise ValueError('Periapside osculatore sotto la superficie terrestre.')
    circular = np.sqrt(plots.MU_EARTH/rp)
    dv = circular*ecc/(np.sqrt(1+ecc)+1)
    return rp, dv


def hohmann_delta_v(rp, altitude_km):
    """Moduli dei due impulsi fra circolari, anche per un trasferimento in discesa."""
    rp = np.asarray(rp, dtype=float)
    if (not np.isfinite(altitude_km) or altitude_km <= 0
            or not np.isfinite(rp).all() or np.any(rp <= plots.R_EARTH_KM)):
        raise ValueError('Quota LEO e raggi delle circolari devono essere sopra la superficie.')
    r0 = plots.R_EARTH_KM + altitude_km
    ratio = np.abs(rp-r0)/(rp+r0)
    # Forme razionalizzate: evitano sottrazioni fra velocita quasi uguali.
    departure = np.sqrt(plots.MU_EARTH/r0)*ratio/(np.sqrt(2*rp/(r0+rp))+1)
    arrival = np.sqrt(plots.MU_EARTH/rp)*ratio/(np.sqrt(2*r0/(r0+rp))+1)
    return departure, arrival


def _digest_arrays(settings, arrays):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare(output, from_leo_altitude_km=None):
    pools, jobs = {}, []
    radius, maximum, minimum = 0., 0., np.inf
    for name, loader in (('catalog', catalog_driver.load_render_inputs), ('phi', phi_driver.load_render_inputs)):
        loaded = loader(SOURCES[name])
        curves, members, indices, winners, counts, scores, grids, summary = loaded
        rp, dv = parking_delta_v(members[:, 9], members[:, 10])
        components, origin_data = {}, {}
        if from_leo_altitude_km is not None:
            injection = dv
            departure, arrival = hohmann_delta_v(rp, from_leo_altitude_km)
            hohmann = departure+arrival
            dv = hohmann+injection
            components = dict(delta_v_injection_km_s=injection,
                              delta_v_hohmann_departure_km_s=departure,
                              delta_v_hohmann_arrival_km_s=arrival,
                              delta_v_hohmann_km_s=hohmann, delta_v_total_km_s=dv)
            origin_data = dict(leo_altitude_km=from_leo_altitude_km,
                               leo_radius_km=plots.R_EARTH_KM+from_leo_altitude_km)
        ra = -plots.MU_EARTH*(1+members[:, 10])/(2*members[:, 9])
        event_radius = np.linalg.norm(curves[:, :, 0], axis=1)
        event_is_periapsis = np.abs(event_radius-rp) <= np.abs(event_radius-ra)
        data_file = output/f'delta_v_{name}_members.npz'
        np.savez_compressed(data_file, source_ids=members[:, 0].astype(np.int64),
                            cluster_ids=members[:, 1].astype(np.int64), eps_perigee=members[:, 9],
                            eccentricity=members[:, 10], rp_km=rp, ra_km=ra, delta_v_km_s=dv,
                            event_radius_km=event_radius, event_is_periapsis=event_is_periapsis,
                            circular_speed_km_s=np.sqrt(plots.MU_EARTH/rp), mu_earth_km3_s2=plots.MU_EARTH,
                            **components, **origin_data)
        for start in range(0, len(curves), 2048):
            radius = max(radius, float(np.linalg.norm(curves[start:start+2048], axis=1).max()))
        maximum = max(maximum, float(dv.max()))
        minimum = min(minimum, float(dv.min()))
        pool = dict(name=name, curves=curves, members=members, indices=indices,
                    winners=winners, counts=counts, scores=scores, grids=grids, summary=summary,
                    phases=np.array([w['phase_etd_deg'] for w in summary['winners']]),
                    dv=dv, rp=rp, peri=event_is_periapsis, data_file=str(data_file),
                    components=components, origin_data=origin_data)
        pools[name] = pool
        columns = {g['key']: i for i, g in enumerate(grids)}
        for record in summary['figures']:
            source_figure = Path(record['file'])
            source_selection = source_figure.with_name(source_figure.name.replace('_tre_riferimenti.png', '_selezione.npz'))
            job = dict(record, pool=name, column=columns[record['map']],
                       source_figure=str(source_figure), source_selection=str(source_selection))
            job['id'] = f"{record['map']}/{record['criterion']}/P{record['percentile']:g}"
            jobs.append(job)
        print(f'{name}: {len(members):,} membri; DV {dv.min():.9g}–{dv.max():.9g} km/s; '
              f'{np.count_nonzero(~event_is_periapsis)} eventi apocentrici osculatori.', flush=True)
    cat, phi = pools['catalog'], pools['phi']
    np.testing.assert_array_equal(cat['phases'], phi['phases'])
    for a, b in zip(cat['summary']['winners'], phi['summary']['winners']):
        if any(a[k] != b[k] for k in ('spkid', 'cluster_id', 'jd_etd', 'jd_arrival')):
            raise ValueError('Vincitori diversi fra i due prodotti originali.')
    mapping = np.full(len(phi['members']), -1, dtype=np.int32)
    eligible = np.unique(phi['indices'][phi['scores'] >= min(phi['summary']['individual_thresholds'])])
    cat_rows = np.searchsorted(cat['members'][:, 0], phi['members'][eligible, 0])
    if np.any(cat_rows >= len(cat['members'])):
        raise ValueError('Membro individuale phi assente dal pool comune del catalogo.')
    np.testing.assert_array_equal(cat['members'][cat_rows], phi['members'][eligible])
    np.testing.assert_array_equal(cat['dv'][cat_rows], phi['dv'][eligible])
    for start in range(0, len(eligible), 2048):
        np.testing.assert_array_equal(cat['curves'][cat_rows[start:start+2048]], phi['curves'][eligible[start:start+2048]])
    mapping[eligible] = cat_rows.astype(np.int32)
    phi['individual_to_catalog'] = mapping
    # Every individual family must share the exact same ordered instances.
    for threshold in cat['summary']['individual_thresholds']:
        a = np.flatnonzero(cat['scores'] >= threshold)
        b = np.flatnonzero(phi['scores'] >= threshold)
        np.testing.assert_array_equal(cat['members'][cat['indices'][a]], phi['members'][phi['indices'][b]])
        np.testing.assert_array_equal(cat['winners'][a], phi['winners'][b])
        np.testing.assert_array_equal(cat['scores'][a], phi['scores'][b])
    jobs.sort(key=lambda j: (j['map'], j['criterion'] != 'celle', j['percentile']))
    if len(jobs) != 56 or len({j['id'] for j in jobs}) != 56:
        raise ValueError('Attese tutte le 56 selezioni delle mappe 05–12.')
    apo_sources = np.unique(np.concatenate([p['members'][~p['peri'], 0] for p in pools.values()])).astype(np.int64)
    metadata = dict(jobs=jobs, source_roots={k: str(v) for k, v in SOURCES.items()},
                    delta_v_definition='sqrt(mu/rp) * e / (sqrt(1+e)+1); rp = -mu*(1-e)/(2*eps)',
                    injection_reference=FOOTER, mu_earth_km3_s2=plots.MU_EARTH,
                    norm_limits_km_s=[0., maximum], trajectory_cmap='viridis_r', grid_cmap='viridis',
                    fixed_alpha=.03, ordering='Original occurrence/cluster-frequency scores, unchanged.',
                    global_limits_km=[[-radius, -radius], [radius, radius]],
                    osculating_apocenter_source_ids=apo_sources.tolist(),
                    pool_data={k: dict(file=p['data_file'], members=len(p['members']),
                                      provenance=p['summary']['curve_provenance']) for k, p in pools.items()},
                    renderer_sha256=hashlib.sha256(inspect.getsource(draw_combined).encode()).hexdigest())
    if from_leo_altitude_km is not None:
        altitude = f'{from_leo_altitude_km:g}'
        metadata.update(
            cost_mode='leo_hohmann_parking_injection', leo_altitude_km=from_leo_altitude_km,
            leo_radius_km=plots.R_EARTH_KM+from_leo_altitude_km,
            norm_limits_km_s=[minimum, maximum],
            delta_v_definition='abs(Hohmann departure) + abs(Hohmann arrival/circularization) + osculating pericenter injection.',
            injection_reference=(
                f'ΔV totale: Hohmann dalla circolare a quota {altitude} km alla circolare r_p + immissione al periapside osculatore.\n'
                'Somma dei moduli dei tre impulsi; manovre complanari e stesso verso. Arco Hohmann non raffigurato.'),
            color_description=f'ΔV da LEO {altitude} km: Hohmann + immissione',
            colorbar_label=r'$\Delta V_{\mathrm{Hohmann}}+\Delta V_{\mathrm{immissione}}$ [km/s]',
            pdf_filename=f'delta_v_leo{altitude}_hohmann_tutte_mappe_05_12.pdf',
            hohmann_burns='Due impulsi: la circolare intermedia viene raggiunta prima dell’immissione, senza compensare impulsi opposti.',
            geometry_and_epochs='Archi, ETD, fasi e selezioni originali; Hohmann aggiunto solo al costo impulsivo.')
    _atomic_json(output/'preparazione.json', metadata)
    return pools, jobs, metadata


def _job_inputs(job, pools):
    pool = pools[job['pool']]
    column = job['column']
    ranking = pool['counts'][:, column] if job['criterion'] == 'celle' else pool['scores']
    take = np.flatnonzero(ranking >= job['threshold'])
    ids = pool['indices'][take].astype(np.int32)
    winners = pool['winners'][take]
    members = pool['members'][ids]
    original = ranking[take]
    phases = pool['phases'][winners]
    with np.load(job['source_selection']) as saved:
        for name, value in (('source_ids', members[:, 0]), ('cluster_ids', members[:, 1]),
                            ('winner_indices', winners), ('phase_etd_deg', phases), ('scores', original)):
            np.testing.assert_array_equal(saved[name], value)
        if float(saved['percentile']) != job['percentile'] or float(saved['threshold']) != job['threshold']:
            raise ValueError('Percentile o soglia diversi dalla selezione originale.')
    if (len(take) != job['instances'] or len(np.unique(ids)) != job['trajectories']
            or len(np.unique(winners)) != job['asteroids']):
        raise ValueError('Numero di istanze/membri/asteroidi diverso dall’originale.')
    render_pool = pool
    if job['criterion'] == 'individuali':
        render_pool = pools['catalog']
        if job['pool'] == 'phi':
            ids = pool['individual_to_catalog'][ids].astype(np.int32)
            if np.any(ids < 0):
                raise ValueError('Membro individuale non associato al pool comune.')
        np.testing.assert_array_equal(render_pool['members'][ids], members)
    return pool, render_pool, ids, winners, members, original, phases


def render_job(job):
    pools, metadata, output, resume = _STATE
    pool, render_pool, ids, winners, members, ordering, phases = _job_inputs(job, pools)
    dv = render_pool['dv'][ids]
    rp = render_pool['rp'][ids]
    peri = render_pool['peri'][ids]
    apo_count = int(np.count_nonzero(~render_pool['peri'][np.unique(ids)]))
    tag = f"P{job['percentile']:g}".replace('.', '_')
    folder = output/job['criterion']/job['map']
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder/f'{tag}_tre_riferimenti.png'
    selection_path = folder/f'{tag}_selezione.npz'
    grid = dict(pool['grids'][job['column']])
    grid['normalization'] = Normalize(0, int(grid['occurrences'].max()))
    grid['cmap'] = 'viridis'
    if job['criterion'] == 'celle':
        grid['threshold'] = job['threshold']
        grid['selection_title'] = 'Celle sopra soglia'
        grid['selection_caption'] = 'Colore: celle selezionate · grigio: altre celle occupate'
        criterion = f"P{job['percentile']:g} delle celle — almeno {int(np.ceil(job['threshold']))} asteroidi/cella"
    else:
        level = pool['summary']['individual_percentiles'].index(job['percentile'])
        grid['selected_occurrences'] = grid['individual_counts'][level]
        grid['selection_title'] = 'Celle con traiettorie selezionate'
        grid['selection_caption'] = 'Colore: copertura dei membri selezionati nelle rispettive epoche'
        criterion = (f"P{job['percentile']:g} delle singole traiettorie — "
                     f"almeno {int(np.ceil(job['threshold']))} asteroidi per cluster")
    footer = metadata.get('injection_reference', FOOTER)
    if apo_count:
        footer += (f' In questa selezione {apo_count} eventi salvati sono apocentrici osculatori: '
                   + ('l’immissione si riferisce al periapside osculatore e non a quei pallini.'
                      if 'leo_radius_km' in metadata else
                      'il ΔV si riferisce al periapside osculatore e non a quei pallini.'))
    settings = dict(version='parking-injection-v1', renderer=metadata['renderer_sha256'], job=job['id'],
                    norm=metadata['norm_limits_km_s'], limits=metadata['global_limits_km'],
                    source_provenance=render_pool['summary']['curve_provenance']['signature'],
                    cmap='viridis_r', alpha=.03, grid_cmap='viridis', footer=footer,
                    threshold=job['threshold'], percentile=job['percentile'])
    if 'leo_radius_km' in metadata:
        settings.update(version='parking-injection-hohmann-v1', cost_mode=metadata['cost_mode'],
                        leo_radius_km=metadata['leo_radius_km'])
    signature = _digest_arrays(settings, (members, winners, phases, ordering, dv,
        grid['occurrences'], grid['edges_x'], grid['edges_y'],
        grid.get('selected_occurrences', np.empty(0))))
    reusable = False
    if resume and image_path.is_file() and selection_path.is_file():
        try:
            with np.load(selection_path) as previous:
                reusable = str(previous['render_signature']) == signature
            if reusable:
                with Image.open(image_path) as image:
                    reusable = image.size == (3600, 2340)
                    image.verify()
        except (OSError, ValueError, KeyError):
            reusable = False
    if not reusable:
        color_description = metadata.get('color_description', 'ΔV da circolare al periapside osculatore')
        title = (f"Mappa {job['map'][:2]}: {criterion}\n"
                 f"{job['trajectories']:,} traiettorie distinte · {job['instances']:,} istanze a ETD diverse · "
                 f"{job['asteroids']} asteroidi · colore: {color_description}")
        print(f"Disegno {job['id']}: {len(dv):,} istanze; DV {dv.min():.6g}–{dv.max():.6g} km/s.", flush=True)
        draw_combined(render_pool['curves'], members, dv,
                      metadata.get('colorbar_label', r'$\Delta V_{\mathrm{parcheggio}}$ al periapside osculatore [km/s]'), image_path,
                      pool['summary']['population_count'], title=title,
                      normalization=Normalize(*metadata['norm_limits_km_s']), fixed_alpha=.03,
                      global_limits=np.array(metadata['global_limits_km']), raster_chunks=True,
                      perigee_markers=True, selection_grid=grid, phase_etd_deg=phases,
                      curve_indices=ids, third_frame='catalog', trajectory_cmap='viridis_r',
                      ordering_scores=ordering, footer_note=footer,
                      trajectory_layer_dir=(output/'livelli_traiettorie' if job['criterion'] == 'individuali' else None),
                      trajectory_layer_token=json.dumps(render_pool['summary']['curve_provenance']['signature'], sort_keys=True))
        np.savez_compressed(selection_path, source_ids=members[:, 0].astype(np.int64),
                            cluster_ids=members[:, 1].astype(np.int64), winner_indices=winners,
                            phase_etd_deg=phases, scores=ordering, delta_v_km_s=dv, rp_km=rp,
                            event_is_periapsis=peri, threshold=job['threshold'], percentile=job['percentile'],
                            render_signature=signature,
                            **{name: values[ids] for name, values in render_pool['components'].items()},
                            **render_pool['origin_data'])
    else:
        print(f"Tavola verificata e riusata: {job['id']}.", flush=True)
    record = dict(job, file=str(image_path), selection_file=str(selection_path), render_signature=signature,
                  dv_min_km_s=float(dv.min()), dv_max_km_s=float(dv.max()),
                  osculating_apocenter_events=apo_count, reused=reusable)
    gc.collect()
    return record


def package_pdf(paths, output):
    """Lossless PNG pages loaded one at a time; no all-pages RGB allocation."""
    with PdfPages(output) as pdf:
        for path in paths:
            with Image.open(path) as source:
                rgb = np.array(source.convert('RGB'))
            height, width = rgb.shape[:2]
            fig = plt.figure(figsize=(width/180, height/180))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(rgb, aspect='auto', interpolation='none')
            ax.set_axis_off()
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
            del rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--from-leo-altitude-km', type=float, help='Aggiunge il Hohmann dalla circolare alla quota indicata [km].')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--per-map-pdf', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers deve essere positivo.')
    if args.from_leo_altitude_km is not None and (not np.isfinite(args.from_leo_altitude_km) or args.from_leo_altitude_km <= 0):
        parser.error('La quota della circolare di partenza deve essere finita e positiva.')
    if args.output is None:
        args.output = (BASE/'traiettorie_delta_v_parcheggio' if args.from_leo_altitude_km is None else
                       BASE/f'traiettorie_delta_v_leo{args.from_leo_altitude_km:g}_hohmann_parcheggio')
    args.output.mkdir(parents=True, exist_ok=True)
    pools, jobs, metadata = prepare(args.output, args.from_leo_altitude_km)
    if args.prepare_only:
        return
    global _STATE
    _STATE = pools, metadata, args.output, args.resume
    completed = {}

    def record(result):
        completed[result['id']] = result
        _atomic_json(args.output/'avanzamento.json', dict(
            completed=len(completed), total=len(jobs),
            figures=[completed[j['id']] for j in jobs if j['id'] in completed]))
        print(f"Completate {len(completed)}/{len(jobs)} tavole: {result['file']}", flush=True)

    # One draw per individual threshold primes the exact shared trajectory layers.
    primers = [next(j for j in jobs if j['map'] == key and j['criterion'] == 'individuali'
                    and j['percentile'] == percentile)
               for key, percentile in (('11_eps_phi', 99.99), ('05_eps_e', 99.9), ('05_eps_e', 99.95))]
    if args.from_leo_altitude_km is not None and args.workers > 1:
        with get_context('fork').Pool(min(args.workers, len(primers))) as pool:
            for result in pool.imap_unordered(render_job, primers, chunksize=1):
                record(result)
    else:
        for job in primers:
            record(render_job(job))
    remaining = [j for j in jobs if j['id'] not in completed]
    if args.from_leo_altitude_km is not None:
        remaining.sort(key=lambda j: j['instances'] if j['criterion'] == 'celle' else 0, reverse=True)
    if args.workers > 1:
        with get_context('fork').Pool(args.workers) as pool:
            for result in pool.imap_unordered(render_job, remaining, chunksize=1):
                record(result)
    else:
        for job in remaining:
            record(render_job(job))
    figures = [completed[j['id']] for j in jobs]
    pdf = args.output/metadata.get('pdf_filename', 'delta_v_parcheggio_tutte_mappe_05_12.pdf')
    package_pdf([r['file'] for r in figures], pdf)
    per_map = {}
    if args.per_map_pdf:
        folder = args.output/'pdf_per_mappa'
        folder.mkdir(exist_ok=True)
        for key in sorted({j['map'] for j in jobs}):
            path = folder/f'{key}.pdf'
            package_pdf([r['file'] for r in figures if r['map'] == key], path)
            per_map[key] = str(path)
    metadata.update(figures=figures, pdf=str(pdf), per_map_pdf=per_map,
                    trajectory_panels=['Sinodico Terra-Luna', 'Geocentrico ECLIPJ2000',
                                       'Geocentrico del catalogo, Luna su +x a ETD'])
    _atomic_json(args.output/'riepilogo.json', metadata)
    print(f'Completate e raccolte tutte le {len(figures)} tavole: {pdf}', flush=True)


if __name__ == '__main__':
    main()

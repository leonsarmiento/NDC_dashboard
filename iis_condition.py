#!/usr/bin/env python3
"""IIS Land Condition engine - query layer over the enginev2 precompute
bundle, packaged for the NDC Dashboard ("Land Condition (IIS)" page).

The Bundle class is ported verbatim from the desktop engine GUI
(enginev2/code/engine_v2_gui_plotsBeta.py, 2026-11-05) including the
"bin 0 (no lens data)" invariant: degraded/gap totals are identical with
any lens, another lens, or no lens. Data ships as
data/iis_condition_bundle.tar.gz; the T7 Shield precompute dir is used
directly when it is mounted (local dev).
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
BUNDLE_ARCHIVE = APP_DIR / 'data' / 'iis_condition_bundle.tar.gz'
LOCAL_CANDIDATES = [
    Path('/Volumes/T7 Shield/GIS_DATA/iis/nature/geodata/enginev2/outputs/precompute'),
]


def resolve_bundle_dir() -> str:
    """Local precompute dir if mounted, else extract the shipped archive."""
    for p in LOCAL_CANDIDATES:
        if (p / 'registry.json').exists():
            return str(p)
    marker = Path(tempfile.gettempdir()) / 'ndc_iis_bundle_extracted'
    if not (marker / 'registry.json').exists():
        marker.mkdir(parents=True, exist_ok=True)
        with tarfile.open(BUNDLE_ARCHIVE, 'r:gz') as t:
            t.extractall(marker, filter='data')
    return str(marker)



LENS_NAMES = {
    'lens_10': 'Government effectiveness',
    'lens_11': 'eHANPP - human appropriation of NPP',
    'lens_112': 'Subnational poverty, MPI 10 quantiles',
    'lens_12': 'Poverty, national MPI',
    'lens_13': 'Employment in Agriculture %',
    'lens_14': 'Tenure insecurity',
    'lens_15': 'Tenure security - communal lands',
    'lens_16': 'Tenure security - indigenous lands',
    'lens_17': 'Voice & accountability',
    'lens_18': '% Ind. & Comm. lands Not acknow. by gov.',
    'lens_21': 'Farm size',
    'lens_22': 'Protected areas, IUCN categories',
    'lens_23': 'Communal Lands acknown by gov',
    'lens_24': 'Indigenous people lands acknown and not by gov.',
    'lens_31': 'Water stress, Aqueduct BWS',
    'lens_32': 'Riverine flood risk, Aqueduct RFR',
    'lens_33': 'Drought risk, Aqueduct DRR',
    'lens_34': 'WRI forest-loss drivers 2001-2025',
    'lens_41': 'Biome, 14 classes',
    'lens_42': 'Ecoregion, 846 classes',
}


INDEX_NAMES = {
    'eii': 'EII (Ecological Integrity Index)',
    'func': 'Functional integrity',
    'struct': 'Structural integrity',
    'comp': 'Composition integrity',
    'bhi': 'Biodiversity Habitat Index',
    'beri': 'BERI',
    'msa_plants': 'MSA plants',
    'msa_wbvert': 'MSA warm-blooded vertebrates',
}


COVER_LABELS = {
    'croplands': 'Croplands', 'cultivatedgrasslands': 'Cultivated grasslands',
    'forests': 'Forests', 'naturalgrasslands': 'Natural grasslands',
    'shrublands': 'Shrublands', 'wetlands': 'Wetlands', 'deserts': 'Deserts',
}


FACTOR_LABELS = {
    'factor_10/sdg_indicator_15_3_1_degraded_binary':
        'SDG 15.3.1 degraded (binary)',
    'factor_10/sdg_indicator_15_3_1_status_classes':
        'SDG 15.3.1 status classes (categorical)',
    'factor_11/land_productivity_dynamics_degraded_binary':
        'Land productivity dynamics (binary)',
    'factor_12/land_cover_degradation_binary': 'Land cover degradation (binary)',
    'factor_13/soc_pct_change_2000_2023_degraded_le_-10_binary':
        'Soil organic carbon change ≤ -10% (binary)',
    'factor_31/ece_2015_2024_salinity_binary': 'Salinity ECE 2015-2024 (binary)',
    'factor_32/esp_2015_2024_salinity_binary': 'Salinity ESP 2015-2024 (binary)',
    'factor_33/degradation_process_f33_binary': 'Degradation process F33 (binary)',
    'factor_34/degradation_process_f34_binary': 'Degradation process F34 (binary)',
}


class Bundle:
    """Lazy loader / query layer over the precompute bundle. No tkinter."""

    def __init__(self, bundle=None):
        d = bundle or resolve_bundle_dir()
        self.dir = d
        self.reg = json.load(open(os.path.join(d, 'registry.json')))
        self.const = json.load(open(os.path.join(d, 'constancy.json')))
        cpath = os.path.join(d, 'countries.json')
        self.countries = (json.load(open(cpath)) if os.path.exists(cpath) else {})
        self.country_ids = self.const['country_ids']          # cidx 1..NC
        self.lens_names = sorted(self.reg['lenses'])          # totals lens axis
        self.factors = sorted(self.reg['factors'])
        self.covers = sorted(self.reg['covers'])
        self.indexes = sorted({k.split('|')[0] for k in self.reg['gaps']})
        self.cfgs = sorted({k.split('|')[-1] for k in self.reg['gaps']})
        self._state_cache, self._gap_cache, self._totals = {}, {}, None

    # -- basic views -------------------------------------------------------
    def lens_label(self, lname):
        if lname == 'lens_1':
            return 'Country (macro lens)'
        d = LENS_NAMES.get(lname)
        return f'{lname} ({d})' if d else lname

    def constant_lenses(self):
        return ['lens_1'] + [l for l in self.lens_names
                             if l != 'lens_1' and self.const[l].get('country_constant')]

    def spatial_lenses(self):
        return [l for l in self.lens_names
                if l != 'lens_1' and not self.const[l].get('country_constant')]

    def country_label(self, cidx):
        g = str(self.country_ids[cidx]) if 0 <= cidx < len(self.country_ids) else '?'
        c = self.countries.get(g, {})
        return f"{c.get('name', 'GAUL ' + g)} ({c.get('iso3', '?')})"

    def selected_country_idxs(self, gaules):
        pos = {str(g): i for i, g in enumerate(self.country_ids)}
        return [pos[str(g)] for g in gaules if str(g) in pos]

    def primary_bins(self, cidxs, plens):
        """{cidx: primary-bin value} — per-country dominant bin from constancy."""
        if plens == 'lens_1':
            return {c: c for c in cidxs}
        dom = self.const[plens]['dominant_bin']
        return {c: dom[c] for c in cidxs}

    def bin_label(self, lens, b):
        u = self.const.get(lens, {}).get('unique') or []
        return f'bin {u[b]}' if b < len(u) else f'bin {b}'

    # -- arrays ------------------------------------------------------------
    def state(self, factor, cover):
        fn = os.path.join(self.dir, f'state_{factor.replace("/", "__")}__{cover}.npz')
        if not os.path.exists(fn):
            return None
        key = (factor, cover)
        if key not in self._state_cache:
            z = np.load(fn, allow_pickle=False)
            self._state_cache[key] = ({k: z[k] for k in z.files if k != 'classes'},
                                      [str(x) for x in z['classes']])
            while len(self._state_cache) > 8:
                self._state_cache.pop(next(iter(self._state_cache)))
        return self._state_cache[key]

    def gap(self, index, cover, cfg):
        fn = os.path.join(self.dir, f'gap_{index}_{cover}_{cfg}.npz')
        if not os.path.exists(fn):
            return None
        key = (index, cover, cfg)
        if key not in self._gap_cache:
            z = np.load(fn, allow_pickle=False)
            self._gap_cache[key] = {k: z[k] for k in z.files}
            while len(self._gap_cache) > 8:
                self._gap_cache.pop(next(iter(self._gap_cache)))
        return self._gap_cache[key]

    def totals(self):
        if self._totals is None and os.path.exists(os.path.join(self.dir, 'totals.npz')):
            self._totals = np.load(os.path.join(self.dir, 'totals.npz'))
        return self._totals

    # -- shared helpers ----------------------------------------------------
    @staticmethod
    def _cell_parts(hists, lens, cidxs):
        """hist restricted to countries -> [c, bin] arrays."""
        h = hists[lens]
        return h[cidxs]

    def _pgroups(self, cidxs, pbinmap):
        if pbinmap is None:
            return [None]
        return sorted({pbinmap[c] for c in cidxs})

    # -- state-factor query --------------------------------------------------
    def query_state(self, factor, covers, cidxs, primary='lens_1', secondary=None,
                    box_class='1'):
        """rows: (cover, class, primary, secondary-bin, ha, pct)
        box:  (cover, group, bin, country, pct) — relative only.
        pct denominator = Σ assessed-class ha in the SAME (cover, primary, bin) cell."""
        rows, box = [], []
        for cov in covers:
            got = self.state(factor, cov)
            if got is None:
                continue
            hists, classes = got
            assessed = [i for i, c in enumerate(classes) if c != 'F_NODATA']
            pbinmap = self.primary_bins(cidxs, primary) if primary != 'lens_1' else None
            # aggregation lens = secondary if set, else primary if not macro, else lens_1
            alens = secondary or (primary if primary != 'lens_1' else 'lens_1')
            h = hists.get(alens)
            if h is None:
                continue
            hsub = h[cidxs]                                  # [c, bin, cls]
            ass_tot = hsub[:, :, assessed].sum(axis=2)       # [c, bin] denominator
            box_ci = classes.index(box_class) if box_class in classes else assessed[0]
            for pb in self._pgroups(cidxs, pbinmap):
                cc = [c for c in cidxs if pbinmap is None or pbinmap[c] == pb]
                plab = (self.lens_label(primary) + f' bin {pb}') if pb is not None \
                    else 'All countries'
                sub = hsub[[cidxs.index(c) for c in cc]]     # aligned to cc
                den = ass_tot[[cidxs.index(c) for c in cc]]              # [c, bin] kept 2-D
                if secondary is None and primary == 'lens_1':
                    # country totals: single aggregate cell
                    for ci, cls in enumerate(classes):
                        if cls == 'F_NODATA':
                            continue
                        ha = float(sub[:, :, ci].sum())
                        t = float(den.sum())
                        rows.append((COVER_LABELS.get(cov, cov), cls, plab,
                                     'country total', ha, 100.0 * ha / t if t else None))
                    for j, c in enumerate(cc):
                        hc = float(den[j].sum())
                        if hc > 0:
                            box.append((COVER_LABELS.get(cov, cov), plab, 'country',
                                        self.country_label(c),
                                        100.0 * float(sub[j, :, box_ci].sum()) / hc))
                    continue
                for b in range(sub.shape[1]):
                    t = float(den[:, b].sum())
                    if t <= 0:
                        continue
                    for ci, cls in enumerate(classes):
                        if cls == 'F_NODATA':
                            continue
                        ha = float(sub[:, b, ci].sum())
                        rows.append((COVER_LABELS.get(cov, cov), cls, plab,
                                     self.bin_label(alens, b),
                                     ha, 100.0 * ha / t))
                    for j, c in enumerate(cc):
                        hj = float(den[j, b])
                        if hj > 0:
                            box.append((COVER_LABELS.get(cov, cov), plab,
                                        self.bin_label(alens, b),
                                        self.country_label(c),
                                        100.0 * float(sub[j, b, box_ci]) / hj))
                # -- bin 0: raster value 99 = pixels with no lens data ------
                # macro (lens_1) minus the sum over lens bins, reported as
                # its own group so degraded totals stay invariant across
                # lens choices (with / without / different lens).
                h1 = hists.get('lens_1')
                if h1 is not None:
                    rem_cls = h1[cc].sum(axis=1) - sub.sum(axis=1)  # [c, cls]; cc = absolute rows
                    rem_t = float(rem_cls[:, assessed].sum())
                    if rem_t > 1e-6:
                        for ci, cls in enumerate(classes):
                            if cls == 'F_NODATA':
                                continue
                            hcl = float(rem_cls[:, ci].sum())
                            rows.append((COVER_LABELS.get(cov, cov), cls, plab,
                                         'bin 0 (no lens data)',
                                         hcl, 100.0 * hcl / rem_t))
                        for j in range(len(cc)):
                            hj = float(rem_cls[j, assessed].sum())
                            if hj > 0:
                                box.append((COVER_LABELS.get(cov, cov), plab,
                                            'bin 0 (no lens data)',
                                            self.country_label(cc[j]),
                                            100.0 * float(rem_cls[j, box_ci]) / hj))
        return {'rows': rows, 'box': box}

    # -- 2x gap query --------------------------------------------------------
    def query_gap(self, index, cfg, covers, cidxs, primary='lens_1', secondary=None):
        """rows: (cover, cfg, primary, secondary-bin, ha, pct vs cover total)."""
        rows, box = [], []
        tot_npz = self.totals()
        for cov in covers:
            g = self.gap(index, cov, cfg)
            if g is None:
                continue
            alens = secondary or (primary if primary != 'lens_1' else 'lens_1')
            gsub = g.get(alens)
            if gsub is None:
                continue
            gsub = gsub[cidxs]
            den = None
            if tot_npz is not None and cov in tot_npz.files and alens in self.lens_names:
                li = self.lens_names.index(alens)
                den = tot_npz[cov][li][:, :gsub.shape[1]][cidxs]     # [c, bin] cover ha
            pbinmap = self.primary_bins(cidxs, primary) if primary != 'lens_1' else None
            for pb in self._pgroups(cidxs, pbinmap):
                cc = [c for c in cidxs if pbinmap is None or pbinmap[c] == pb]
                plab = (self.lens_label(primary) + f' bin {pb}') if pb is not None \
                    else 'All countries'
                jj = [cidxs.index(c) for c in cc]
                den_cc = den[jj] if den is not None else None
                if secondary is None and primary == 'lens_1':
                    ha = float(gsub[jj].sum())
                    t = float(den_cc.sum()) if den_cc is not None else None
                    rows.append((COVER_LABELS.get(cov, cov), cfg, plab, 'country total',
                                 ha, 100.0 * ha / t if t else None))
                    for j, c in enumerate(cc):
                        hj = float(gsub[j].sum())
                        tj = float(den_cc[j].sum()) if den_cc is not None else None
                        if hj > 0:
                            box.append((COVER_LABELS.get(cov, cov), plab, 'country',
                                        self.country_label(c),
                                        100.0 * hj / tj if tj else None))
                    continue
                for b in range(gsub.shape[1]):
                    ha = float(gsub[jj, b].sum())
                    t = float(den_cc[:, b].sum()) if den_cc is not None else None
                    if ha == 0 and (t or 0) == 0:
                        continue
                    rows.append((COVER_LABELS.get(cov, cov), cfg, plab,
                                 self.bin_label(alens, b),
                                 ha, 100.0 * ha / t if t else None))
                    for j, c in enumerate(cc):
                        hj = float(gsub[j, b])
                        tj = float(den_cc[j, b]) if den_cc is not None else None
                        if hj > 0:
                            box.append((COVER_LABELS.get(cov, cov), plab,
                                        self.bin_label(alens, b),
                                        self.country_label(c),
                                        100.0 * hj / tj if tj else None))
                # -- bin 0: raster value 99 = pixels with no lens data ------
                # macro (lens_1) minus the sum over lens bins, reported as
                # its own group so gap totals stay invariant across lenses.
                g1 = g.get('lens_1')
                if g1 is not None:
                    rem = g1[cidxs][jj].sum(axis=1) - gsub[jj].sum(axis=1)  # [c]
                    ha = float(rem.sum())
                    if abs(ha) > 1e-6:
                        trows = None
                        if tot_npz is not None and cov in tot_npz.files \
                                and 'lens_1' in self.lens_names:
                            li1 = self.lens_names.index('lens_1')
                            if li1 < tot_npz[cov].shape[0]:
                                trows = tot_npz[cov][li1][:, :gsub.shape[1]][cidxs][jj]
                        tr = float(trows.sum() - den_cc.sum()) \
                            if trows is not None and den_cc is not None else None
                        rows.append((COVER_LABELS.get(cov, cov), cfg, plab,
                                     'bin 0 (no lens data)',
                                     ha, 100.0 * ha / tr if tr and tr > 0.5 else None))
                        for j in range(len(cc)):
                            hj = float(rem[j])
                            if hj > 0:
                                tj = None
                                if trows is not None and den_cc is not None:
                                    tj = float(trows[j].sum() - den_cc[j].sum())
                                box.append((COVER_LABELS.get(cov, cov), plab,
                                            'bin 0 (no lens data)',
                                            self.country_label(cc[j]),
                                            100.0 * hj / tj if tj and tj > 0.5 else None))
        return {'rows': rows, 'box': box}




_BUNDLE = None


def get_bundle() -> Bundle:
    """Process-wide singleton (the npz caches live on the instance)."""
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = Bundle(resolve_bundle_dir())
    return _BUNDLE

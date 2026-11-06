#!/usr/bin/env python3
"""Streamlit page: "Land Condition (IIS)" — web port of the desktop
IIS land-condition engine GUI (enginev2 thin client).

Answers: hectares of degraded land (or weighted gap surface) per factor
per lens bin, for any country selection, under any primary/aggregation
lens — with the "bin 0 (no lens data)" invariant intact: totals are the
same no matter which lens is chosen. All heavy lifting is precomputed
(see iis_condition.py); queries are in-memory array slices.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import iis_condition as ic
from iis_condition import (COVER_LABELS, FACTOR_LABELS, INDEX_NAMES,
                           LENS_NAMES)

BIN0 = 'bin 0 (no lens data)'


@st.cache_resource
def bundle():
    return ic.get_bundle()


@st.cache_data
def run_query(kind, fac_or_idx, covers, cfg, country_idxs, primary, secondary,
              box_class):
    """Thin cache wrapper; args must be hashable (tuples)."""
    b = bundle()
    if kind.startswith('Factor'):
        return b.query_state(fac_or_idx, list(covers), list(country_idxs),
                             primary=primary, secondary=secondary,
                             box_class=box_class)
    return b.query_gap(fac_or_idx, cfg, list(covers), list(country_idxs),
                       primary=primary, secondary=secondary)


def _bin_key(s):
    try:
        return (0, int(str(s).split()[-1]))
    except (ValueError, IndexError):
        return (1, str(s))


def _fmt_ha(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return f'{v / 1e6:,.2f}'


def page_land_condition():
    b = bundle()

    st.title('Land Condition (IIS)')
    st.caption(
        'Precomputed land-condition engine: hectares of degraded land per '
        f'factor (and weighted gap surface per index), by lens bin, over '
        f'{len(b.country_ids)} countries. "bin 0 (no lens data)" keeps '
        'totals invariant: the sum of degraded area is identical with any '
        'lens, another lens, or no lens at all.'
    )

    # ---- sidebar (left panel) controls -----------------------------------
    with st.sidebar:
        st.markdown('### Land Condition (IIS)')
        kind = st.radio('Surface', ['Factor (state)', 'Index (gap)'],
                        horizontal=True)

        fac = idx = ''
        prim, sec = 'lens_1', None
        sel_covers, cfg = ['forests'], 'linear'
        if kind == 'Factor (state)':
            fkeys = sorted(FACTOR_LABELS, key=lambda k: FACTOR_LABELS[k])
            flabels = [FACTOR_LABELS[k] for k in fkeys]
            fac = {FACTOR_LABELS[k]: k for k in fkeys}[st.selectbox(
                'Factor', flabels,
                index=flabels.index(FACTOR_LABELS[fac]) if fac in fkeys
                else flabels.index('SDG 15.3.1 degraded (binary)'))]
            ckeys = b.covers
            cov_by_label = {COVER_LABELS.get(c, c): c for c in ckeys}
            picked_covs = st.multiselect(
                'Covers', list(cov_by_label),
                default=[COVER_LABELS.get(c, c) for c in sel_covers
                         if c in ckeys] or [COVER_LABELS.get(ckeys[0], ckeys[0])])
            sel_covers = [cov_by_label[l] for l in picked_covs]
        else:
            ikeys = sorted(INDEX_NAMES)
            idx = {INDEX_NAMES.get(k, k): k for k in ikeys}[st.selectbox(
                'Index (2x family)', [INDEX_NAMES.get(k, k) for k in ikeys])]
            ckeys = b.covers
            cov_by_label = {COVER_LABELS.get(c, c): c for c in ckeys}
            clabels_cov = list(cov_by_label)
            sel_covers = [cov_by_label[st.selectbox(
                'Cover', clabels_cov,
                index=clabels_cov.index(COVER_LABELS.get(sel_covers[0],
                                                         sel_covers[0]))
                if sel_covers[0] in ckeys else 0)]]
            cfg = st.selectbox('Aggregation threshold', b.cfgs, index=0)

        # countries: multiselect by label; empty = all.
        # query_* take ROW indices into the bundle arrays (position in
        # country_ids), NOT gaul codes — same convention as the desktop GUI's
        # selected_country_idxs().
        rows = list(range(len(b.country_ids)))
        clabels = [b.country_label(i) for i in rows]
        label_to_row = dict(zip(clabels, rows))
        picked = st.multiselect('Countries (empty = all)', clabels, default=[])
        country_idxs = ([label_to_row[l] for l in picked] if picked
                        else rows)

        consts = b.constant_lenses()
        lens_by_label = {b.lens_label(l): l for l in consts}
        prim = lens_by_label[st.selectbox(
            'Primary lens', list(lens_by_label),
            index=list(lens_by_label).index(b.lens_label(prim)))]
        nat = [l for l in consts if l != 'lens_1' and l != prim]
        spats = b.spatial_lenses()
        sec_keys = nat + list(spats)
        sec_opts = ['(none — macro only)'] + [
            b.lens_label(l) + (' (national)' if l in nat else '')
            for l in sec_keys]
        sec_pick = st.selectbox(
            'Aggregation lens (national or spatial)', sec_opts,
            index=(sec_opts.index(b.lens_label(sec))
                   if sec in sec_keys and sec not in nat
                   else (sec_opts.index(b.lens_label(sec) + ' (national)')
                         if sec in nat else 0)))
        sec = (None if sec_pick.startswith('(none')
               else {b.lens_label(l) + (' (national)' if l in nat else ''): l
                     for l in sec_keys}[sec_pick])

    if not sel_covers or not country_idxs:
        st.info('Pick at least one cover and one country.')
        st.stop()

    box_class = '1'
    classes = []
    if kind == 'Factor (state)':
        _, classes = b.state(fac, sel_covers[0])
        classes = [c for c in classes if c != 'F_NODATA']
        box_class = st.selectbox(
            'Class for charts / % (relative view)', classes,
            index=classes.index('1') if '1' in classes else 0)

    # ---- query ------------------------------------------------------------
    res = run_query(kind, fac or idx, tuple(sel_covers), cfg,
                    tuple(country_idxs), prim, sec, str(box_class))

    rows = res['rows']
    ha_col = 'Million ha'
    if kind == 'Factor (state)':
        cols = ['Cover', 'Class', 'Primary lens group', 'Aggregation bin',
                'ha', '% of cover']
    else:
        cols = ['Cover', 'Config', 'Primary lens group', 'Aggregation bin',
                'ha (weighted)', '% of cover']
    df = pd.DataFrame(rows, columns=cols)
    hac, pcc = cols[4], cols[5]
    df[hac] = pd.to_numeric(df[hac], errors='coerce')  # None -> NaN
    df[pcc] = pd.to_numeric(df[pcc], errors='coerce')
    df[ha_col] = (df[hac] / 1e6).round(2)
    df['% of cover'] = df[pcc].round(1)

    if kind.startswith('Factor'):
        # headline = the selected class (the degraded class), not all classes
        tot = df.loc[df[cols[1]] == str(box_class), hac].sum()
    else:
        tot = df[hac].sum()
    tot_fmt = f'{tot / 1e6:,.2f} M ha'
    st.subheader(f'{tot_fmt} — '
                 + (FACTOR_LABELS.get(fac, fac) if kind == 'Factor (state)'
                    else f'{INDEX_NAMES.get(idx, idx)} · {cfg}')
                 + ' · ' + ' + '.join(COVER_LABELS.get(c, c) for c in sel_covers))

    m1, m2, m3 = st.columns(3)
    m1.metric('Countries selected', len(country_idxs))
    m2.metric('Primary lens', b.lens_label(prim).replace('lens_', ''))
    m3.metric('Aggregation lens',
              '(none)' if sec is None else b.lens_label(sec).replace('lens_', ''))

    # ---- summary table ----------------------------------------------------
    st.markdown('#### Summary by lens bin')
    st.dataframe(
        df[[cols[1], cols[2], cols[3], ha_col, '% of cover']],
        width='stretch', hide_index=True,
    )

    # ---- per-country table (from box output, relative only) ---------------
    box = res['box']  # (cover, group, bin, country_label, pct)
    bx = pd.DataFrame(box, columns=['Cover', 'Group', 'Bin', 'Country', 'pct'])
    grp = bx[bx['Bin'] == 'country'][['Group', 'Country', 'pct']]
    per = (grp.pivot_table(index='Country', columns='Group', values='pct')
              .sort_index(axis=1))
    st.markdown(f'#### Relative share of "{box_class}" per country (%)'
                if kind == 'Factor (state)' else
                '#### Relative weighted surface per country (%)')
    st.dataframe(per.round(1), width='stretch')

    # ---- charts -----------------------------------------------------------
    bl = bx[bx['Bin'] != 'country'].copy()
    bl['_sort'] = bl['Bin'].map(lambda s: '{:04d}'.format(_bin_key(s)[1])
                                if _bin_key(s)[0] == 0 else 'zzz' + str(s))
    bl = bl.sort_values('_sort')
    top = (grp.groupby('Country')['pct'].sum()
              .sort_values(ascending=False).index.tolist())
    if len(top) > 20:
        keep = set(top[:20])
        bl['Country'] = bl['Country'].map(
            lambda c: c if c in keep else f'Other ({len(top) - 20})')
    fig = px.bar(
        bl, x='Country', y='pct', color='Bin', barmode='stack',
        labels={'pct': '% of cover', 'Country': ''},
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='Relative share by aggregation bin — top 20 countries',
    )
    fig.update_layout(legend_title_text='Aggregation bin', height=420)
    st.plotly_chart(fig, width='stretch')

    fig2 = px.box(
        grp, x='Group', y='pct', points='all',
        labels={'pct': '% of cover', 'Group': ''},
        color_discrete_sequence=['#4c78a8'],
        title='Distribution across countries by primary lens group',
    )
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, width='stretch')

    # ---- exports ----------------------------------------------------------
    tag = (f"{(fac or idx).split('/')[-1] if kind == 'Factor (state)' else idx}"
           f"_{'-'.join(sel_covers)}_{cfg}_{prim}"
           f"{'_' + sec if sec else ''}")
    sbuf = io.StringIO()
    df.to_csv(sbuf, index=False)
    st.download_button('Download summary CSV', sbuf.getvalue(),
                       file_name=f'iis_summary_{tag}.csv', mime='text/csv')
    pbuf = io.StringIO()
    per.round(2).to_csv(pbuf)
    st.download_button('Download per-country CSV', pbuf.getvalue(),
                       file_name=f'iis_by_country_{tag}.csv', mime='text/csv')

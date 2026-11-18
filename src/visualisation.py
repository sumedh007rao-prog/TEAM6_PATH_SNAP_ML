"""Folium map builder used by the notebooks and by the evaluator UI.

Layers (toggle with the layer control, legend bottom-left):
  * GPS observations   (blue points + thin blue line)
  * Road network       (grey; context arcs near the trip)
  * Reference route    (green; only when supplied, never for evaluator trips)
  * Classifier-only    (orange)
  * HMM / Viterbi      (red)
"""
from __future__ import annotations

LEGEND_HTML = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #888; border-radius: 6px; font-size: 13px;">
  <b>{title}</b><br>
  {rows}
</div>
"""
ROW = '<span style="display:inline-block;width:18px;height:4px;background:{c};margin-right:6px;"></span>{n}<br>'


def _bounds(points):
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def build_trip_map(gps_latlon, network_segments=None, reference_polylines=None,
                   classifier_polylines=None, hmm_polylines=None,
                   title="PathSnap trip", max_gps_markers=600):
    """Return a ``folium.Map``.

    gps_latlon          : list of [lat, lon]
    network_segments    : list of [[lat, lon], [lat, lon]] (context road arcs)
    *_polylines         : list of polylines, each a list of [lat, lon]
    """
    import folium  # imported lazily: only needed where maps are drawn

    m = folium.Map(location=gps_latlon[len(gps_latlon) // 2], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    legend = []

    def add_lines(name, polylines, color, weight, opacity, show=True):
        fg = folium.FeatureGroup(name=name, show=show)
        for pl in polylines or []:
            if len(pl) >= 2:
                folium.PolyLine(pl, color=color, weight=weight, opacity=opacity,
                                tooltip=name).add_to(fg)
        fg.add_to(m)
        legend.append(ROW.format(c=color, n=name))

    if network_segments:
        add_lines("Road network", network_segments, "#9a9a9a", 1.5, 0.7, show=False)
    if reference_polylines:
        add_lines("Reference route", reference_polylines, "#1a9850", 7, 0.55)
    if classifier_polylines:
        add_lines("Classifier-only route", classifier_polylines, "#f28e2b", 4, 0.9, show=False)
    if hmm_polylines:
        add_lines("HMM / Viterbi route", hmm_polylines, "#d62728", 4, 0.9)

    fg = folium.FeatureGroup(name="GPS observations")
    folium.PolyLine(gps_latlon, color="#1f77b4", weight=1, opacity=0.5).add_to(fg)
    step = max(1, len(gps_latlon) // max_gps_markers)
    for i in range(0, len(gps_latlon), step):
        folium.CircleMarker(gps_latlon[i], radius=2.5, color="#1f77b4", fill=True,
                            fill_opacity=0.9, weight=1, popup=f"obs {i}").add_to(fg)
    folium.Marker(gps_latlon[0], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker(gps_latlon[-1], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    fg.add_to(m)
    legend.append(ROW.format(c="#1f77b4", n="GPS observations"))

    m.fit_bounds(_bounds(gps_latlon))
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        LEGEND_HTML.format(title=title, rows="".join(legend))))
    return m

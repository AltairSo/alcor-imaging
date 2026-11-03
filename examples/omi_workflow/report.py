"""Summarize measured usability; lunar context is descriptive, not a veto."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

from archive import atomic_json


def summarize(root):
    root=Path(root)
    rows=json.loads((root/'quality/frame_report.json').read_text())
    overrides=json.loads((root/'quality/overrides.json').read_text()) if (root/'quality/overrides.json').exists() else {}
    grouped=defaultdict(list)
    for r in rows:
        r=dict(r)
        if r['path'] in overrides:
            r['status']=overrides[r['path']]['status']
            r['reasons']+=['Manual review: '+overrides[r['path']].get('reason','')]
        grouped[(r['night'],r.get('object') or 'Unknown',r.get('filter') or 'Unknown')].append(r)
    summary=[];moon={}
    for (night,obj,filt),group in sorted(grouped.items()):
        counts=Counter(r['status'] for r in group)
        for r in group:
            if night not in moon and r.get('date_obs'):
                try:
                    t=Time(r['date_obs'],scale='utc')
                    e=get_body_barycentric('earth',t).xyz.value
                    s=get_body_barycentric('sun',t).xyz.value
                    m=get_body_barycentric('moon',t).xyz.value
                    cosine=float(np.dot(e-m,s-m)/(np.linalg.norm(e-m)*np.linalg.norm(s-m)))
                    moon[night]=round(100*(1+cosine)/2,1)
                except (ValueError,TypeError):pass
        keep=[r for r in group if r['status'] in {'candidate','accepted'}]
        summary.append({'night':night,'object':obj,'filter':filt,'total':len(group),
                        'candidate':len(keep),'review':counts['review'],'reject':counts['reject'],
                        'planetary_or_test':counts['planetary_or_test'],'non_image':counts['non_image'],
                        'candidate_minutes':round(sum(float(r.get('exposure') or 0) for r in keep)/60,1),
                        'moon_illumination_percent_approx':moon.get(night)})
    atomic_json(root/'quality/summary.json',summary)
    text=['# OMI archive quality report','',
          'Candidate = passes conservative screening; final use still requires successful registration and visual review. '
          'Review frames are excluded by default. Rejected originals remain archived. Planetary/test frames need their own workflow.','',
          'Moon phase is approximate context from FITS DATE-OBS and Astropy built-in ephemerides. It is not an exclusion rule or proof of the cause of a defect.','',
          '| Night | Target | Filter | Total | Candidate | Review | Reject | Planet/test | Non-image | Candidate min | Moon lit % |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append('| '+' | '.join(str(r[k]) for k in ['night','object','filter','total','candidate','review','reject','planetary_or_test','non_image','candidate_minutes','moon_illumination_percent_approx'])+' |')
    text+=['','## Processing limits','',
           '- Supplied lights are not dark/flat calibrated unless matching calibration is added separately.',
           '- Sky background, gradients, stellar elongation, banding and saturation are measured separately.',
           '- Five native-pixel star samples are supplemented by full-field morphology to catch long trails and arcs.',
           '- Automatic colour outputs are review images. SHO/HOO colours are mappings; RGB is not photometrically calibrated.',
           '- The notebook produces 4×4-binned review masters by default; binning=1 produces native-resolution masters.',
           '- No outside image pixels or generated astronomical detail are used.']
    (root/'QUALITY_REPORT.md').write_text('\n'.join(text)+'\n')
    return summary

"""Probe for the rank-fit switch (38fef7e): stops FeatureForge at the point where it decides where its search-time
frequency ranks are fitted, and writes the share of later rows holding levels the search rows lack to PROBE_OUT.
Put this folder first on PYTHONPATH (and the worktree under test after it); subprocesses inherit it."""
import json, os, sys
try:
    from tabularaml.generate import forge as _F
except Exception:  # not a FeatureForge process
    _F = None
if _F is not None and os.environ.get('PROBE_OUT'):
    _orig = _F.FeatureForge._log

    def _log(self, msg):
        rec = None
        if str(msg).startswith('frequency ranks:'):
            rec = dict(msg=str(msg), hc_cols=list(getattr(self, 'hc_cols_', []))[:20], n_hc=len(getattr(self, 'hc_cols_', [])))
        elif hasattr(self, 'base_cv_loss_') and not getattr(self, 'hc_cols_', None):
            rec = dict(msg='no high-cardinality columns: switch not reached', n_hc=0)
        if rec is not None:
            with open(os.environ['PROBE_OUT'], 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print('PROBE', json.dumps(rec), flush=True)
            os._exit(0)
        return _orig(self, msg)
    _F.FeatureForge._log = _log

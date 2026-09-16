"""Bias, conformal, and category-probability calibration for V4."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from spatial.rule_contract import category

LEVELS=np.asarray((.05,.10,.25,.50,.75,.90,.95))


def apply_bias(quantiles, offsets, leads):
    result=np.asarray(quantiles,float).copy(); shift=np.asarray([offsets.get(str(float(x)),offsets.get("global",0.0)) for x in leads])
    return result+shift[:,None]


def choose_bias(fit_actual,fit_q,fit_leads,val_actual,val_q,val_leads):
    global_offset=float(np.clip(np.mean(fit_actual-fit_q[:,3]),-1.5,1.5))
    lead_offsets={}
    for lead in np.unique(fit_leads):
        selected=fit_leads==lead;shrink=float(selected.sum()/(selected.sum()+200.0))
        raw=float(np.mean(fit_actual[selected]-fit_q[selected,3]));lead_offsets[str(float(lead))]=float(np.clip(shrink*raw+(1-shrink)*global_offset,-1.5,1.5))
    candidates={"none":{"global":0.0},"global":{"global":global_offset},"lead":lead_offsets}
    def score(offsets):
        prediction=apply_bias(val_q,offsets,val_leads)[:,3];error=prediction-val_actual
        return np.mean(np.abs(error)),np.sqrt(np.mean(error**2)),abs(np.mean(error))
    baseline=score(candidates["none"]);ranking=[]
    for name,offsets in candidates.items(): ranking.append((score(offsets),name,offsets))
    ranking.sort();selected=next((x for x in ranking if x[0][2]<=baseline[2] and not(x[0][0]>baseline[0] and x[0][1]>baseline[1])),
                                 next(x for x in ranking if x[1]=="none"))
    return {"type":selected[1],"offsets":selected[2],"validation":dict(zip(("mae","rmse","absolute_bias"),selected[0]))}


def conformal(actual,quantiles,target=.8):
    scores=np.maximum.reduce((quantiles[:,1]-actual,actual-quantiles[:,5],np.zeros(len(actual))))
    return float(np.quantile(scores,target,method="higher"))


def apply_conformal(quantiles,expansion):
    result=np.asarray(quantiles).copy();result[:,:3]-=expansion;result[:,4:]+=expansion;return result


def category_probability(quantiles,rh,wind_ms,samples=101):
    grid=np.linspace(0,1,samples);result=[]
    for row,r,w in zip(quantiles,rh,wind_ms):
        values=np.interp(grid,LEVELS,row,left=max(0,row[0]-(row[1]-row[0])),right=row[-1]+(row[-1]-row[-2]))
        classified=[category(f,r,float(w)*1.9438444924406) for f in values]
        result.append(np.mean([value is not None and value>=2 for value in classified]))
    return np.asarray(result)


def fit_isotonic(probability,actual):
    model=IsotonicRegression(out_of_bounds="clip").fit(probability,np.asarray(actual,float))
    return {"x":model.X_thresholds_.tolist(),"y":model.y_thresholds_.tolist()}


def apply_isotonic(probability,parameters):
    return np.interp(probability,parameters["x"],parameters["y"])

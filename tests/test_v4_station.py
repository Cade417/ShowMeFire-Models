import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spatial.observations import causal_initial_and_weather_targets
from spatial.physics import evolve_fm_with_rain
from spatial.rule_contract import RULE_SPEC_SHA256, category, check_api_copy
from spatial.v4_calibration import apply_bias, apply_conformal, category_probability, choose_bias, conformal
from spatial.v4_contract import create_manifest, reject_forbidden, validate_manifest, prospective_runs
from spatial.v4_features import FEATURES, add_v4_features, precipitation_features
from spatial.v4_guard import fit_lead_guard, sequence_weights
from spatial.v4_model import GuardedQuantileGRU
from spatial.v4_training import base_configs, residual_configs, train_residual
from spatial.station_contract import detailed_metrics
from spatial.register_v4 import validate_registration


class V4EvidenceTests(unittest.TestCase):
    def test_realistic_contract_includes_chronological_summer_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset=Path(directory)/"data.csv";dataset.write_text("fixture")
            dates=pd.date_range("2025-07-14",periods=120,tz="UTC")
            frame=pd.DataFrame({"run_id":[str(x) for x in range(120)],"forecast_init_time":dates})
            dev=[str(x) for x in range(100)]
            inherited=[{"name":f"old-{n}","train_runs":dev[:50+n*20],"validation_runs":dev[50+n*20:60+n*20]} for n in range(3)]
            v3={"development_runs":dev,"calibration_runs":[str(x) for x in range(100,110)],
                "locked_test_runs":[str(x) for x in range(110,120)],"folds":inherited}
            manifest=create_manifest(frame,dataset,v3,FEATURES);summer=manifest["folds"][0]
            self.assertEqual(summer["name"],"fold-summer")
            mapping=dict(zip(frame.run_id,frame.forecast_init_time))
            self.assertLess(max(mapping[x] for x in summer["train_runs"]),min(mapping[x] for x in summer["validation_runs"]))
            self.assertTrue(all(mapping[x].month==8 for x in summer["validation_runs"]))

    def test_v3_test_runs_are_permanently_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset=Path(directory)/"data.csv";dataset.write_text("fixture")
            frame=pd.DataFrame({"run_id":[str(x) for x in range(10)],"forecast_init_time":pd.date_range("2026-01-01",periods=10,tz="UTC")})
            v3={"development_runs":[str(x) for x in range(6)],"calibration_runs":["6","7"],"locked_test_runs":["8","9"],
                "folds":[{"name":"fold-1","train_runs":["0","1"],"validation_runs":["2"]}]}
            manifest=create_manifest(frame,dataset,v3,FEATURES)
            self.assertFalse(set(manifest["forbidden_v3_test_runs"]) & set(manifest["development_runs"]))
            with self.assertRaisesRegex(RuntimeError,"forbidden"):
                reject_forbidden(manifest,["8"],"test")

    def test_prospective_rows_can_append_without_changing_frozen_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset=Path(directory)/"data.csv";dataset.write_text("fixture")
            frame=pd.DataFrame({"run_id":[str(x) for x in range(10)],"forecast_init_time":pd.date_range("2026-01-01",periods=10,tz="UTC")})
            v3={"development_runs":[str(x) for x in range(6)],"calibration_runs":["6","7"],"locked_test_runs":["8","9"],
                "folds":[{"name":"fold-1","train_runs":["0","1"],"validation_runs":["2"]}]}
            manifest=create_manifest(frame,dataset,v3,FEATURES)
            appended=pd.concat((frame,pd.DataFrame({"run_id":["new"],"forecast_init_time":[pd.Timestamp("2026-09-01T12:00Z")]})),ignore_index=True)
            validate_manifest(manifest,appended,dataset,FEATURES);self.assertEqual(prospective_runs(manifest,appended),["new"])

    def test_rule_contract_matches_api(self):
        self.assertEqual(len(RULE_SPEC_SHA256),64);self.assertEqual(check_api_copy(),RULE_SPEC_SHA256)
        self.assertEqual(category(6.9,19.9,25),4);self.assertIsNone(category(np.nan,20,25))

    def test_registration_refuses_non_prospective_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,"prospective"):
                validate_registration(Path(directory),{"status":"calibration-validation","pass":True,"beta_registration_allowed":False})


class V4DataTests(unittest.TestCase):
    def test_weather_targets_share_one_timestamp(self):
        obs=pd.DataFrame({"time":pd.to_datetime(["2026-01-01T00:00Z","2026-01-01T04:05Z"]),"fuel_moisture":[10,7],
                          "obs_rh":[50,20],"obs_wind_ms":[1,8]})
        initial,age,targets=causal_initial_and_weather_targets(obs,pd.Timestamp("2026-01-01T00:00Z"),[pd.Timestamp("2026-01-01T04:00Z")])
        self.assertEqual((initial,age),(10,0));self.assertEqual(targets[0]["target_fm"],7);self.assertEqual(targets[0]["target_rh"],20)
        self.assertEqual(targets[0]["target_match_age_minutes"],5)

    def test_precip_increment_reset_and_causal_rollup(self):
        frame=pd.DataFrame({"run_id":["1"]*4,"station_id":["A"]*4,"lead_hour":[4,5,6,7],
                            "hrrr_precip_mm":[0,2,1,4],"hrrr_rh":[40,60,50,70]})
        result=precipitation_features(frame)
        self.assertEqual(result.hrrr_precip_increment_mm.tolist(),[0,2,0,3])
        self.assertEqual(result.precip_reset_flag.tolist(),[0,0,1,0]);self.assertEqual(result.precip_3h_mm.tolist(),[0,2,2,5])

    def test_summer_features_use_only_forecast_inputs(self):
        frame=pd.DataFrame({"run_id":["1"],"station_id":["A"],"lead_hour":[4],
            "valid_time":pd.to_datetime(["2026-07-01T16:00Z"]),"initial_fm":[10.],"physics_fm":[9.],
            "rtma_temp_c":[25.],"rtma_rh":[40.],"rtma_wind_ms":[2.],"hrrr_temp_c":[35.],
            "hrrr_rh":[25.],"hrrr_wind_ms":[3.],"hrrr_precip_mm":[0.],"initial_age_hours":[1.],
            "lat":[38.],"lon":[-92.]})
        result=add_v4_features(frame,[8.],"legacy")
        self.assertEqual(result.summer_indicator.iloc[0],1.)
        self.assertGreater(result.hot_dry_interaction.iloc[0],0.)
        changed=frame.assign(target_fm=1.,target_rh=1.,target_wind_ms=99.)
        second=add_v4_features(changed,[8.],"legacy")
        self.assertEqual(result[FEATURES].to_numpy().tolist(),second[FEATURES].to_numpy().tolist())

    def test_rain_physics_wets_state(self):
        dry=evolve_fm_with_rain(8,[25,25],[30,30],[0,0]);wet=evolve_fm_with_rain(8,[25,25],[30,30],[5,0])
        self.assertGreater(wet[0],dry[0])


class V4ModelTests(unittest.TestCase):
    def test_search_spaces_are_exact(self):
        self.assertEqual(len(base_configs()),16);self.assertEqual(len(residual_configs()),12)

    def test_seven_quantiles_are_ordered_and_zero_guard_is_base(self):
        model=GuardedQuantileGRU(len(FEATURES),32,4);x=torch.randn(5,12,len(FEATURES));base=torch.randn(5,12)
        prediction=model(x,base,torch.zeros(5,12));self.assertEqual(prediction.shape,(5,12,7))
        self.assertTrue(torch.all(torch.diff(prediction,dim=-1)>0));self.assertTrue(torch.equal(prediction[...,3],base))

    def test_reduced_training_is_deterministic(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(8,12,len(FEATURES))).astype("float32")
        base=rng.normal(10,1,size=(8,12)).astype("float32");target=(base-.4).astype("float32")
        mask=np.ones_like(target,"float32");leads=np.tile(np.arange(4,16,dtype="float32"),(8,1));arrays=(x,base,target,mask,leads)
        arguments=dict(hidden_size=32,gate_regularization=.01,residual_cap=2,fixed_epochs=2,epochs=2,device="cpu",seed=417)
        _,first,_=train_residual(arrays,None,**arguments);_,second,_=train_residual(arrays,None,**arguments)
        for key in first["state_dict"]:self.assertTrue(torch.equal(first["state_dict"][key],second["state_dict"][key]))

    def test_oof_lead_guard_rejects_harmful_correction(self):
        rows=[]
        for fold in ("a","b","c"):
            for index in range(500):
                rows.append({"fold":fold,"lead_hour":4.0,"actual":10.0,"base":10.5,"residual_p50":10.0})
                rows.append({"fold":fold,"lead_hour":5.0,"actual":10.0,"base":10.1,"residual_p50":15.0})
        guard=fit_lead_guard(pd.DataFrame(rows));self.assertGreater(guard["4.0"],0);self.assertEqual(guard["5.0"],0)
        self.assertEqual(sequence_weights(np.array([[4.,5.]]),guard).shape,(1,2))


class V4CalibrationTests(unittest.TestCase):
    def test_detailed_metrics_accepts_v4_p10_p50_p90_slice(self):
        table=pd.DataFrame({"target_fm":np.arange(30.),"prediction":np.arange(30.),
            "initial_fm":np.arange(30.)+1,"lead_hour":[4.]*30,"station_id":["A"]*30,
            "month":[7]*30})
        seven=np.column_stack([table.target_fm+x for x in (-3,-2,-1,0,1,2,3)])
        report=detailed_metrics(table,seven[:,[1,3,5]])
        self.assertEqual(report["interval_coverage"],1.)

    def test_bias_selection_order_and_conformal(self):
        actual=np.arange(20,dtype=float);q=np.column_stack([actual-3,actual-2,actual-1,actual-.5,actual+1,actual+2,actual+3]);leads=np.full(20,4.)
        selected=choose_bias(actual[:10],q[:10],leads[:10],actual[10:],q[10:],leads[10:])
        shifted=apply_bias(q,selected["offsets"],leads);expanded=apply_conformal(shifted,conformal(actual,shifted))
        self.assertTrue(np.all(np.diff(expanded,axis=1)>=0));self.assertGreaterEqual(np.mean((actual>=expanded[:,1])&(actual<=expanded[:,5])),.8)

    def test_conformal_fit_half_is_applied_without_validation_leakage(self):
        fit_actual=np.arange(100,dtype=float);fit_q=np.column_stack([fit_actual+x for x in (-3,-2,-1,0,1,2,3)])
        expansion=conformal(fit_actual,fit_q);self.assertEqual(expansion,0.)
        shifted_actual=fit_actual+10;valid=apply_conformal(fit_q,expansion)
        self.assertLess(np.mean((shifted_actual>=valid[:,1])&(shifted_actual<=valid[:,5])),.8)

    def test_category_probability_tracks_dryness(self):
        dry=np.tile([3,4,5,6,7,8,9],(4,1));wet=dry+20
        self.assertGreater(category_probability(dry,[20]*4,[10]*4).mean(),category_probability(wet,[20]*4,[10]*4).mean())


if __name__=="__main__":unittest.main()

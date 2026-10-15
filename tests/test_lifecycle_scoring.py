"""Regression checks for record/window calibration-unit confusion; no data files."""
import unittest
import numpy as np
from csgmm.lifecycle_scoring import aggregate_records,fit_record_calibration,apply_record_calibration


class LifecycleCalibrationTest(unittest.TestCase):
    def test_q95_uses_records_not_windows(self):
        raw=np.concatenate([np.full(16,i,float) for i in range(10)])
        condition=np.zeros(len(raw),int);record=np.repeat(np.arange(10),16)
        rows=aggregate_records(raw,condition,record)
        scales,thresholds=fit_record_calibration(rows)
        score,prediction,threshold=apply_record_calibration(rows,scales,thresholds)
        self.assertAlmostEqual(scales['0']['median'],4.5)
        self.assertAlmostEqual(scales['0']['iqr'],4.5)
        expected=float(np.quantile((np.arange(10,dtype=np.float32)-4.5)/4.5,.95))
        self.assertEqual(thresholds['0'],expected)
        window_threshold=float(np.quantile(raw,.95))
        self.assertNotAlmostEqual(window_threshold,thresholds['0']*4.5+4.5)
        np.testing.assert_array_equal(prediction,score>=threshold)

    def test_conditions_with_same_record_ids_remain_separate(self):
        raw=np.r_[np.ones(16),np.full(16,10.)]
        rows=aggregate_records(raw,np.repeat([0,1],16),np.zeros(32,int))
        np.testing.assert_array_equal(rows['raw'],[1.,10.])
        scales,thresholds=fit_record_calibration(rows)
        self.assertEqual(scales['0']['iqr'],1e-8)
        self.assertEqual(scales['1']['median'],10.)
        np.testing.assert_array_equal(apply_record_calibration(rows,scales,thresholds)[0],[0.,0.])

    def test_incomplete_record_is_rejected(self):
        with self.assertRaises(ValueError):aggregate_records(np.zeros(15),np.zeros(15),np.zeros(15))

    def test_test_records_do_not_refit_calibration(self):
        cal=dict(raw=np.array([1.,2.,3.,4.],np.float32),condition=np.zeros(4,int))
        scales,thresholds=fit_record_calibration(cal)
        test=dict(raw=np.array([100.],np.float32),condition=np.zeros(1,int))
        self.assertEqual(apply_record_calibration(test,scales,thresholds)[1][0],1)
        self.assertEqual(scales['0']['median'],2.5)


if __name__=='__main__':unittest.main()

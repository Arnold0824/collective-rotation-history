import unittest
from .core import *

class Checks(unittest.TestCase):
    def test_thresholds(self):
        np.testing.assert_array_equal(states(np.array([-.31,-.3,-.299,0,.299,.3,.31])),[0,0,1,1,1,2,2])
    def test_probability_score_scale(self):
        p=np.array([[1.,0,0],[0,0,1.]])
        score=probability_metrics(p,np.array([0,0]))
        self.assertEqual(score['brier'],1.)
        self.assertEqual(score['rps'],1.)
        perfect=probability_metrics(np.eye(3),np.arange(3))
        self.assertEqual(perfect['rps'],0.)
    def test_rotation_and_reflection(self):
        angle=np.arange(10)*2*np.pi/10
        p=np.stack([np.cos(angle),np.sin(angle)],axis=-1)[None,:,:]
        v=np.stack([-np.sin(angle),np.cos(angle)],axis=-1)[None,:,:]
        a,*_=rotation(p,v,np.ones((1,10),bool));np.testing.assert_allclose(a,1)
        p[:,:,1]*=-1;v[:,:,1]*=-1
        b,*_=rotation(p,v,np.ones((1,10),bool));np.testing.assert_allclose(b,-1)
    def test_identity_break_and_future_independence(self):
        rng=np.random.default_rng(2);p=np.cumsum(rng.normal(0,.1,(160,10,2)),axis=0)+np.arange(10)[None,:,None]*100
        p[80:]=p[80:,::-1]
        head=np.full((160,10),.2);f=np.arange(160)
        a=paired_observables(p,head,f,1e-12)
        self.assertTrue(a['boundary'][80]);self.assertFalse(a['valid'][80]);self.assertNotEqual(a['run_id'][79],a['run_id'][81])
        p[120:]+=20000
        b=paired_observables(p,head,f,1e-12)
        np.testing.assert_allclose(a['heading'][:120],b['heading'][:120],equal_nan=True)
        np.testing.assert_allclose(a['displacement'][:120],b['displacement'][:120],equal_nan=True)
    def test_common_individual_mask(self):
        rng=np.random.default_rng(4);p=rng.normal(size=(8,10,2));h=rng.normal(size=(8,10));p[3,0]=p[2,0];h[5,:3]=np.nan
        o=paired_observables(p,h,np.arange(8),1e-12)
        self.assertEqual(o['count'][3],9);self.assertFalse(o['valid'][5])
        self.assertTrue(np.array_equal(np.isfinite(o['heading']),np.isfinite(o['displacement'])))
    def test_gap_rows_and_windows(self):
        f=np.r_[np.arange(100),np.arange(130,230)];m=.01*f
        table=pd.DataFrame(dict(frame=f,heading=m,displacement=m,valid=True,run_id=np.r_[np.ones(100),np.full(100,2)],fps=30))
        r=make_rows(table,'heading',10)
        self.assertTrue(((r.frame<90)|(r.frame>=160)).all())
        np.testing.assert_allclose(r.target,.01*r.target_frame)
        np.testing.assert_allclose(r.state,r.current,atol=1e-14)
        np.testing.assert_allclose(r.trend,.3,atol=1e-14)
    def test_cached_poly_equivalent_to_original(self):
        rng=np.random.default_rng(5);r=pd.DataFrame(dict(current=rng.uniform(-.9,.9,400),delayed=rng.uniform(-.9,.9,400),target=rng.uniform(-1,1,400)))
        r['state']=.9*r.current;r['trend']=r.current-r.delayed
        for family,degree in [('current_matched',11),('history',3),('state_trend',3)]:
            gram,rhs,info=prepared(r,family,degree);ridge=.01
            info.update(family=family,coefficients=np.linalg.solve(gram+ridge*np.eye(len(rhs)),rhs))
            x,y,base=xy(r,family)
            original=ParityRidgePredictor.fit(x,y,parity=[-1]*x.shape[1],maximum_degree=degree,ridge=ridge)
            np.testing.assert_allclose(predict(info,r),base+original.predict(x),rtol=1e-12,atol=1e-12)
    def test_cached_spline_equivalent(self):
        rng=np.random.default_rng(6);r=pd.DataFrame(dict(current=rng.uniform(-.9,.9,400),delayed=0.,target=rng.uniform(-1,1,400)))
        gram,rhs,info=prepared(r,'current_spline',11);ridge=.1
        info.update(family='current_spline',coefficients=np.linalg.solve(gram+ridge*np.eye(len(rhs)),rhs))
        cr=ContinuousRows(r.current.to_numpy(),r.delayed.to_numpy(),(r.target-r.current).to_numpy(),np.arange(len(r)),np.zeros(len(r)))
        old=OddSplinePredictor.fit([cr],11,ridge)
        np.testing.assert_allclose(predict(info,r),r.current.to_numpy()+old.predict(r[['current']].to_numpy()),rtol=1e-12,atol=1e-12)
    def test_real_rows_common_support(self):
        path=INPUT/'observation_frames.parquet'
        self.assertTrue(path.exists())
        t=pd.read_parquet(path)
        for (_,tid),g in t.groupby(['variant','trial_id']):
            for h in (3,10,30):
                a=make_rows(g,'heading',h);b=make_rows(g,'displacement',h)
                np.testing.assert_array_equal(a.frame,b.frame)
                valid=g.set_index('frame')
                for offset in (-30,-5,0,h):
                    np.testing.assert_array_equal(valid.loc[a.frame+offset,'run_id'],a.run_id)
                np.testing.assert_allclose(a.target,valid.loc[a.target_frame,'heading'])
                np.testing.assert_allclose(b.target,valid.loc[b.target_frame,'displacement'])
            da=discrete_rows(g,'heading');db=discrete_rows(g,'displacement')
            np.testing.assert_array_equal(da['frame'],db['frame'])

if __name__=='__main__':
    import io
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    text=stream.getvalue();print(text)
    write_json(OUT/'test_results.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()))
    raise SystemExit(0 if result.wasSuccessful() else 1)

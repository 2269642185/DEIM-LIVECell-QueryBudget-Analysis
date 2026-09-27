"""Mask-only COCO-style greedy detection and score-free maximum-cardinality audit."""
import json,gzip,csv,copy,time,os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from faster_coco_eval import COCO,COCOeval_faster
import faster_coco_eval.core.mask as mu
R=Path(os.environ["AUDIT_ROOT"]);TH=[.05,.1,.3,.5,.7];IO=[.5,.75]
def write(name,rows):
 if not rows:return
 with open(R/name,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cardinal(a):
 if not a.shape[0] or not a.shape[1]:return 0
 return int((maximum_bipartite_matching(csr_matrix(a),perm_type='column')>=0).sum())
def greedy(ious,scores,dc,gc,crowd,t):
 order=np.argsort(-np.asarray(scores),kind='stable');used=set();matches=np.full(len(scores),-1,int);ignore=np.zeros(len(scores),bool)
 # COCO: nonignored GT preferred, equal IoU resolves to later GT; crowd may repeat.
 gs=sorted(range(len(gc)),key=lambda j:bool(crowd[j]))
 for i in order:
  best=-1;v=min(t,1-1e-10)
  for j in gs:
   if dc[i]!=gc[j] or (j in used and not crowd[j]):continue
   if best>=0 and not crowd[best] and crowd[j]:break
   if ious[i,j]<v:continue
   v=ious[i,j];best=j
  if best>=0:
   matches[i]=best;ignore[i]=crowd[best]
   if not crowd[best]:used.add(best)
 return matches,ignore
def tests():
 for mat,sc,expected in [(np.eye(2),[.9,.8],2),(np.ones((2,1)),[.9,.8],1),(np.array([[.4]]),[.9],0),(np.empty((0,1)),[],0),(np.empty((1,0)),[.9],0)]:
  m,ig=greedy(mat,sc,np.zeros(len(sc)),np.zeros(mat.shape[1]),np.zeros(mat.shape[1],bool),.5);assert ((m>=0)&~ig).sum()==expected
 assert cardinal(np.array([[1,1],[1,0]],bool))==2
 assert cardinal(np.empty((0,2),bool))==0
 a=np.zeros((8,8),np.uint8);a[1:4,1:4]=1;b=np.zeros_like(a);b[5:7,5:7]=1
 ar=mu.encode(np.asfortranarray(a));br=mu.encode(np.asfortranarray(b));pixel_iou=mu.iou([ar,ar,br],[ar],[0]);assert np.allclose(pixel_iou[:,0],[1,1,0])
 mm,ii=greedy(pixel_iou,[.9,.8,.7],[0,0,0],[0],[False],.5);assert (mm>=0).sum()==1
 difficult=np.array([[.9,.51],[.51,.49]])
 mm,ii=greedy(difficult,[.9,.8],[0,0],[0,0],[False,False],.5)
 assert (mm>=0).sum()==1 and cardinal(difficult>=.5)==2
 m,ig=greedy(np.ones((2,1)),[.9,.8],[0,0],[0],[True],.5);assert ig.sum()==2
 print('SYNTHETIC_TESTS_PASSED',flush=True)
def metrics(rows):
 g=sum(x['gt'] for x in rows);tp=sum(x['tp'] for x in rows);fp=sum(x['fp'] for x in rows);fn=g-tp;n=len(rows)
 return dict(images=n,gt=g,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/g if g else None,mean_image_recall=float(np.mean([x['tp']/x['gt'] for x in rows if x['gt']])) if g else None,fn_per_image=fn/n,fp_per_image=fp/n,candidate_coverage_recall=sum(x['candidate_count'] for x in rows)/g if g else None)
def main():
 tests();man=json.loads((R/'manifest.json').read_text());first=next(iter(man['models'].values()));models={};coco=COCO(first['annotation']);ids=sorted(coco.imgs)
 per=[];pr=[];repro=[];gtrows={};budget=[];coverage=[];score_events={};observed=[]
 for name in ['r18_300','r18_400','r18_500','r18_600']:
  deadline=time.monotonic()+3600
  while True:
   live=json.loads((R/'manifest.json').read_text());info=live['models'].get(name)
   if any(x['model']==name for x in live['missing']):break
   if info and info.get('complete'):break
   if time.monotonic()>deadline:raise RuntimeError('Cache did not complete '+name)
   time.sleep(5)
  if any(x['model']==name for x in live['missing']):continue
  models[name]=info
  assert sorted(info['image_ids'])==ids;assert info['annotation_sha256']==first['annotation_sha256']
  orig=[];diag=[];events={t:[] for t in IO};budgetvals=[];dups=[]
  for ii,iid in enumerate(ids):
   with gzip.open(R/'cache'/name/f'{iid}.json.gz','rt') as f:d=json.load(f)
   gt=coco.imgToAnns[iid];crowd=np.array([bool(a.get('iscrowd',0)) for a in gt]);gc=np.array([a['category_id'] for a in gt]);valid=~crowd;ng=int(valid.sum())
   # Existing faster-coco ignores iscrowd; standalone ignore is overwritten. Record any such anomaly.
   assert all(not a.get('ignore',0) or a.get('iscrowd',0) for a in gt),'standalone ignore requires explicit protocol decision'
   gr=[coco.annToRLE(a) for a in gt];ious=mu.iou(d['rles'],gr,crowd.astype(int).tolist()) if len(gt) else np.empty((info['queries'],0));assert np.isfinite(ious).all()
   scores=np.array(d['scores']);dc=np.array(d['labels']);mapping={q:k for k,q in enumerate(d['query_ids'])};dups.append(len(d['original_query_ids'])-len(set(d['original_query_ids'])))
   for k,r in enumerate(d['rles']):diag.append(dict(image_id=iid,category_id=int(dc[k]),score=float(scores[k]),segmentation=r))
   for q,c,s in zip(d['original_query_ids'],d['original_labels'],d['original_scores']):orig.append(dict(image_id=iid,category_id=c,score=s,segmentation=d['rles'][mapping[q]]))
   group='gt_le_300' if ng<=300 else 'gt_301_500' if ng<=500 else 'gt_gt_500';gap=max(ng-info['queries'],0);budgetvals.append(gap);budget.append(dict(model=name,image_id=iid,gt=ng,query_budget=info['queries'],budget_gap=gap))
   for t in IO:
    # Primary candidate metric is class-agnostic mask geometry. Add class-aware for diagnosis.
    cc=cardinal(ious[:,valid]>=t);cca=cardinal((ious[:,valid]>=t)&(dc[:,None]==gc[valid][None,:]))
    coverage.append(dict(model=name,image_id=iid,iou=t,gt=ng,candidate_count=cc,class_aware_candidate_count=cca))
    matched,ignored=greedy(ious,scores,dc,gc,crowd,t)
    # Reproduce evaluator category scope: annotation category_ids, not all 8 logit channels.
    ignored=ignored|~np.isin(dc,coco.getCatIds())
    for k in np.argsort(-scores,kind='stable'):
     if not ignored[k]:events[t].append((float(scores[k]),int(matched[k]>=0),iid))
    for threshold in TH:
     take=(scores>=threshold)&~ignored;hit=set(matched[take&(matched>=0)].tolist());tp=len(hit);fp=int(take.sum())-tp;fn=ng-tp
     assert tp+fn==ng and tp+fp==int(take.sum()) and tp<=min(ng,int(take.sum()))
     row=dict(model=name,protocol='diagnostic_one_class_per_query',image_id=iid,density_group=group,iou=t,threshold=threshold,gt=ng,tp=tp,fp=fp,fn=fn,candidate_count=cc,class_aware_candidate_count=cca,budget_gap=gap,excluded_out_of_scope_category=int(((scores>=threshold)&~np.isin(dc,coco.getCatIds())).sum()))
     per.append(row)
     if threshold==.5:
      for j,a in enumerate(gt):
       if crowd[j]:continue
       key=(iid,a['id'],t)
       if key not in gtrows:gtrows[key]=dict(image_id=iid,gt_id=a['id'],iou=t,density_group=group,image_gt_count=ng,area_original_pixels_squared=a['area'],mask_area_original_pixels_squared=int(mu.area(gr[j])),category_id=a['category_id'])
       gtrows[key][name+'_detected']=int(j in hit)
   if ii%100==0:print('ANALYSIS',name,ii,'images',flush=True)
  for t,ev in events.items():
   ev.sort(key=lambda x:-x[0]);tp=np.cumsum([e[1] for e in ev]);fp=np.arange(1,len(ev)+1)-tp;ng=sum(r['gt'] for r in per if r['model']==name and r['iou']==t and r['threshold']==.5)
   for k,e in enumerate(ev):pr.append(dict(model=name,iou=t,rank=k+1,score=e[0],tp=int(tp[k]),fp=int(fp[k]),precision=float(tp[k]/(k+1)),recall=float(tp[k]/ng)))
   # Only ends of score ties are achievable with a scalar threshold.
   for p in [.8,.9]:
    eligible=[k for k,e in enumerate(ev) if (k==len(ev)-1 or ev[k+1][0]!=e[0]) and tp[k]/(k+1)>=p]
    k=max(eligible,key=lambda k:tp[k]) if eligible else None
    observed.append(dict(model=name,iou=t,target_precision=p,recall_at_precision=float(tp[k]/ng) if k is not None else 'NA',achieved_precision=float(tp[k]/(k+1)) if k is not None else 'NA',score=float(ev[k][0]) if k is not None else 'NA',method='maximum micro recall at attainable complete-score-tie operating point with precision >= target'))
  for protocol,preds in [('original',orig),('diagnostic_one_class_per_query',diag)]:
   dt=coco.loadRes(copy.deepcopy(preds));e=COCOeval_faster(coco,dt,'segm',print_function=print);e.params.maxDets=[1,10,100,3000];e.evaluate();e.accumulate();e.summarize()
   # Read precision tensor explicitly at largest maxDet, all areas, across categories.
   p=e.eval['precision'];a=p[:,:,:,0,-1];ap=float(a[a>=0].mean());a50=p[0,:,:,0,-1];ap50=float(a50[a50>=0].mean())
   repro.append(dict(model=name,protocol=protocol,mask_ap=ap,mask_ap50=ap50,max_dets=3000,query_count=info['queries'],original_duplicate_query_count=sum(dups)))
   if protocol.startswith('diagnostic'):
    # independent greedy agrees with evaluator's full-ranking micro final recall after category weighting
    rr=e.eval['recall'][:,:,0,-1]
    for ti,t in enumerate(IO):
     idx=int(np.argmin(abs(e.params.iouThrs-t)));counts=np.array([sum(not a.get('iscrowd',0) for a in coco.anns.values() if a['category_id']==c) for c in e.params.catIds]);expected=float((rr[idx]*counts).sum()/counts.sum());actual=sum(x[1] for x in events[t])/counts.sum();assert abs(expected-actual)<1e-8,(name,t,expected,actual)
  write('reproduction.csv',repro);write('per_image.csv',per);write('budget_per_image.csv',budget)
  print('MODEL_COMPLETE',name,flush=True)
 write('candidate_coverage.csv',coverage);write('pr_curve.csv',pr);write('recall_at_precision.csv',observed)
 summary=[];density=[]
 for name in models:
  for t in IO:
   for th in TH:
    rr=[x for x in per if x['model']==name and x['iou']==t and x['threshold']==th];summary.append(dict(model=name,iou=t,threshold=th,**metrics(rr)))
    if th==.5:
     for group in ['gt_le_300','gt_301_500','gt_gt_500']:density.append(dict(model=name,iou=t,threshold=th,density_group=group,**metrics([x for x in rr if x['density_group']==group])))
 write('threshold_metrics.csv',summary);write('density_summary.csv',density)
 paircounts=[]
 for row in gtrows.values():
  a=row.get('r18_300_detected');b=row.get('r18_500_detected');row['r18_pair']='both' if a and b else 'only500' if b else 'only300' if a else 'neither'
 write('per_gt.csv',list(gtrows.values()))
 for t in IO:
  for label in ['both','only500','only300','neither']:paircounts.append(dict(iou=t,pair=label,count=sum(x['iou']==t and x['r18_pair']==label for x in gtrows.values())))
 write('paired_gt_summary.csv',paircounts)
 boot=[];rng=np.random.default_rng(20260904)
 for t in IO:
  aa=sorted([x for x in per if x['model']=='r18_300' and x['iou']==t and x['threshold']==.5],key=lambda x:x['image_id']);bb=sorted([x for x in per if x['model']=='r18_500' and x['iou']==t and x['threshold']==.5],key=lambda x:x['image_id']);assert [x['image_id'] for x in aa]==[x['image_id'] for x in bb]
  g=np.array([x['gt'] for x in aa]);delta=np.array([b['tp']-a['tp'] for a,b in zip(aa,bb)]);idx=rng.integers(0,len(g),(2000,len(g)));values=delta[idx].sum(1)/g[idx].sum(1);lo,hi=np.quantile(values,[.025,.975]);boot.append(dict(comparison='r18_500_minus_r18_300',iou=t,threshold=.5,delta_micro_recall=float(delta.sum()/g.sum()),ci95_low=lo,ci95_high=hi,replicates=2000,seed=20260904))
 write('paired_bootstrap.csv',boot)
 diffs=[]
 for t in IO:
  for group in ['gt_le_300','gt_301_500','gt_gt_500']:
   a=next(x for x in density if x['model']=='r18_300' and x['iou']==t and x['density_group']==group);b=next(x for x in density if x['model']=='r18_500' and x['iou']==t and x['density_group']==group)
   diffs.append(dict(iou=t,density_group=group,**{k:b[k]-a[k] for k in ['precision','recall','mean_image_recall','fn_per_image','fp_per_image','candidate_coverage_recall']}))
 write('density_differences.csv',diffs)
 bs=[]
 for name in models:
  a=np.array([x['budget_gap'] for x in budget if x['model']==name]);g=np.array([x['gt'] for x in budget if x['model']==name]);bs.append(dict(model=name,images=len(a),gt=int(g.sum()),mean_gt=float(g.mean()),median_gt=float(np.median(g)),max_gt=int(g.max()),budget_gap_sum=int(a.sum()),budget_gap_mean=float(a.mean()),budget_gap_median=float(np.median(a)),affected_images=int((a>0).sum()),affected_image_fraction=float((a>0).mean()),budget_gap_fraction_of_gt=float(a.sum()/g.sum())))
 write('budget_summary.csv',bs)
 small=[]
 for t in IO:
  for group in ['gt_le_300','gt_301_500','gt_gt_500']:
   for label in ['both','only500','only300','neither']:
    rows=[x for x in gtrows.values() if x['iou']==t and x['density_group']==group and x['r18_pair']==label];area=[x['mask_area_original_pixels_squared'] for x in rows]
    small.append(dict(iou=t,density_group=group,pair=label,instances=len(area),median_mask_area_original_pixels_squared=float(np.median(area)) if area else None,mean_mask_area_original_pixels_squared=float(np.mean(area)) if area else None))
 write('exploratory_size_within_density.csv',small)
 print('ALL_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()

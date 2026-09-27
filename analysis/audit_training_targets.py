"""Read-only CPU, full-training-set equality audit using the actual loader branch."""
import ast,collections,csv,hashlib,json,os,sys,time,inspect
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

import os
ROOT=Path(os.environ["MODEL_PROJECT"])
sys.path.insert(0,str(ROOT))
OUT=ROOT/'p0_audit'/'training_target_audit'

class Stream:
 def __init__(self,f):self.f=f;self.buf='';self.pos=0;self.eof=False;self.decoder=json.JSONDecoder()
 def more(self):
  self.buf=self.buf[self.pos:];self.pos=0;s=self.f.read(131072);self.buf+=s;self.eof=not s
 def peek(self):
  while True:
   while self.pos<len(self.buf) and self.buf[self.pos].isspace():self.pos+=1
   if self.pos<len(self.buf):return self.buf[self.pos]
   if self.eof:raise EOFError('Unexpected end of JSON')
   self.more()
 def eat(self,s):
  assert self.peek()==s,(self.peek(),s);self.pos+=1
 def value(self):
  self.peek()
  while True:
   try:v,self.pos=self.decoder.raw_decode(self.buf,self.pos);return v
   except json.JSONDecodeError:
    if self.eof:raise
    if len(self.buf)-self.pos>16777216:raise ValueError('Single JSON value exceeded streaming limit')
    self.more()

def items(path,key):
 with open(path,encoding='utf-8') as f:
  s=Stream(f);s.eat('{')
  while s.peek()!='}':
   k=s.value();s.eat(':')
   if s.peek()=='[':
    s.eat('[')
    while s.peek()!=']':
     v=s.value()
     if k==key:yield v
     if s.peek()==',':s.eat(',')
     else:break
    s.eat(']')
   else:s.value()
   if k==key:return
   if s.peek()==',':s.eat(',')
   else:break
 raise KeyError(key)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def branch_from_source(path):
 tree=ast.parse(path.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ConvertCocoPolysToMask');method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
 branch=next(n for n in method.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith('grid_size > 0'))
 fn=ast.FunctionDef(name='dynamic_target',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ['anno','w','h','grid_size']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=branch.body+[ast.Return(value=ast.Name(id='mask_grids',ctx=ast.Load()))],decorator_list=[])
 mod=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={'torch':torch};exec(compile(mod,str(path),'exec'),scope)
 return scope['dynamic_target'],ast.unparse(mod)

def main():
 torch.set_num_threads(1);cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
 man=json.loads((ROOT/'p0_audit/manifest.json').read_text());configs={k:v['resolved_config'] for k,v in man['models'].items() if k in ['r18_300','r18_500']}
 assert len(configs)==2
 from engine.core import YAMLConfig
 for k in configs:assert YAMLConfig(str(ROOT/man['models'][k]['config'])).yaml_cfg==configs[k],'Expanded config drift since P0: '+k
 datasets={k:v['train_dataloader']['dataset'] for k,v in configs.items()};assert len({v['ann_file'] for v in datasets.values()})==1;assert len({v['img_folder'] for v in datasets.values()})==1
 ann_path=ROOT/next(iter(datasets.values()))['ann_file'];img_dir=ROOT/next(iter(datasets.values()))['img_folder'];before=ann_path.stat()
 source=ROOT/'engine/data/dataset/coco_dataset.py';dynamic,extracted=branch_from_source(source);(OUT/'extracted_dynamic_branch.py').write_text(extracted)
 hashes={k:sha(ROOT/k) for k in ['engine/data/dataset/coco_dataset.py','engine/core/workspace.py','engine/data/dataloader.py','engine/data/transforms/_transforms.py','engine/deim/deim_criterion.py']}
 assert all(h==man['source_sha256'][k] for k,h in hashes.items()),'Source drift since P0; re-audit before comparison'
 print('HASHING_TRAINING_JSON',flush=True);ann_sha=sha(ann_path)
 images={i['id']:i for i in items(ann_path,'images')};image_rows=[];actual_sizes={}
 for iid,im in images.items():
  p=img_dir/im['file_name']
  with Image.open(p) as image:w,h=image.size
  actual_sizes[iid]=(w,h);image_rows.append({'image_id':iid,'file_name':im['file_name'],'actual_width':w,'actual_height':h,'annotation_width':im['width'],'annotation_height':im['height'],'size_equal':int((w,h)==(im['width'],im['height']))})
 with open(OUT/'image_sizes.csv','w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=image_rows[0]);wr.writeheader();wr.writerows(image_rows)
 from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask,CocoDetection
 dataset_default=inspect.signature(CocoDetection.__init__).parameters['mask_grid_size'].default
 assert dataset_default==0
 legacy=ConvertCocoPolysToMask(False,0);online=ConvertCocoPolysToMask(False,16)
 counts=collections.Counter();per_image=collections.defaultdict(collections.Counter);lengths=collections.Counter();parts=collections.Counter();seen=set();examples=[];start=time.monotonic();converter_checks=0
 with open(OUT/'differences.csv','w',newline='') as differences:
  writer=csv.DictWriter(differences,fieldnames=['annotation_id','image_id','reason','different_pixels','max_abs_difference','polygon_count']);writer.writeheader()
  for a in items(ann_path,'annotations'):
   iid=a['image_id'];aid=a['id'];assert aid not in seen,'Duplicate annotation ID';seen.add(aid);w,h=actual_sizes[iid];pc=len(a['segmentation']) if isinstance(a.get('segmentation'),list) else -1
   counts['annotations']+=1;per_image[iid]['annotations']+=1;parts[str(pc)]+=1
   has='mask_grid' in a;counts['with_mask_grid_all' if has else 'missing_mask_grid_all']+=1
   if has:lengths[str(len(a['mask_grid']))]+=1
   if a.get('iscrowd',0):counts['excluded_crowd']+=1;continue
   b=np.array(a['bbox'],dtype=np.float32);b[2:]+=b[:2];b[0::2]=np.clip(b[0::2],0,w);b[1::2]=np.clip(b[1::2],0,h)
   if not (b[2]>b[0] and b[3]>b[1]):counts['excluded_invalid_bbox']+=1;continue
   counts['training_instances']+=1;per_image[iid]['training_instances']+=1;reason='';diff='';maxabs=''
   if not has:reason='missing_mask_grid'
   else:
    old=np.asarray(a['mask_grid'],dtype=np.float32)
    if old.shape!=(256,):reason='invalid_grid_shape'
    elif not np.isfinite(old).all():reason='nonfinite_grid'
    else:
     if not np.isin(old,[0.,1.]).all():counts['nonbinary_stored_grid']+=1
     try:new=dynamic([a],w,h,16).numpy()[0]
     except Exception as e:
      reason='dynamic_error:'+type(e).__name__
      if len(examples)<100:examples.append({'annotation_id':aid,'image_id':iid,'error':str(e)})
     else:
      assert new.shape==(256,) and np.isfinite(new).all()
      counts['compared']+=1;per_image[iid]['compared']+=1;diff=int((old!=new).sum());maxabs=float(np.max(np.abs(old-new)))
      if diff:reason='target_mismatch';counts['different_pixels']+=diff;per_image[iid]['different_pixels']+=diff
      else:counts['identical']+=1;per_image[iid]['identical']+=1
      if converter_checks<64:
       blank=Image.new('RGB',(w,h));_,l=legacy(blank,{'image_id':iid,'annotations':[a]});_,r=online(blank,{'image_id':iid,'annotations':[a]})
       assert np.array_equal(l['mask_grid'].numpy()[0],old);assert np.array_equal(r['mask_grid'].numpy()[0],new);assert torch.equal(l['boxes'],r['boxes']);converter_checks+=1
   if reason:
    counts[reason]+=1;per_image[iid][reason]+=1;writer.writerow(dict(annotation_id=aid,image_id=iid,reason=reason,different_pixels=diff,max_abs_difference=maxabs,polygon_count=pc))
    if len(examples)<100:examples.append({'annotation_id':aid,'image_id':iid,'reason':reason,'different_pixels':diff,'max_abs_difference':maxabs,'polygon_count':pc})
   if counts['annotations']%25000==0:print('PROGRESS',counts['annotations'],'compared',counts['compared'],'mismatch',counts['target_mismatch'],'seconds',round(time.monotonic()-start,1),flush=True)
 after=ann_path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Training annotation changed during audit'
 report={'status':'complete','runtime_dataset_mask_grid_size_default':dataset_default,'training_annotation':str(ann_path),'training_annotation_sha256':ann_sha,'annotation_bytes':before.st_size,'annotation_mtime_ns':before.st_mtime_ns,'source_sha256':hashes,'precompute_script_sha256':sha(ROOT/'precompute_binary_mask.py') if (ROOT/'precompute_binary_mask.py').exists() else None,'script_sha256':sha(Path(__file__)),'opencv':cv2.__version__,'numpy':np.__version__,'torch':torch.__version__,'images':len(images),'image_dimension_mismatches':sum(not x['size_equal'] for x in image_rows),'counts':dict(counts),'stored_grid_lengths':dict(lengths),'segmentation_polygon_counts':dict(parts),'actual_converter_checks':converter_checks,'comparison':'Exact float32 equality of every stored grid pixel against current loader dynamic branch for all noncrowd valid-box annotations; no tolerance','elapsed_seconds':time.monotonic()-start,'examples':examples,'historical_limitation':'Current server training file and P0 source snapshot checked; historical training-time file identity cannot be proven without archived hashes/snapshots.'}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 keys=sorted({k for c in per_image.values() for k in c})
 with open(OUT/'per_image.csv','w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['image_id']+keys);writer.writeheader()
  for iid in sorted(images):writer.writerow({'image_id':iid,**{k:per_image[iid][k] for k in keys}})
 print('AUDIT_COMPLETE',json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

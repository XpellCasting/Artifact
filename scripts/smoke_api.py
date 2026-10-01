"""Smoke HTTP: usar únicamente contra una instalación de prueba desechable.

Carga datos sintéticos, analiza y guarda/ejecuta un flujo de 144 observaciones.
Requiere un modelo v2 activo. No usar contra la instalación de trabajo.
"""
import json, urllib.request, urllib.error, math, time
from datetime import datetime, timedelta, timezone
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',default='http://127.0.0.1:18000')
parser.add_argument('--disposable',action='store_true',required=True,help='Confirmar que la API usa datos desechables')
base=parser.parse_args().base.rstrip('/')
def call(path,data=None):
    raw=data if isinstance(data,bytes) else json.dumps(data).encode() if data is not None else None
    try:
        with urllib.request.urlopen(urllib.request.Request(base+path,data=raw,headers={'Content-Type':'text/csv' if isinstance(data,bytes) else 'application/json'}),timeout=240) as r: return json.load(r)
    except urllib.error.HTTPError as e: raise RuntimeError(path+' '+e.read().decode()) from e
def csv(start,count):
    rows=['entity_id,state,last_changed']
    for i in range(count):
        at=(start+timedelta(minutes=5*i)).isoformat()
        rows += [f'sensor.office_sensor_temperature,{21+math.sin(i/12)+(40 if i%47==0 else 0)},{at}',f'binary_sensor.door_office,{"on" if i%8<2 else "off"},{at}']
    return '\n'.join(rows).encode()
print('models',json.dumps(call('/diagnostics/classifier-models'))[:1000],flush=True)
print('upload',json.dumps(call('/history/real/upload',csv(datetime(2026,9,1,tzinfo=timezone.utc),576)))[:1200],flush=True)
print('analyze',str(call('/history/analyze',{}))[:400],flush=True)
print('future',str(call('/validation/replay/upload',csv(datetime(2026,9,4,tzinfo=timezone.utc),72)))[:500],flush=True)
f=call('/flows')['default']
f['name']='Refactor Docker smoke'
f['edges']=[{'id':f'{a}-{b}','source':a,'target':b} for a,b in zip(['input','statistical','contextual','classifier'],['statistical','contextual','classifier','output'])]
f=call('/flows',f)
print('saved flow',f['flow_id'],flush=True)
job=call('/flows/'+f['flow_id']+'/runs',{})
print('job',job,flush=True)
for _ in range(180):
    job=call('/flow-evaluations/'+job['job_id'])
    if job['status'] in ('completed','failed','error'): break
    time.sleep(1)
print('completed',job['status'],job.get('processed'),flush=True)
assert job['status']=='completed',job
results=call('/flow-runs/'+job['run']['run_id']+'/results?limit=3')
assert results['total']==144 and len(results['items'])==3
print('persisted results:',results['total'],flush=True)
print('node counts:',job['run']['summary']['node_counts'])


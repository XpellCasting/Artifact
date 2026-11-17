"""Trazabilidad de etiquetas, reconstrucción de entradas y sensibilidad categórica."""
import argparse, csv, json, sqlite3, sys, itertools
from pathlib import Path
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from copy import deepcopy


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    root=a.workspace; previous=root/'outputs/incorporacion_main_C_41'; out=root/'outputs/auditoria_causas_41';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(previous/'codigo_main/backend'))
    import numpy as np
    import pandas as pd
    import torch
    from app.modulos.clasificador.training import load_candidate,_tensor_batch
    from app.modulos.clasificador.config import Config,parse_utc
    from app.modulos.clasificador.temporal import build_causal_sample
    from app.modulos.clasificador.segments import BASE_NAMES,MAX_POINTS,MAX_SEGMENTS
    from app.modulos.clasificador.model_constants import KIND_TO_ID
    torch.set_num_threads(2)
    model,prep,graph,manifest=load_candidate('4d238952b2af4730a52c1c1a355f738c',previous/'modelo_compatible')
    cfg=Config(**manifest['config']); cases=json.loads((previous/'predicciones.json').read_text());profiles=json.loads((previous/'referencias_nuevas.json').read_text())
    db=sqlite3.connect(f'file:{(a.experiment/"observations.sqlite3").as_posix()}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    histories={e:[dict(r) for r in db.execute('SELECT row_id,t,state FROM observations WHERE entity=? ORDER BY t,row_id',(e,))] for e in profiles}
    timelines={e:[parse_utc(r['t']) for r in rows] for e,rows in histories.items()}
    def provider(start,end,entities):
        answer=[]
        for e in entities:
            lo=bisect_left(timelines[e],start);hi=bisect_right(timelines[e],end)
            answer += [dict(entity_id=e,timestamp=r['t'],state=r['state'],source_row=r['row_id']) for r in histories[e][max(0,lo-1):hi]]
        assert all(parse_utc(r['timestamp'])<=end for r in answer)
        return answer
    traces=[];features=[];sensitivity=[];arrays={}; reproduce=0
    for c in cases:
        e=c['entity_id'];target=parse_utc(c['timestamp']);info=profiles[e]['entity']
        newprep={**prep,'entity_ids':[e],'entities':{e:info},'entity_kinds':{e:info['kind']}}
        newgraph={**graph,'entity_ids':[e],'adjacency':[[True]]}
        sample=build_causal_sample(provider,target,newprep,newgraph,cfg)
        x,states,present,_,_,adj=_tensor_batch([sample],'cpu');kinds=torch.tensor([KIND_TO_ID[info['kind']]])
        arrays[f'X_{c["id"]}']=x.numpy();arrays[f'present_{c["id"]}']=present.numpy()
        with torch.no_grad(): base=model(x,states,adj,present,kinds)[0][0,0].softmax(-1).numpy()
        expected=list(c['prediction']['class_scores'][k] for k in ['normal','environmental_anomaly','sensor_fault'])
        assert np.allclose(base,expected,atol=1e-6);reproduce+=1
        events=provider(target-__import__('datetime').timedelta(hours=72),target,[e])
        last=events[-1]
        for scale,coverage in enumerate(sample['metadata']['scale_coverage'][e]):
            sl=x[0,scale*MAX_SEGMENTS:(scale+1)*MAX_SEGMENTS,0];mask=present[0,scale*MAX_SEGMENTS:(scale+1)*MAX_SEGMENTS,0].bool()
            used=sl[mask]
            row={'id':c['id'],'sensor':e,'scale':scale,'target_state':last['state'],**coverage}
            row.update({name:float(used[:,j].mean()) if len(used) else None for j,name in enumerate(BASE_NAMES)})
            features.append(row)
        traces.append({'id':c['id'],'sensor':e,'timestamp':c['timestamp'],'target_state':last['state'],
                       'states_72h':dict(Counter(r['state'] for r in events)), 'vocab':info['vocab'],
                       'hourly_profile_available':info['hourly_profile']['available'],'rows_72h':len(events)})
        offset=len(BASE_NAMES)+2*MAX_POINTS
        categories=x[:,:,:,offset:offset+MAX_POINTS]
        for perm in itertools.permutations([3,4,5]):
            modified=x.clone();dest=modified[:,:,:,offset:offset+MAX_POINTS]
            for old,new in zip([3,4,5],perm): dest[categories==old]=new
            with torch.no_grad(): score=model(modified,states,adj,present,kinds)[0][0,0].softmax(-1).numpy()
            sensitivity.append({'id':c['id'],'sensor':e,'permutation':str(perm),'p_normal':float(score[0]),'p_ambiental':float(score[1]),'p_falla':float(score[2]),'argmax':int(score.argmax()),'max_delta':float(np.abs(score-base).max())})
    np.savez_compressed(out/'entradas_41.npz',**arrays)
    pd.DataFrame(features).to_csv(out/'caracteristicas_por_escala.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'sensibilidad_codigos.csv',index=False)
    (out/'trazas_entrada.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2))
    # Check the two original exports against identity, never row position alone.
    evidence=json.loads((previous/'trazabilidad_etiquetas.json').read_text())
    keys={(r['entity'],parse_utc(r['t']),r['state']) for r in evidence}
    export_stats=[]
    for filename in ['1046e6137b1a424a2f84f2c0f9d4abe0.csv','a22f360715b0694e7a446858e64fd6aa.csv']:
        matching=[]
        with (root/'Artifact/data/review/sources'/filename).open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if (r['entity_id'],parse_utc(r['last_changed']),r['state']) in keys:matching.append(r)
        export_stats.append({'file':filename,'matched':len(matching),'labels':dict(Counter(r['tipo'] for r in matching))})
    # Look for independently documented annotations covering these exact events.
    annotations=[]
    for path in (root/'Artifact/data/review/scenarios').glob('*/metadata.json'):
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        for ann in data.get('annotation_snapshots',[]):
            if ann.get('entity_id') not in profiles:continue
            hits=[r['row_id'] for r in evidence if ann['entity_id']==r['entity'] and parse_utc(ann['start'])<=parse_utc(r['t'])<=parse_utc(ann['end'])]
            if hits:annotations.append({'file':str(path),'matching_rows':hits,'annotation':ann})
    (out/'anotaciones_coincidentes.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2))
    training_counts=Counter();files=[]
    for path in (a.experiment/'sequences').glob('*train*metadata.jsonl'):
        n=0
        for line in path.open():
            r=json.loads(line);e=r.get('target_entity_id');kind=prep['entities'].get(e,{}).get('kind','fuera_catalogo')
            training_counts[(kind,str(r.get('target_label')))]+=1;n+=1
        files.append({'file':path.name,'rows':n})
    train_rows=[{'kind':k[0],'label':k[1],'targets':n} for k,n in training_counts.items()]
    pd.DataFrame(train_rows).to_csv(out/'soporte_entrenamiento_tipo_clase.csv',index=False)
    summary={'baseline_reproduced':reproduce,'exports':export_stats,'matching_annotation_records':len(annotations),
             'training_metadata_files':files,'training_targets_by_kind_and_label':train_rows,
             'categorical_training_vocab':{e:i['vocab'] for e,i in prep['entities'].items() if i['kind']!='continuous'},
             'sensitivity_argmax_counts':dict(Counter(str(r['argmax']) for r in sensitivity)),
             'sensitivity_max_absolute_change':max(r['max_delta'] for r in sensitivity),
             'sensitivities_are_counterfactual_not_new_evaluation':True}
    (out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()

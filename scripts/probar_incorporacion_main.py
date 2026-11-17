"""Prueba aislada de incorporación causal de los 41 objetivos ambientales nuevos."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--database', type=Path, required=True)
    args = p.parse_args()
    root = args.workspace
    out = root/'outputs/incorporacion_main_C_41'
    out.mkdir(parents=True, exist_ok=True)
    codezip = root/'tmp/main_onboarding_snapshot.zip'
    modelzip = root/'outputs/Modelo_C_20261030_importar_compatible.zip'
    for source, target in [(codezip, out/'codigo_main'), (modelzip, out/'modelo_compatible')]:
        with zipfile.ZipFile(source) as archive:
            archive.extractall(target)
    sys.path.insert(0, str(out/'codigo_main/backend'))
    import torch
    torch.set_num_threads(2)
    from app.modulos.clasificador.training import load_candidate
    from app.modulos.clasificador.runtime import predict_at
    from app.modulos.clasificador.config import parse_utc
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    identifier = '4d238952b2af4730a52c1c1a355f738c'
    snapshot = load_candidate(identifier, out/'modelo_compatible')
    model, prep, graph, manifest = snapshot
    def weight_digest():
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before = weight_digest()
    with zipfile.ZipFile(root/'outputs/Modelo_C_20261030_importar.zip') as archive:
        original_weights = archive.read(f'weights/{identifier}.pt')
    assert hashlib.sha256(original_weights).hexdigest() == manifest['weights_sha256']
    source = root/'outputs/abstenciones_ambientales_C_20261030/registros_ambientales_originales.jsonl'
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    cases = sorted([r for r in cases if not r['evaluable']], key=lambda r:(parse_utc(r['context_end']), r['sample_id']))
    assert len(cases) == 41 and all(r['scenario_id']=='base-test' and r['true_label']==1 for r in cases)
    entities = sorted({r['target_entity_id'] for r in cases})
    assert not set(entities).intersection(prep['entity_ids'])
    db = sqlite3.connect(f'file:{args.database.as_posix()}?mode=ro', uri=True)
    histories = {}
    for entity in entities:
        # Only observed values are input; no label/audit table is read.
        histories[entity] = sorted([{'entity_id':entity, 'timestamp':t, 'state':state, 'source_row':rowid}
                                   for rowid,t,state in db.execute('SELECT row_id,t,state FROM observations WHERE entity=?', (entity,))],
                                  key=lambda r:parse_utc(r['timestamp']))
    db.close()
    timelines = {e:[parse_utc(r['timestamp']) for r in rows] for e,rows in histories.items()}
    cutoff = None
    calls = []
    def provider(start,end,requested):
        assert end <= cutoff, 'Consulta al futuro'
        result = []
        for entity in requested:
            rows = histories.get(entity, [])
            times = timelines.get(entity, [])
            left,right = bisect_left(times,start),bisect_right(times,end)
            result.extend(rows[max(0,left-1):right])
        assert all(parse_utc(r['timestamp'])<=end for r in result)
        calls.append({'start':start.isoformat(),'end':end.isoformat(),'target':cutoff.isoformat(),'entities':requested,'rows':len(result)})
        return result
    profiles, results = {}, []
    for i, case in enumerate(cases,1):
        cutoff = parse_utc(case['context_end'])
        cache = {}
        prediction = predict_at(snapshot,provider,case['target_entity_id'],case['context_end'],profiles,prediction_cache=cache)
        if prediction.get('onboarding'):
            assert parse_utc(prediction['onboarding']['reference_end']) < cutoff
        raw_types = next((v['type_scores_by_node'][0] for v in cache.values() if v.get('type_scores_by_node') is not None), None)
        results.append({'id':i,'sample_id':case['sample_id'],'entity_id':case['target_entity_id'],
                        'timestamp':case['context_end'],'true_label':'environmental_anomaly',
                        'label_certainty':case['label_certainty'],'label_origin':case['label_origin'],
                        'prediction':prediction,'raw_type_scores':dict(zip(FAULT_TYPES,raw_types)) if raw_types else {}})
        print(f'{i}/41 {prediction["status"]}',flush=True)
    assert weight_digest() == before, 'Pesos modificados'
    (out/'predicciones.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'referencias_nuevas.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'consultas_causales.json').write_text(json.dumps(calls,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style='white',font='DejaVu Sans')
    rows=[]
    for r in results:
        pred=r['prediction']; scores=pred.get('class_scores',{})
        rows.append({'id':r['id'],'sensor':r['entity_id'],'instante_utc':r['timestamp'],
                     'tipo_inferido':profiles.get(r['entity_id'],{}).get('sensor_kind'),
                     'p_normal':scores.get('normal'),'p_ambiental':scores.get('environmental_anomaly'),
                     'p_falla':scores.get('sensor_fault'),'clase_preferida':max(scores,key=scores.get) if scores else None,
                     'diagnostico':pred.get('predicted_label'),'tipo_falla_publicado':pred.get('fault_type'),
                     'motivos':';'.join(pred.get('reasons',[])), 'sample_id':r['sample_id']})
    frame=pd.DataFrame(rows).set_index('id')
    frame.to_csv(out/'puntuaciones_41.csv',encoding='utf-8-sig')
    for name, data, title in [
        ('puntuaciones_clases',frame[['p_normal','p_ambiental','p_falla']].rename(columns={'p_normal':'Normal','p_ambiental':'Anomalía ambiental','p_falla':'Falla'}),'C sin reentrenar · incorporación de entidades con main'),
        ('puntuaciones_tipos',pd.DataFrame([r['raw_type_scores'] for r in results],index=frame.index),'Puntuaciones de tipos · salida interna, no diagnóstico aceptado')]:
        fig,ax=plt.subplots(figsize=(10 if name.endswith('clases') else 13,16))
        sns.heatmap(data,annot=True,fmt='.1%',vmin=0,vmax=1,cmap='Blues',ax=ax,yticklabels=True)
        ax.set(title=title,ylabel='ID en puntuaciones_41.csv',xlabel='')
        ax.tick_params(axis='y',rotation=0)
        fig.text(.02,.01,'41 objetivos etiquetados como ambientales (etiquetas estimadas).\nPuntuaciones softmax sin calibración; blanco = no evaluable. Referencias causales, pesos originales.',fontsize=9)
        fig.tight_layout(rect=(0,.045,1,1));fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    summary={'model_id':identifier,'main_commit':'8975e3aabfbde5f35ed7a20f845c209c35334858',
             'main_snapshot_sha256':sha(codezip),'model_zip_sha256':sha(modelzip),'weights_sha256':manifest['weights_sha256'],
             'weights_unchanged':True,'database_sha256':sha(args.database),'cases_source_sha256':sha(source),
             'policy':manifest['rejection_policy'],'total':len(results),'evaluable':int(frame.p_normal.notna().sum()),
             'preferred_class':dict(Counter(frame.clase_preferida.fillna('no_evaluable'))),
             'accepted_diagnosis':dict(Counter(frame.diagnostico.fillna('abstencion'))),
             'inferred_kinds':{e:p['sensor_kind'] for e,p in profiles.items()},
             'reference_policy':'Perfil congelado por entidad desde su primer objetivo de esta prueba; no reproducción de todas las lecturas previas.',
             'limits':['Misma selección de 41 objetivos; etiquetas estimadas, no episodios independientes.',
                       'Referencias construidas con pasado sin etiquetas; normalidad no verificada.',
                       'Evaluación adicional de adaptación, no modificación de las métricas del experimento original.']}
    (out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

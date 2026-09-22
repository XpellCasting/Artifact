import React, { useEffect, useMemo, useRef, useState } from "react";
import { FAULT_TYPES, diagnosticLabel } from "./diagnostic-data.js";
import { scenarioConflicts } from "./scenario-selection.js";

function GroundTruthPreview({ scenario }) {
  if (!scenario.ground_truth_preview) return null;
  return <details><summary>Ver etiquetas exportadas · {scenario.ground_truth_count} tramos</summary>
    <div className="review-table-scroll"><table><thead><tr><th>Entidad</th><th>Clase / falla</th><th>Origen</th><th>Inicio</th><th>Fin</th><th>Evaluable</th><th>Observación</th></tr></thead><tbody>{scenario.ground_truth_preview.map((row, i) => <tr key={i}><td>{row.entity_id}</td><td>{diagnosticLabel(row.kind || row.label)}</td><td>{row.reference_origin === "controlled_injection" ? "Inyección controlada" : "Revisión del histórico"}</td><td>{row.start}</td><td>{row.end}</td><td>{row.evaluable === "true" ? "Sí" : "No"}</td><td>{row.evaluation_note}</td></tr>)}</tbody></table></div>
    {scenario.ground_truth_preview_sampled && <p className="review-help">Se muestran los primeros 300 tramos; el CSV contiene todos.</p>}
  </details>;
}

export default function ScenarioWorkspace({ apiBaseUrl, dataset, runs, annotations = [], lastCreated, Comparison }) {
  const storageKey = `history-scenario:${dataset.dataset_id}`;
  const [selected, setSelected] = useState(() => {
    try { const value = JSON.parse(localStorage.getItem(storageKey) || "[]"); return Array.isArray(value) ? value.filter(x => typeof x === "string") : []; } catch { return []; }
  });
  const [name, setName] = useState("Escenario del histórico");
  const [saved, setSaved] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const operation = useRef(null);
  useEffect(() => () => operation.current?.abort(), []);
  const base = apiBaseUrl.replace(/\/$/, "");
  useEffect(() => { try { localStorage.setItem(storageKey, JSON.stringify(selected)); } catch { /* Server snapshots remain available. */ } }, [storageKey, selected]);
  useEffect(() => {
    if (lastCreated?.dataset_id === dataset.dataset_id) setSelected(current => [...new Set([...current, lastCreated.injection_id])]);
  }, [lastCreated, dataset.dataset_id]);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${base}/review/scenarios?dataset_id=${encodeURIComponent(dataset.dataset_id)}`, { signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error(data.detail || "No se pudieron cargar los escenarios."); return data; })
      .then(data => setSaved(data.scenarios || [])).catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [base, dataset.dataset_id]);
  const cases = useMemo(() => selected.map(id => runs.find(run => run.injection_id === id)).filter(Boolean), [selected, runs]);
  const groups = [...new Set(cases.map(run => run.entity_id))];
  const conflicts = scenarioConflicts(cases);
  function toggle(id) { setSelected(current => current.includes(id) ? current.filter(value => value !== id) : [...current, id]); setNotice(""); }
  async function download(id, signal, archive = false) {
    const response = await fetch(`${base}/review/scenarios/${id}/${archive ? "export" : "csv"}`, { signal });
    if (!response.ok) throw new Error("No se pudo descargar el histórico del escenario.");
    const blob = await response.blob();
    if (signal.aborted) return;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = archive ? `historico-escenario-${id}.zip` : `historico-etiquetado-${id}.csv`; document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  async function exportScenario(id, archive = false) {
    const controller = new AbortController();
    operation.current = controller;
    setBusy(true); setError(""); setNotice("");
    try {
      if (!id) {
        const response = await fetch(`${base}/review/scenarios`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_id: dataset.dataset_id, injection_ids: selected, name }), signal: controller.signal });
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "No se pudo construir el escenario.");
        if (controller.signal.aborted) return;
        setSaved(current => [...current, data]); id = data.scenario_id;
      }
      await download(id, controller.signal, archive);
      if (controller.signal.aborted) return;
      setNotice("Descarga preparada con las marcas y las intervenciones de la versión guardada del escenario.");
    } catch (e) { if (!controller.signal.aborted) setError(e.message); } finally { if (!controller.signal.aborted) setBusy(false); }
  }
  return <section className="injection-result" aria-label="Escenario del histórico completo">
    <h3>6. Acumular fallas y descargar el histórico completo</h3>
    <p className="review-help">Cada caso nuevo se añade a esta selección. Cambia de entidad en la revisión, genera otra falla y vuelve aquí: las anteriores permanecen. Puedes retirar casos sin borrar las inyecciones guardadas.</p>
    {error && <div className="review-alert" role="alert">{error}</div>}{notice && <div className="review-notice" role="status">{notice}</div>}
    <label>Nombre del escenario<input value={name} maxLength={160} onChange={e => setName(e.target.value)} disabled={busy} /></label>
    <p><strong>{dataset.rows?.toLocaleString("es-CL")} registros · {dataset.entities.length} entidades en el histórico completo</strong><br />{cases.length} fallas seleccionadas en {groups.length} entidades.</p>
    <div className="review-table-scroll"><table><thead><tr><th>Incluir</th><th>Entidad</th><th>Falla registrada</th><th>Inicio UTC</th><th>Fin UTC</th><th>Filas modificadas</th></tr></thead><tbody>{runs.map(run => <tr key={run.injection_id}><td><input type="checkbox" aria-label={`Incluir ${run.entity_id}, ${FAULT_TYPES[run.kind]}, ${run.injection_id}`} checked={selected.includes(run.injection_id)} disabled={busy} onChange={() => toggle(run.injection_id)} /></td><td>{run.entity_id}</td><td>{FAULT_TYPES[run.kind] || run.kind}</td><td>{run.start}</td><td>{run.end}</td><td>{run.changed_rows}</td></tr>)}</tbody></table></div>
    {conflicts.length > 0 && <p className="review-inline-error" role="alert">Hay intervalos superpuestos en {conflicts.join(", ")}. Retira uno de los casos para que cada fragmento conserve una falla inequívoca. Las comparaciones de abajo muestran los casos individuales; todavía no constituyen un escenario válido.</p>}
    <div className="review-actions"><button className="review-button review-button-primary" disabled={busy || (!cases.length && !annotations.length && !dataset.columns?.includes("tipo")) || cases.length !== selected.length || conflicts.length > 0} onClick={() => exportScenario()}>{busy ? "Preparando histórico…" : "Guardar escenario y descargar CSV etiquetado"}</button><button className="review-button" disabled={busy || !selected.length} onClick={() => setSelected([])}>Vaciar selección</button></div>
    <p className="review-help">El CSV contiene todas las filas con las columnas entity_id, state, last_changed, es_sintetica, tipo y tipo_falla. Solo los valores efectivamente modificados llevan es_sintetica=true. Las marcas manuales llevan false. Los datos sin revisar conservan NULL en tipo y tipo_falla; no se asumen normales.</p>
    <p className="review-help">El ZIP de respaldo incluye original.csv, injected.csv, labelled.csv, ground_truth.csv, annotations.csv, interventions.csv y manifest.json. Se incorporan todas las anotaciones actuales del archivo. Las pendientes e inciertas se conservan sin habilitarlas para métricas. Las inyecciones sustituyen la etiqueta del original solo donde se aplican. Cada intervalo conserva el nombre de su falla en las etiquetas; las columnas del histórico mantienen su estructura. Los registros sin modificar no se etiquetan automáticamente como normales.</p>
    <h4>Anotaciones manuales que se incorporarán · {annotations.length}</h4>
    <p className="review-help">Incluye normalidad, fallas del sensor y anomalías observadas, con origen Revisión del histórico. Se guarda la revisión actual de cada anotación. Puedes exportar estas referencias aunque no selecciones inyecciones.</p>
    <div className="review-table-scroll"><table><thead><tr><th>Entidad</th><th>Clase / falla</th><th>Inicio</th><th>Fin</th><th>Revisión</th><th>Evaluable</th></tr></thead><tbody>{annotations.map(a => <tr key={a.annotation_id}><td>{a.entity_id}</td><td>{diagnosticLabel(a.fault_type || a.label)}</td><td>{a.start}</td><td>{a.end}</td><td>{a.revision}</td><td>{a.evaluable ? "Sí, referencia del histórico" : "No, pendiente o incierto"}</td></tr>)}</tbody></table></div>
    <h4>Previsualización de todas las entidades modificadas</h4>
    {!cases.length && <p className="review-help">Genera una falla o selecciona un caso guardado para verlo aquí.</p>}
    {groups.map(entity => <section key={entity} className="injection-result"><h4>{entity}</h4>{cases.filter(run => run.entity_id === entity).sort((a,b) => a.start.localeCompare(b.start)).map(run => <article key={run.injection_id}><h4>{FAULT_TYPES[run.kind] || run.kind} · {run.changed_rows} filas modificadas</h4><p className="review-help">{run.start} → {run.end} · Semilla {run.seed} · {run.evaluable ? "Inyección evaluable" : "Caso exploratorio"}</p><Comparison run={run} continuous={dataset.entities.find(item => item.entity_id === entity)?.kind === "continuous"} /></article>)}</section>)}
    <h4>Escenarios guardados</h4><p className="review-help">Cada descarga guardada conserva su versión. Los escenarios anteriores a esta actualización solo tienen ZIP. Para obtener su CSV etiquetado o incorporar marcas nuevas, carga su selección y pulsa Guardar escenario y descargar CSV etiquetado.</p>
    {saved.map(item => <article key={item.scenario_id}><div className="injection-run"><span><strong>{item.name}</strong> · {item.rows} registros · {item.injection_ids.length} fallas · {item.modified_entities.length} entidades modificadas · {item.annotation_count == null ? "Versión anterior: solo inyecciones" : `${item.annotation_count} anotaciones manuales incluidas`}</span><div className="review-actions"><button className="review-button" disabled={busy} onClick={() => { setSelected(item.injection_ids); setName(item.name); }}>Cargar selección y previsualizar</button><button className="review-button" disabled={busy || !item.labelled_csv} onClick={() => exportScenario(item.scenario_id)}>Descargar CSV etiquetado</button><button className="review-button" disabled={busy} onClick={() => exportScenario(item.scenario_id, true)}>ZIP de respaldo</button></div></div><GroundTruthPreview scenario={item} /></article>)}
  </section>;
}

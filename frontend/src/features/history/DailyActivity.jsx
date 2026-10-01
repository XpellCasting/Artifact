import React from "react";

export default function DailyActivity({ activity, selectedDay, onSelect, disabled }) {
  const max = Math.max(1, ...activity.map(day => day.valid + day.invalid));
  const index = activity.findIndex(day => day.date === selectedDay);
  return <>
    <h4>Actividad diaria · UTC</h4>
    <p className="review-help">Pulsa un día para abrir su revisión. Verde: registros válidos. Coral: inválidos. Gris: sin reportes; no implica desconexión.</p>
    <svg className="exploration-activity" viewBox="0 0 1000 155" role="group" aria-label="Seleccionar un día para revisar">
      <text x="0" y="12" fill="#a6b5bd" fontSize="11">{max.toLocaleString("es-CL")} registros / día</text>
      {activity.map((day, i) => {
        const width = 1000 / activity.length, v = day.valid / max * 100, n = day.invalid / max * 100;
        return <g key={day.date} role="button" tabIndex={disabled ? -1 : 0} aria-disabled={disabled} aria-pressed={day.date === selectedDay}
          aria-label={`Revisar ${day.date}: ${day.valid} válidos, ${day.invalid} inválidos`}
          className="exploration-day" onClick={() => !disabled && onSelect(day.date)}
          onKeyDown={event => { if (!disabled && ["Enter", " "].includes(event.key)) { event.preventDefault(); onSelect(day.date); } }}>
          <title>{`${day.date}: ${day.valid} válidos, ${day.invalid} inválidos. Pulsar para revisar.`}</title>
          <rect className="exploration-day-hit" x={i * width} y="24" width={Math.max(.5, width - 1)} height="105" fill={day.date === selectedDay ? "#536c7c" : "#26333c"} />
          <rect x={i * width} y={129 - v} width={Math.max(.5, width - 1)} height={v} fill="#66c6b2" />
          <rect x={i * width} y={129 - v - n} width={Math.max(.5, width - 1)} height={n} fill="#ec9b87" />
        </g>;
      })}
      <text x="0" y="150" fill="#a6b5bd" fontSize="11">{activity[0]?.date}</text><text x="1000" y="150" textAnchor="end" fill="#a6b5bd" fontSize="11">{activity.at(-1)?.date}</text>
    </svg>
    <div className="review-actions"><button className="review-button" disabled={disabled || index <= 0} onClick={() => onSelect(activity[index - 1].date)}>Día anterior</button><label>Día para revisar<select value={index < 0 ? "" : selectedDay} disabled={disabled} onChange={event => onSelect(event.target.value)}><option value="" disabled>Selecciona un día</option>{activity.map(day => <option key={day.date} value={day.date}>{day.date}{day.valid + day.invalid === 0 ? " · sin reportes" : ""}</option>)}</select></label><button className="review-button" disabled={disabled || index < 0 || index >= activity.length - 1} onClick={() => onSelect(activity[index + 1].date)}>Día siguiente</button></div>
    <details><summary>Conteos diarios y revisión por día</summary><div className="review-table-scroll exploration-daily"><table><thead><tr><th>Día UTC</th><th>Válidos</th><th>Inválidos</th><th>Acción</th></tr></thead><tbody>{activity.map(day => <tr key={day.date}><td>{day.date}</td><td>{day.valid}</td><td>{day.invalid}</td><td><button className="review-button" disabled={disabled} onClick={() => onSelect(day.date)}>Revisar {day.date}</button></td></tr>)}</tbody></table></div></details>
  </>;
}

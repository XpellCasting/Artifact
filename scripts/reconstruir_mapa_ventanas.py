"""Regenera el mapa v3 desde Desarrollo y conserva la cache previa para comparar."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Artifact/backend'))
from app.Cargar_Historial.Cargar_Historial import CargadorHistorial
from app.Filtro_Contextual import FiltroContextualEspacioTemporal


def main():
    path = ROOT / 'Artifact/data/cache/contextual_map.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    history = CargadorHistorial().cargar_real(ROOT / 'HistoricoPuro/historico_desarrollo.csv')
    filtro = FiltroContextualEspacioTemporal()
    mapa = filtro.construir_mapa(history.eventos, history.entidades,
                                 trigger='window_selection_upgrade', mapa_anterior=previous)
    output = ROOT / 'outputs/analisis_temperaturas'
    output.mkdir(parents=True, exist_ok=True)
    backup = output / 'contextual_map_antes_v3.json'
    if previous and not backup.exists():
        backup.write_text(json.dumps(previous, ensure_ascii=False), encoding='utf-8')
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(mapa, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    saved = json.loads(temporary.read_text(encoding='utf-8'))
    assert saved['map_version'] == filtro.MAP_VERSION
    temporary.replace(path)
    ids = {'sensor.weatherstation_inside_temp_2', 'sensor.hall_sensor_temperature'}
    print(json.dumps({'version': mapa['map_version'], 'summary': mapa['summary'],
                      'edges': [e for e in mapa['edges'] if {e['source'], e['target']} == ids]}, indent=2))


if __name__ == '__main__':
    main()

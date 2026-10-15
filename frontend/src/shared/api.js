export async function request(base, path, options = {}) {
  const url = `${base.replace(/\/$/, "")}${path}`;
  let response;
  try { response = await fetch(url, { ...options, headers: { "Content-Type": "application/json", ...options.headers } }); }
  catch (error) {
    if (error instanceof TypeError) throw new Error(`No se pudo conectar con la API (${url}). Comprueba que Docker/API esté activo y que el navegador pueda acceder a esa dirección.`);
    throw error;
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = new Error(typeof payload.detail === "string" ? payload.detail : payload.detail ? JSON.stringify(payload.detail) : `Error HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

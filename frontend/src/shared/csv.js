export function csvFromRows(rows) {
  return `${rows
    .map((row) => row.map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(","))
    .join("\r\n")}\r\n`;
}

export function downloadCsv(rows, filename) {
  const url = URL.createObjectURL(new Blob([`\ufeff${csvFromRows(rows)}`], { type: "text/csv;charset=utf-8" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

if (typeof process !== "undefined" && process.argv?.[1]?.replaceAll("\\", "/").endsWith("/csv.js")) {
  const actual = csvFromRows([["Campo", "Cantidad"], ['a,"b"', 2]]);
  const expected = '"Campo","Cantidad"\r\n"a,""b""","2"\r\n';
  if (actual !== expected) {
    throw new Error(`CSV invalido: ${actual}`);
  }
}

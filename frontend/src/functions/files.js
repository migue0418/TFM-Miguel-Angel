
/**
 * Convierte el nombre del archivo en algo más amigable
 *  Ej: “ENCUESTA_SATISFACCION_2024.xlsx” ⇒ “ENCUESTA SATISFACCION 2024”
 *  – quita la extensión,
 *  – reemplaza _ y - por espacios,
 *  – trim final.
 */
export const prettifyFileName = (filename) =>
  filename
    .replace(/\.[^/.]+$/, '')      // quita extensión
    .replace(/[_]+/g, ' ')        // guiones/bajos → espacios
    .trim();

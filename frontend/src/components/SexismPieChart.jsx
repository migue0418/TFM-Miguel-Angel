import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Label } from 'recharts';

/**
 * SexismPieChart
 * --------------
 * Componente reutilizable que muestra un gráfico circular con el porcentaje
 * de contenido sexista (0‑100).
 *
 * Props:
 *  • percentage   Número (0‑100) con el % de sexismo.
 *  • height       Altura del gráfico (opcional, 200 por defecto).
 *  • colors       Array de dos colores [sexist, notSexist] (opcional).
 */
const SexismPieChart = ({
  percentage,
  isFraction = false,
  height = 180,
  colors = ['#dc3545', '#e5e7eb'],
}) => {
  let pct = Number(percentage);
  if (!Number.isFinite(pct)) pct = 0;
  if (isFraction) pct *= 100;     // <-- solo convertimos si lo indicas
  pct = Math.max(0, Math.min(100, pct));

  // evita sectores de tamaño 0 para que siempre se vean
  const tiny = 0.0001;
  const data = [
    { name: 'Sexista', value: Math.max(pct, tiny) },
    { name: 'No sexista', value: Math.max(100 - pct, tiny) },
  ];

  const outerR = Math.max(60, Math.floor(height / 2) - 10);
  const innerR = Math.max(40, outerR - 22);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          innerRadius={innerR}
          outerRadius={outerR}
          dataKey="value"
          startAngle={90}
          endAngle={-270}
          minAngle={1}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={colors[i]} stroke="#fff" strokeWidth={1} />
          ))}
          <Label
            content={({ viewBox }) => {
              const { cx, cy } = viewBox || {};
              if (cx == null || cy == null) return null;
              return (
                <text
                  x={cx}
                  y={cy}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ fontSize: '1rem', fontWeight: 700 }}
                >
                  {pct.toFixed(2)}%
                </text>
              );
            }}
          />
        </Pie>
      </PieChart>
    </ResponsiveContainer>
  );
};

export default SexismPieChart;

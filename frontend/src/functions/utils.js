
// Helper para mostrar el valor de un campo dado un diccionario y su id
export const getDictionaryValue = (dictionary, id_name, id_value, field_name) => {
    const item = dictionary.find(obj => obj[id_name] === id_value);
    return item ? item[field_name] : '';
};


export const getPredictionStyle = (pred, scoreSexist) => {
    if (pred === 'not sexist' && scoreSexist < 0.35) return { color: 'green', fontWeight: 'bold' };
    if (scoreSexist >= 0.35 && scoreSexist <= 0.65) return { color: '#555', fontWeight: 'bold' };
    if (pred === 'sexist' && scoreSexist > 0.65) return { color: 'red', fontWeight: 'bold' };
    return { fontWeight: 'bold' };
};

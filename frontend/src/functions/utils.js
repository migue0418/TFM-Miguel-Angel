
// Helper para mostrar el valor de un campo dado un diccionario y su id
export const getDictionaryValue = (dictionary, id_name, id_value, field_name) => {
    const item = dictionary.find(obj => obj[id_name] === id_value);
    return item ? item[field_name] : '';
};

#!/usr/bin/env python

__author__ = "francazorla7"
__license__ = "GPL"
__version__ = "1.0"

import pandas as pd

'''
@param filename: Nombre del fichero de datos de weka
@return tuple: String de la cabecera de Weka y el número de línea donde comienzan los datos
'''
def read_header(filename):

	# Buscamos la cabecera y la guardamos
	header = ""
	skip_rows = 1

	with open(filename, "r") as f:

		lines = f.readlines()
		for line in lines:
			if line.replace("\n", "") == "@data":
				break
			skip_rows += 1

		header = "".join(lines[:skip_rows])

	return (header, skip_rows)

'''
@param filename: Nombre del fichero de datos de weka
@param k: Numero de subconjuntos generados
@param size: Tamaño de los subconjuntos respecto al archivo original
'''
def generate_bagging_datasets(filename, k, size=0.626):

	# Buscamos la cabecera y la guardamos
	header, skip_rows = read_header(filename)

	# Cargamos el archivo de Weka obviando su cabecera
	data = pd.read_csv(filename, header=None, skiprows=skip_rows, sep=",")

	# Calculamos el tamaño de los subconjuntos
	subset_size = int(len(data.index)*size)

	# Generamos cada subconjunto
	for b in range(k):

		bagging_set = data.sample(subset_size, replace=True)

		with open("{}_b{}".format(filename, b), "w") as f:
			f.write("{}{}".format(header, bagging_set.to_csv(index=False)))

'''
@param filename: Nombre del fichero de datos de weka
@param k: Numero de subconjuntos generados
'''
def generate_crossing_datasets(filename, k):

	# Buscamos la cabecera y la guardamos
	header, skip_rows = read_header(filename)

	# Cargamos el archivo de Weka obviando su cabecera
	data = pd.read_csv(filename, header=None, skiprows=skip_rows, sep=",")

	# Generamos los partes
	chunk_size = int(float(len(data.index))/k)
	chunks = [data[i*chunk_size: (i+1)*chunk_size] for i in range(k)]

	# Generamos los subconjuntos
	subsets = [pd.concat([chunk for j, chunk in enumerate(chunks) if j != i]) for i in range(k)]

	# Guardamos los subconjuntos
	for i, subset in enumerate(subsets):

		with open("{}_c{}".format(filename, i), "w") as f:
			f.write("{}{}".format(header, subset.to_csv(index=False)))

if __name__ == '__main__':
	
	generate_bagging_datasets ("test/test.arff", 5)
	generate_crossing_datasets("test/test.arff", 5)
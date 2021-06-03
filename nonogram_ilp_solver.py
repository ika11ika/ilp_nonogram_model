import pyscipopt
from pyscipopt import Model, quicksum
from PIL import Image, ImageDraw
from statistics import mean
import numpy as np
import os.path
import datetime
import time
from os import listdir

def count_k(black_blocks_list):
	#in fact return k_row or k_col
	list_to_return = list()
	for element in black_blocks_list:
		list_to_return.append(len(element))
	return list_to_return
	
def count_e(black_blocks_list):
	e_array = list()
	block_counter = 0
	for row_index, blocks in enumerate(black_blocks_list):
		first_block_flag = 0
		if (len(black_blocks_list[row_index]) == 0):
			e_array.append([])
		for block_index, block in enumerate(blocks):
			if first_block_flag == 0:
				e_array.append([0]) 
				first_block_flag = 1
			else: 
				e_array[row_index].append(block_counter)
			block_counter += block + 1
		block_counter = 0
	return e_array	

def count_l(black_blocks_list, row_col_indicator):
	l_array = list()
	block_counter = row_col_indicator
	for row_index, blocks in enumerate(black_blocks_list):
		first_block_flag = 0
		if (len(black_blocks_list[row_index]) == 0):
			l_array.append([])
		for block_index, block in enumerate(blocks):
			if first_block_flag == 0:
				l_array.append([row_col_indicator-block])
				first_block_flag = 1
			else: 
				l_array[row_index].insert(0, block_counter-block)
			block_counter -= block
			block_counter -= 1
		block_counter = row_col_indicator
	l_array = list(reversed(l_array))
	return l_array

def reverse_everything(array):
	#reverse whole lost & every list element in it
	array = list(reversed(array))
	new_array = list()
	for element in array:
		new_array.append(list(reversed(element)))
	return new_array
	
def add_z(z, M, N, model):
	for i in range(M):
		for j in range(N):
			z[i,j] = model.addVar(vtype="I", lb=0, ub=1, name="z[%s,%s]" % (i, j))
	return z

def add_y(y, model, k_row, e_row, l_row, M):
	for i in range(M):
		for t in range(k_row[i]):
			for j in range(e_row[i][t], l_row[i][t]+1):
				y[i,t,j] = model.addVar(vtype="I", lb=0, ub=1, name="y[%s,%s,%s]" % (i,t,j))
	return y

def add_x(x, model, k_col, e_col, l_col, N):
	for j in range(N):
		for t in range(k_col[j]):
			for i in range(e_col[j][t], l_col[j][t]+1):
				x[j,t,i] = model.addVar(vtype="I", lb=0, ub=1, name="x[%s,%s,%s]" % (j,t,i))
	return x

def first_constraint(range_size, k, var, e, l, model):
	for i in range(range_size):
		for t in range(k[i]):
			model.addCons(quicksum(var[i,t,j] for j in range(e[i][t], l[i][t]+1) if (i,t,j) in var) == 1, name="Cluster row appear(%s)" % i)
	return model

def second_constraint(range_size, k, var, e, l, model,black_blocks_list):
	for i in range(range_size):
		for t in range(k[i]-1): #-1 чтобы не выйти за пределы, если k_row[i] == 1, то в строке только один блок и правило можно не проверять 
			for j in range(e[i][t]+1, l[i][t]+1):
				model.addCons(quicksum(var[i,t+1,j_stroke] for j_stroke in range(j + black_blocks_list[i][t] + 1, l[i][t+1]+1) if (i,t+1,j_stroke) in var) >= var[i,t,j], name="Cluster place(%s)" % i)
	return model

def third_constraint(row_range, col_range, k, var, e, l, model, black_blocks_list, z, flag_row_col): #cross-check
	for i in range(1, row_range+1):
		for j in range(1, col_range+1):
			if flag_row_col:
				z_key = (i-1,j-1)
			else: 
				z_key = (j-1,i-1)
			t_meanings = range(k[i-1])
			model.addCons(quicksum(var[i-1,t_1,j_stroke] for t_1 in t_meanings for j_stroke in range(max(e[i-1][t_1], j-1-black_blocks_list[i-1][t_1]+1),min(l[i-1][t_1],j-1)+1) if ((i-1,t_1,j_stroke) in var)) >= z[z_key], name="Cross con(%s)" % (i-1))
	return model

def fourth_constraint(row_range, col_range, k, var, e, l, model, black_blocks_list, z, flag_row_col):
	for i in range(row_range):
		for j in range(col_range):
			for t in range(k[i]):
				for j_stroke in range(e[i][t],l[i][t]+1):
					if flag_row_col:
						z_key = (i,j)
					else: 
						z_key = (j,i)
					if j_stroke >= j-black_blocks_list[i][t]+1 and j_stroke <= j:
						if ((i,t,j_stroke) in var):
							model.addCons(var[i,t,j_stroke]<= z[z_key], name="White cells con(%s)" % (i-1))
	return model

def board(n, m, nonogram, filename):
	result_path = './test_results/'
	new_color = (255, 255, 255)
	new_image = Image.new("RGB", (n*100, m*100), new_color)
	draw = ImageDraw.Draw(new_image)
	for i in range(m):
		for j in range(n):
			if(nonogram[i][j] == 1):
				draw.rectangle([(100*(j), 100*(i)),(100*(j+1), 100*(i+1))], fill='black')
	new_image.save(os.path.join(result_path,filename),format='PNG')	
	
def add_model_info(model, M, N, tableROW, tableCOL):
	
	k_row = count_k(tableROW)
	k_col = count_k(tableCOL)
	
	e_row = count_e(tableROW)
	e_col = count_e(tableCOL)
	
	l_row = count_l(reverse_everything(tableROW), N)
	l_col =  count_l(reverse_everything(tableCOL), M)
	
	#переменные 
	z = {} #same z as in optima, is pixel black or white
	y = {} #same y as in optima, is pixel j the first block t pixel
	x = {} #same x as in optima, is pixel i the first block t pixel
	
	z = add_z(z, M, N, model)
	y = add_y(y, model, k_row, e_row, l_row, M)	
	x = add_x(x, model, k_col, e_col, l_col, N)

	#ограничения
	model = first_constraint(M, k_row, y, e_row, l_row, model)
	model = first_constraint(N, k_col, x, e_col, l_col, model)
	
	model = second_constraint(M, k_row, y, e_row, l_row, model, tableROW)
	model = second_constraint(N, k_col, x, e_col, l_col, model, tableCOL)
	
	model = third_constraint(M, N, k_row, y, e_row, l_row, model, tableROW, z, True)
	model = third_constraint(N, M, k_col, x, e_col, l_col, model, tableCOL, z, False)
	
	model = fourth_constraint(M, N, k_row, y, e_row, l_row, model, tableROW, z, True)
	model = fourth_constraint(N, M, k_col, x, e_col, l_col, model, tableCOL, z, False)
	return model,z

def prepare_test_example():
	THRESHOLD_VALUE = 200
	path = './image_base/'
	result_path = './prepared_base/'
	for file in listdir(path):
		image = Image.open(os.path.join(path,file))
		fn = lambda x : 255 if x > THRESHOLD_VALUE else 0
		result_image = image.convert('L').point(fn, mode='1')	
		a = (list(result_image.getdata()))
		result_image.save(os.path.join(result_path,file),format='PNG')
		#matrix = np.array(result_image.getdata()) для печати
		
def get_table_rows(image_data, width, height):
	# i и j поменяны местами, так как доступ к значениям пикселей производится по координатам xy
	table = []
	for j in range(height):
		row_blocks = []
		size_counter = 0
		for i in range(width):
			if (image_data[i,j] == 0): 
				size_counter += 1
			if ((image_data[i,j] == 255 or i == (width - 1)) and size_counter > 0):
				row_blocks.append(size_counter)
				size_counter = 0
		table.append(row_blocks)
	return table
		
def get_table_columns(image_data, width, height):
	table = []
	for i in range(width):
		col_blocks = []
		size_counter = 0
		for j in range(height):
			if (image_data[i,j] == 0): 
				size_counter += 1
			if ((image_data[i,j] == 255 or j == (height - 1)) and size_counter > 0):
				col_blocks.append(size_counter)
				size_counter = 0
		table.append(col_blocks)
	return table
		

def base_solving():
	path = './prepared_base/'
	tests_timers = []
	for file in listdir(path):
		
		image = Image.open(os.path.join(path,file))
		pix = image.load()
		
		width = image.size[0] #Определяем ширину.
		height = image.size[1] #Определяем высоту.
		
		tableROW = get_table_rows(pix, width, height)
		tableCOL = get_table_columns(pix, width, height)
		z = {}
		
		model = pyscipopt.Model("Nonogram solver model")
		model2 = pyscipopt.Model("Nonogram solver model")
		start_test_timer =  datetime.datetime.now()
		model,z = add_model_info(model, height, width, tableROW, tableCOL)
		
		model.setParam('limits/time', 300)
		model.optimize()
		end_test_timer = datetime.datetime.now()
		
		if model.getStatus() != "optimal":
			continue
		model2,z2 = add_model_info(model2, height, width, tableROW, tableCOL)
		model2.setLongintParam("constraints/countsols/sollimit", 2)
		model2.setParam('limits/time', 600)
		model2.count()
		if (model2.getNCountedSols() > 1):
			continue
		current_test_time = end_test_timer - start_test_timer
		tests_timers.append(current_test_time.total_seconds())
		nonogram = []
		if model.getStatus() == "optimal":
			for i in range(height):
				nonogram.append([])
				for j in range(width):
					nonogram[i].append(int(model.getVal(z[i,j])))
		board(width, height, nonogram, file)
	return tests_timers

def main():
	prepare_test_example()
	tests_timers = base_solving()
	print("Max: " + str(max(tests_timers)))
	print("Min: " + str(min(tests_timers)))
	print("Average: " + str(round(mean(tests_timers),3)))
	print("Tests amount: " + str(len(tests_timers)))
	#z = {}
	#model = pyscipopt.Model("Nonogram solver model")
	#model,z = add_model_info(model)	
	#model.optimize()
	#model.writeProblem("try3")
	#nonogram = []
	'''
	if model.getStatus() == "optimal":
		for i in range(M):
			nonogram.append([])
			for j in range(N):
				nonogram[i].append(int(model.getVal(z[i,j])))
	
	board(N,M, nonogram)
	'''	
main()

"""
def prepare_test_example():

	#Перевод в градации серого для дальнейшего преобразования в ч-б

	path = './image_base/'
	result_path = './prepared_base/'
	num_files = len([f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))]) # количество тест-файлов, еще не решила зачем, на будущее 
	for i in listdir(path):
		img = Image.open(os.path.join(path,i))
		arr = np.asarray(img, dtype='uint8')
		x, y, _ = arr.shape
		k = np.array([[[0.2989, 0.587, 0.114]]])
		arr2 = np.round(np.sum(arr*k, axis=2)).astype(np.uint8).reshape((x, y))
		img2 = Image.fromarray(arr2)
		img2.save(os.path.join(result_path,i))
"""
		
		
		
"""
def grey_to_black_white():
	path = './prepared_base/'
	result_path = './test_results/'
	for file in listdir(path):
		image = Image.open(os.path.join(path,file))
		pix = image.load()
		
		width = image.size[0] #Определяем ширину.
		height = image.size[1] #Определяем высоту.
		
		new_image = Image.new(mode = "L", size = (width,height))
		draw = ImageDraw.Draw(new_image)

		factor = 100 #параметр по которому считывается, в какую сторону менять пиксель
		
		for i in range(width):
			for j in range(height):
				a = pix[i, j][0]
				b = pix[i, j][1]
				c = pix[i, j][2]
				S = a + b + c
				if (S > (((255 + factor) // 2) * 3)):
					a, b, c = 255, 255, 255
				else:
					a, b, c = 0, 0, 0
				draw.point((i, j), (a, b, c))
		new_image.save(os.path.join(result_path,file))
	return 0
"""
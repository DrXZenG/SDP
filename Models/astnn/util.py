import javalang
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from javalang.ast import Node


def parse_program(x):
	try:
		tree = javalang.parse.parse(x)
		return tree
	except:
		print("invalid")

def trans_to_sequences(ast):
	sequence = []
	get_sequence(ast, sequence)
	return sequence

def get_sequence(node, sequence):
	token, children = get_token(node), get_children(node)
	sequence.append(token)

	for child in children:
		get_sequence(child, sequence)

	if token in ['ForStatement', 'WhileStatement', 'DoStatement','SwitchStatement', 'IfStatement']:
		sequence.append('End')

def get_children(root):
	if isinstance(root, Node):
		children = root.children
	elif isinstance(root, set):
		children = list(root)
	else:
		children = []

	def expand(nested_list):
		for item in nested_list:
			if isinstance(item, list):
				for sub_item in expand(item):
					yield sub_item
			elif item:
				yield item
	return list(expand(children))

def get_token(node):
	token = ''
	if isinstance(node, str):
		token = node
	elif isinstance(node, set):
		token = 'Modifier'#node.pop()
	elif isinstance(node, Node):
		token = node.__class__.__name__

	return token

class BlockNode(object):
	def __init__(self, node):
		self.node = node
		self.is_str = isinstance(self.node, str)
		self.token = self.get_token(node)
		self.children = self.add_children()

	def is_leaf(self):
		if self.is_str:
			return True
		return len(self.node.children) == 0

	def get_token(self, node):
		if isinstance(node, str):
			token = node
		elif isinstance(node, set):
			token = 'Modifier'
		elif isinstance(node, Node):
			token = node.__class__.__name__
		else:
			token = ''
		return token

	def ori_children(self, root):
		if isinstance(root, Node):
			if self.token in ['CompilationUnit', 'MethodDeclaration', 'ConstructorDeclaration']:
				children = root.children[:-1]
			elif self.token == 'ClassDeclaration':
				children = root.children[:4] + root.children[5:]
			else:
				children = root.children
		elif isinstance(root, set):
			children = list(root)
		else:
			children = []

		def expand(nested_list):
			for item in nested_list:
				if isinstance(item, list):
					for sub_item in expand(item):
						yield sub_item
				elif item:
					yield item

		return list(expand(children))

	def add_children(self):
		if self.is_str:
			return []
		logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
		children = self.ori_children(self.node)
		if self.token in logic:
			return [BlockNode(children[0])]
		elif self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
			return [BlockNode(child) for child in children]
		else:
			return [BlockNode(child) for child in children if self.get_token(child) not in logic]

def get_blocks_v1(node, block_seq):
	name, children = get_token(node), get_children(node)
	logic = ['SwitchStatement','IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
	
	if name in ['CompilationUnit']:
		block_seq.append(BlockNode(node))
		body = node.types
		if body:
			for child in body:
				if get_token(child) not in logic and not hasattr(child, 'block') and get_token(child) not in ['ClassDeclaration','MethodDeclaration']:
					block_seq.append(BlockNode(child))
				#get_blocks_v1(child, block_seq)
				
	elif name in ['ClassDeclaration']:
		block_seq.append(BlockNode(node))
		body = node.body
		if body:
			for child in body:
				if get_token(child) not in logic and not hasattr(child, 'block') and get_token(child) not in ['MethodDeclaration']:
					block_seq.append(BlockNode(child))
				#get_blocks_v1(child, block_seq)
	
	elif name in ['MethodDeclaration', 'ConstructorDeclaration']:
		block_seq.append(BlockNode(node))
		body = node.body
		if body:
			for child in body:
				if get_token(child) not in logic and not hasattr(child, 'block'):
					block_seq.append(BlockNode(child))
				else:
					get_blocks_v1(child, block_seq)
	
	elif name in logic:
		block_seq.append(BlockNode(node))
		for child in children[1:]:
			token = get_token(child)
			if not hasattr(node, 'block') and token not in logic+['BlockStatement']:
				block_seq.append(BlockNode(child))
			else:
				get_blocks_v1(child, block_seq)
			block_seq.append(BlockNode('End'))
	elif name is 'BlockStatement' or hasattr(node, 'block'):
		block_seq.append(BlockNode(name))
		for child in children:
			if get_token(child) not in logic:
				block_seq.append(BlockNode(child))
			else:
				get_blocks_v1(child, block_seq)
	else:
		for child in children:
			get_blocks_v1(child, block_seq)

def get_methods(node, method_seq):
	name, children = get_token(node), get_children(node)
	if name in ["ClassDeclaration", 'MethodDeclaration']:
		body = node.body
		method_seq.append(node)
		if body:
			for child in body:
				get_methods(child, method_seq)
	elif name == 'CompilationUnit':
		body = node.types
		method_seq.append(node)
		if body:
			for child in body:
				get_methods(child, method_seq)
	else:
		for child in children:
			get_methods(child, method_seq)

def trans2seq(r):
	method_seq = []
	tree_seq = []
	get_methods(r, method_seq)
	for method in method_seq:
		blocks = []
		tree = []
		get_blocks_v1(method, blocks)
		for b in blocks:
			btree = tree_to_index(b)
			tree.append(btree)
		tree_seq.append(tree)
	return tree_seq
		
def tree_to_index(node):
	token = node.token
	#result = [token]
	w2v = Word2Vec.load("word2vec_node_64")
	vocab = w2v.wv.vocab
	max_token = w2v.wv.syn0.shape[0]
	result = [vocab[token].index if token in vocab else max_token]
	children = node.children
	for child in children:
		result.append(tree_to_index(child))
	return result

def get_data(Train_path, Test_path):
	source = pd.read_csv(Train_path)
	source['b_label'] = np.where(source['label']<0.5,0,1)
	source['AST'] = source['file'].apply(parse_program)
	source['AST_seq'] = source['AST'].apply(trans_to_sequences)

	corpus = source['AST_seq']
	w2v = Word2Vec(corpus, size=64, workers=16, sg=1, max_final_vocab=3000)
	w2v.save('word2vec_node_64')

	source['method_seq'] = source['AST'].apply(trans2seq)
	source.to_pickle("parsed_source.pkl")
	print("Training data saved ")
	
	source = pd.read_csv(Test_path)
	source['b_label'] = np.where(source['label']<0.5,0,1)

	source['AST'] = source['file'].apply(parse_program)
	source['method_seq'] = source['AST'].apply(trans2seq)
	source.to_pickle("parsed_source_test.pkl")
	print("Testing data saved ")
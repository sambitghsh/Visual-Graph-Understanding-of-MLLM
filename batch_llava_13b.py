from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch
import pandas as pd
from PIL import Image
import time
from tqdm import tqdm
torch.set_float32_matmul_precision('high')
import os
import json
from transformers import LlavaForConditionalGeneration, LlavaProcessor

from transformers import AutoProcessor, LlavaForConditionalGeneration

#torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True
#torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None

prompt1 = '''You are an expert in algorithms, graph theory, and data interpretation. Given the following graph and question, carefully analyze the graph, apply systematic algorithmic reasoning, and deduce the answer.

Input:

Graph Image: <image>

Question: {question}
{instruction}
✅ Final Answer:'''


graph_algorithms = {
    "shortest_path": {
        "description": "Finds the shortest path between two nodes in a graph.",
        "algorithm": "Dijkstra's Algorithm (for non-negative weights):\n"
                     "1. Initialize distances as infinity for all nodes except the source (0).\n"
                     "2. Use a priority queue to always expand the node with the smallest known distance.\n"
                     "3. Update distances to neighbors if a shorter path is found.\n"
                     "4. Repeat until all nodes are processed.\n"
                     "Bellman-Ford (handles negative weights) and Floyd-Warshall (all-pairs shortest paths) are other alternatives."
    },
    "neighbor": {
        "description": "Retrieves all directly connected nodes of a given node.",
        "algorithm": "Look up the adjacency list or adjacency matrix for the given node. Return all nodes that are directly connected."
    },
    "MST": {
        "description": "Finds a minimum spanning tree, a subset of edges connecting all nodes with minimal weight.",
        "algorithm": "Prim's Algorithm:\n"
                     "1. Start with any node.\n"
                     "2. Select the smallest edge connecting an unvisited node.\n"
                     "3. Repeat until all nodes are included.\n"
                     "Kruskal's Algorithm:\n"
                     "1. Sort edges by weight.\n"
                     "2. Use union-find to add edges without forming a cycle.\n"
                     "3. Continue until all nodes are connected."
    },
    "predecessor": {
        "description": "Finds the previous node(s) along a path in a traversal.",
        "algorithm": "Perform BFS or DFS while storing the parent of each node encountered. The stored parent links form a path tree."
    },
    "connected_component": {
        "description": "Finds groups of nodes where each node is reachable from any other in the group.",
        "algorithm": "Use BFS or DFS starting from unvisited nodes to explore entire connected components. Label each visited node with its component ID."
    },
    "DFS": {
        "description": "Depth-First Search explores as far as possible along each branch before backtracking.",
        "algorithm": "1. Start at a node and mark it as visited.\n"
                     "2. Recursively visit all unvisited neighbors.\n"
                     "3. Backtrack when no more unvisited neighbors remain."
    },
    "connectivity": {
        "description": "Checks if all nodes in a graph are connected.",
        "algorithm": "Run DFS or BFS from any node. If all nodes are visited, the graph is connected; otherwise, it is not."
    },
    "cycle": {
        "description": "Determines whether a cycle exists in a graph.",
        "algorithm": "For directed graphs: Use DFS with a recursion stack to detect back edges.\n"
                     "For undirected graphs: Use DFS while checking if a visited node is not the parent."
    },
    "common_neighbor": {
        "description": "Finds shared neighbors between two nodes.",
        "algorithm": "Retrieve the adjacency lists of both nodes and compute their intersection."
    },
    "diameter": {
        "description": "Finds the longest shortest path between any two nodes.",
        "algorithm": "1. Perform BFS from any node to find the farthest node.\n"
                     "2. Perform another BFS from this farthest node.\n"
                     "3. The longest distance found is the diameter."
    },
    "edge": {
        "description": "Represents a connection between two nodes.",
        "algorithm": "Check the adjacency list, matrix, or edge list for the presence of the edge."
    },
    "Jaccard": {
        "description": "Computes node similarity based on common neighbors.",
        "algorithm": "Jaccard Index = |A intersection B| / |A union B|, where A and B are the neighbor sets of two nodes."
    },
    "bipartite": {
        "description": "Checks if a graph can be colored with two colors such that no two adjacent nodes have the same color.",
        "algorithm": "Use BFS or DFS with two alternating colors. If a neighbor has the same color as its parent, the graph is not bipartite."
    },
    "topological_sort": {
        "description": "Orders nodes in a directed acyclic graph (DAG) such that for every edge (u, v), u appears before v.",
        "algorithm": "Kahn's Algorithm (BFS-based):\n"
                     "1. Compute in-degrees of all nodes.\n"
                     "2. Add nodes with zero in-degree to a queue.\n"
                     "3. Remove nodes from the queue and update in-degrees.\n"
                     "4. Repeat until all nodes are processed."
    },
    "degree": {
        "description": "Counts the number of edges connected to a node.",
        "algorithm": "For an adjacency list, count the number of neighbors. For an adjacency matrix, sum the row values."
    },
    "BFS": {
        "description": "Breadth-First Search explores neighbors before moving deeper.",
        "algorithm": "1. Start at a node and mark it as visited.\n"
                     "2. Use a queue to process nodes in FIFO order.\n"
                     "3. Visit all unvisited neighbors and add them to the queue."
    },
    "hamiltonian_path": {
        "description": "Finds a path that visits each node exactly once.",
        "algorithm": "Use backtracking: Try different node sequences and track visited nodes. If a complete path is found, return it."
    },
    "clustering_coefficient": {
        "description": "Measures how well nodes cluster together.",
        "algorithm": "For each node, count the number of triangles formed among its neighbors and compute:\n"
                     "Clustering Coefficient = (Actual triangles) / (Possible triangles)."
    },
    "euler_path": {
        "description": "Finds a path that visits every edge exactly once.",
        "algorithm": "1. Check degree conditions: A Eulerian Circuit requires all nodes to have even degrees; a Eulerian Path requires at most two nodes with odd degrees.\n"
                     "2. Use Hierholzer's algorithm: Start from a node with an unused edge, traverse edges recursively, and construct the path."
    },
    "page_rank": {
        "description": "Computes the importance of nodes based on incoming links.",
        "algorithm": "1. Initialize all node ranks equally.\n"
                     "2. Iteratively update ranks using:\n"
                     "   PR(v) = (1 - d) + d * sum(PR(u) / out-degree(u)) for all in-neighbors u.\n"
                     "3. Repeat until convergence."
    },
    "maximum_flow": {
        "description": "Finds the maximum possible flow from a source to a sink in a network.",
        "algorithm": "Ford-Fulkerson Algorithm:\n"
                     "1. Start with zero flow.\n"
                     "2. Find an augmenting path using BFS (Edmonds-Karp) or DFS.\n"
                     "3. Increase flow along the path.\n"
                     "4. Repeat until no more augmenting paths exist."
    }
}

# Example usage:
#print(graph_algorithms["BFS"])



prompt2 = '''You are an expert in algorithms, graph theory, and data interpretation. Given the following graph and question, carefully analyze the graph, apply systematic algorithmic reasoning, and deduce the answer.

Input:

Graph Image: <image>

Task Description: {task}

Algorithm: {algo}

Question: {question}
{instruction}
Please provide the answer in following format

✅ Final Answer:'''


prompt3 = '''You are an expert in algorithms, graph theory, and data interpretation. Given the following graph and question, carefully analyze the graph, apply systematic algorithmic reasoning, and deduce the answer.

Input:

Graph Image: <image>

Question: {question}

steps: 
{trace}
{instruction}
✅ Final Answer:'''



prompt4 = '''You are an expert in algorithms, graph theory, and data interpretation. Given the following graph and question, carefully analyze the graph, apply systematic algorithmic reasoning, and deduce the answer.

Input:

Graph Image: <image>

Graph Description: 
{gdesc}

Question: {question}
{instruction}
Please provide the answer in following format

✅ Final Answer:'''

######################## Loading the model ##############################
model_id = "llava-hf/llava-1.5-13b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir="/dccstor/mkesar/nltogql/temp_files_exp/Llava_13b", torch_dtype=torch.float16).eval()
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
processor = LlavaProcessor.from_pretrained(model_id, cache_dir="/dccstor/mkesar/nltogql/temp_files_exp/Llava_13b")
path = '/dccstor/mkesar/nltogql/temp_files_exp/Vision_Graph_2/'



######################## First Prompt Evaluation ##############################
print('Staring the first prompt evaluation!!!!!!!!!!!!!!!!!!!!!!!!!')
def generate_with_first_prompt(batch_size=8):
    with torch.inference_mode():
        for i in os.listdir(path):
            print(i)
            if i in ['.DS_Store', '.ipynb_checkpoints', '.locks']:
                continue
            csv_path = os.path.join(path, i, i+'.xlsx')
            df = pd.read_excel(csv_path)
            text_name = os.path.join('/dccstor/mkesar/nltogql/temp_files_exp/Result', i, 'llava13b_output')
            if not os.path.exists(text_name):
                os.makedirs(text_name)
            choice_flag = 1 if 'choices' in df.columns else 0    
            prompts = []
            images = []
            for k, row in df.iterrows():
                options = ""
                instruct = ""
                if choice_flag:
                    opt = []
                    for el in eval(row['choices']):
                        opt.append(str(el))
                    instruct = "\nPlease choose from one of the following options \nAnswer Choices:\n{options}\n" if choice_flag else ""
                    instruct = instruct.format(options='\n'.join(opt))
                
                prompt = prompt1.format(question=row['question'], instruction=instruct)
                img_name = '/dccstor/mkesar/nltogql/temp_files_exp/test.jpg'
                image = Image.open(img_name).convert("RGB")
                prompts.append(prompt)
                images.append(image)
            
            # Process in batches
            for batch_start in tqdm(range(0, len(prompts), batch_size)):
                batch_end = batch_start + batch_size
                batch_prompts = prompts[batch_start:batch_end]
                batch_images = images[batch_start:batch_end]
                
                model_inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.bfloat16)
                
                generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                decoded_texts = processor.batch_decode(generation, skip_special_tokens=True)

                file_name = os.path.join(text_name, 'first_prompt.jsonl') 
                with open(file_name, "a") as outfile:
                    for idx, decoded in enumerate(decoded_texts):
                        temp = {"id": batch_start + idx, "input": batch_prompts[idx], "output": decoded}
                        json.dump(temp, outfile) 
                        outfile.write('\n')
                
# generate_with_first_prompt(batch_size=4)
# print("First Prompt Done #####################")



######################## Second Prompt Evaluation ##############################
print('Staring the second prompt evaluation!!!!!!!!!!!!!!!!!!!!!!!!!')
def generate_with_second_prompt(batch_size=8):
    with torch.inference_mode():
        for i in os.listdir(path):
            print(i)
            if i in ['.DS_Store', '.ipynb_checkpoints', '.locks']:
                continue
            csv_path = os.path.join(path, i, i+'.xlsx')
            df = pd.read_excel(csv_path)
            text_name = os.path.join('/dccstor/mkesar/nltogql/temp_files_exp/Result', i, 'llava13b_output')
            if not os.path.exists(text_name):
                os.makedirs(text_name)
            choice_flag = 1 if 'choices' in df.columns else 0    
            prompts = []
            images = []
            for _, row in df.iterrows():
                instruct = ""
                options = ""
                if choice_flag:
                    opt = []
                    for el in eval(row['choices']):
                        opt.append(str(el))
                    instruct = "\nPlease choose from one of the following options \nAnswer Choices:\n{options}\n" if choice_flag else ""
                    instruct = instruct.format(options = '\n'.join(opt))

                prompt = prompt2.format(task = graph_algorithms[row['task']]['description'], algo =  graph_algorithms[row['task']]['algorithm'], question = row['question'], instruction = instruct)
                img_name = '/dccstor/mkesar/nltogql/temp_files_exp/test.jpg'
                image = Image.open(img_name).convert("RGB")
                prompts.append(prompt)
                images.append(image)
            
            # Process in batches
            for batch_start in tqdm(range(0, len(prompts), batch_size)):
                batch_end = batch_start + batch_size
                batch_prompts = prompts[batch_start:batch_end]
                batch_images = images[batch_start:batch_end]
                
                model_inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.bfloat16)
                
                generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                decoded_texts = processor.batch_decode(generation, skip_special_tokens=True)
                
                file_name = os.path.join(text_name, 'second_prompt.jsonl') 
                with open(file_name, "a") as outfile:
                    for idx, decoded in enumerate(decoded_texts):
                        temp = {"id": batch_start + idx, "input": batch_prompts[idx], "output": decoded}
                        json.dump(temp, outfile) 
                        outfile.write('\n')

# generate_with_second_prompt(batch_size=2)
# print("Second Prompt Done ############################")



######################## Third Prompt Evaluation ##############################
print('Staring the third prompt evaluation!!!!!!!!!!!!!!!!!!!!!!!!!')
remain = ['diameter','edge', 'bipartite', 'jaccard', 'topological_sort', 'common_neighbor', 'page_rank', 'degree','shortest_path']
def generate_with_third_prompt(batch_size=8):
    with torch.inference_mode():
        for i in remain:
            print(i)
            if i in ['.DS_Store', '.ipynb_checkpoints', '.locks']:
                continue
            csv_path = os.path.join(path, i, i+'.xlsx')
            df = pd.read_excel(csv_path)
            text_name = os.path.join('/dccstor/mkesar/nltogql/temp_files_exp/Result', i, 'llava13b_output')
            if not os.path.exists(text_name):
                os.makedirs(text_name)
            choice_flag = 1 if 'choices' in df.columns else 0    
            prompts = []
            images = []
            for _, row in df.iterrows():
                instruct = ""
                tr = "\n"
                options = ""

                if 'steps' in list(row.keys()):
                    tr = "\n{trace}\n".format(trace = row['steps'])

                if choice_flag:
                    opt = []
                    for el in eval(row['choices']):
                        opt.append(str(el))
                    instruct = "\nPlease choose from one of the following options \nAnswer Choices:\n{options}\n" if choice_flag else ""
                    instruct = instruct.format(options = '\n'.join(opt))
                
                prompt = prompt3.format(trace = tr, question = row['question'], instruction = instruct)
                img_name = '/dccstor/mkesar/nltogql/temp_files_exp/test.jpg'
                image = Image.open(img_name).convert("RGB")
                prompts.append(prompt)
                images.append(image)
            
            # Process in batches
            for batch_start in tqdm(range(0, len(prompts), batch_size)):
                batch_end = batch_start + batch_size
                batch_prompts = prompts[batch_start:batch_end]
                batch_images = images[batch_start:batch_end]
                
                model_inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.bfloat16)
                
                generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                decoded_texts = processor.batch_decode(generation, skip_special_tokens=True)
                
                file_name = os.path.join(text_name, 'third_prompt.jsonl') 
                with open(file_name, "a") as outfile:
                    for idx, decoded in enumerate(decoded_texts):
                        temp = {"id": batch_start + idx, "input": batch_prompts[idx], "output": decoded}
                        json.dump(temp, outfile) 
                        outfile.write('\n')

generate_with_third_prompt(batch_size=2)
print("Third Prompt Done ########################")




######################## Fourth Prompt Evaluation ##############################
print('Staring the Fourth prompt evaluation!!!!!!!!!!!!!!!!!!!!!!!!!')
def generate_with_fourth_prompt(batch_size=8):
    with torch.inference_mode():
        for i in os.listdir(path):
            print(i)
            if i in ['.DS_Store', '.ipynb_checkpoints', '.locks']:
                continue
            csv_path = os.path.join(path, i, i+'.xlsx')
            df = pd.read_excel(csv_path)
            text_name = os.path.join('/dccstor/mkesar/nltogql/temp_files_exp/Result', i, 'llava13b_output')
            if not os.path.exists(text_name):
                os.makedirs(text_name)
            choice_flag = 1 if 'choices' in df.columns else 0    
            prompts = []
            images = []
            for _, row in df.iterrows():
                instruct = ""
                options = ""

                direction = 'directed' if row['directed'] else 'undirected'
                graph_desc = 'This is a {direction} graph, the nodes and edges informations are as follows\n{nl}'.format(direction = direction, nl = row['graph_nl'])

                if choice_flag:
                    opt = []
                    for el in eval(row['choices']):
                        opt.append(str(el))
                    instruct = "\nPlease choose from one of the following options \nAnswer Choices:\n{options}\n" if choice_flag else ""
                    instruct = instruct.format(options = '\n'.join(opt))
                
                prompt = prompt4.format(gdesc = graph_desc, question = row['question'], instruction = instruct)
                img_name = '/dccstor/mkesar/nltogql/temp_files_exp/test.jpg'
                image = Image.open(img_name).convert("RGB")
                prompts.append(prompt)
                images.append(image)
            
            # Process in batches
            for batch_start in tqdm(range(0, len(prompts), batch_size)):
                batch_end = batch_start + batch_size
                batch_prompts = prompts[batch_start:batch_end]
                batch_images = images[batch_start:batch_end]
                
                model_inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.bfloat16)
                
                generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                decoded_texts = processor.batch_decode(generation, skip_special_tokens=True)

                file_name = os.path.join(text_name, 'fourth_prompt.jsonl') 
                with open(file_name, "a") as outfile:
                    for idx, decoded in enumerate(decoded_texts):
                        temp = {"id": batch_start + idx, "input": batch_prompts[idx], "output": decoded}
                        json.dump(temp, outfile) 
                        outfile.write('\n')  


generate_with_fourth_prompt(batch_size=2)
print("Forth Prompt Done ##########################")
import re
import copy
import math

def split_text(text, chunk_length=None):
    chunks = []
    # Even more explicit - captures everything up to and INCLUDING the punctuation
    split_regex = r'.+?(?:[\.?!;:—–]+(?:\s|$)|\r?\n|$)'
    splitted_text = re.findall(split_regex, text)
    
    cumulative_length = []
    length = 0
    for token in splitted_text:
        length += len(token) + 1
        cumulative_length.append(length)

    text_length = cumulative_length[-1]
    number_of_chunks = math.ceil(text_length / chunk_length)
    chunk_length = max(int(text_length / number_of_chunks), 40)


    split_indices = []
    
    current_running_length = 0
    prev_length = 0
    last_chunk_index = 0
    last_iteration_index = len(cumulative_length) - 1
    for i, length in enumerate(cumulative_length):
        previous_running_length = current_running_length
        current_running_length += length - prev_length
        if current_running_length > chunk_length or i == last_iteration_index:
            dist1 = current_running_length - chunk_length
            dist2 = chunk_length - previous_running_length
            if dist2 < dist1:
                if i == last_iteration_index:
                    debug = True
                to_append = ' '.join(splitted_text[last_chunk_index:i])
                last_chunk_index = i
                current_running_length = length - prev_length
                split_indices.append(i)
            else:
                to_append = ' '.join(splitted_text[last_chunk_index:i+1])
                last_chunk_index = i+1
                current_running_length = 0
                split_indices.append(i+1)

            chunks.append(to_append)
        prev_length = length
    split_indices.insert(0,0)
    if last_iteration_index+1 not in split_indices:
        split_indices.append(last_iteration_index+1)
    cumulative_length.insert(0,0)
    cost = float('inf')
    original_cost = cost_function(cumulative_length, sorted(split_indices), chunk_length)
    new_cost = original_cost
    while new_cost < cost:
        cost = new_cost
        split_indices = iterative_optimization(cumulative_length, split_indices, chunk_length)
        new_cost = cost_function(cumulative_length, sorted(split_indices), chunk_length)
        debug = True
    if new_cost > cost:
        achtung = "minnen"
    new_chunks = get_chunks(splitted_text, split_indices)

    return new_chunks



def get_chunks(splitted_text, split_indices):
    return [' '.join(splitted_text[split_indices[i]: split_indices[i+1]]) for i in range(len(split_indices)-1)]


def cost_function(cumulative_length, split_indices, target_chunk_length):
    return sum((cumulative_length[split_indices[i+1]] - cumulative_length[split_indices[i]] - target_chunk_length)**2 
                for i in range(len(split_indices)-1))


def iterative_optimization(cumulative_length, split_indices, target_chunk_length):
    current_cost = cost_function(cumulative_length, split_indices, target_chunk_length)

    """
    Possible actions are:
    - Add split => Cost = previous_cost + (chunk1_length - target_chunk_length) **2 + (chunk2_length - target_chunk_length) **2 - (chunk2_length + chunk1_length - target_chunk_length)**2
    - Remove split => Cost = opposite of Add split
    - Move split right
    - Move split left
    """
    split_indices = set(split_indices)
    max_split_index = len(cumulative_length) - 1
    for i, length in enumerate(cumulative_length):
        split_index_before, split_index_after = get_surrounding_split_indices(split_indices, i, max_split_index)

        if i in split_indices:
            if i == 106:
                debug = True
            current_cost = get_current_cost_for_movement_or_removal(cumulative_length, target_chunk_length, i, split_index_before, split_index_after)
            costs = {
                'cost_of_removal': get_cost_of_split_removal(cumulative_length, target_chunk_length, i, split_index_before, split_index_after),
                'cost_of_moving_left': get_cost_of_moving_split_left(cumulative_length, target_chunk_length, i, split_index_before, split_index_after),
                'cost_of_moving_right': get_cost_of_moving_split_right(cumulative_length, target_chunk_length, i, split_index_before, split_index_after),
            }
            best_action = min(costs, key=costs.get)
            best_cost = costs[best_action]
            if best_cost < current_cost:
                if best_action == 'cost_of_removal':
                    split_indices.remove(i)
                elif best_action == 'cost_of_moving_left':
                    split_indices.remove(i)
                    split_indices.add(i-1)
                elif best_action == 'cost_of_moving_right':
                    split_indices.remove(i)
                    split_indices.add(i+1)
            continue
        current_cost = (cumulative_length[split_index_after] - cumulative_length[split_index_before] - target_chunk_length) ** 2
        additional_split_cost = get_cost_of_split_addition(cumulative_length, target_chunk_length, i, split_index_before, split_index_after)
        if additional_split_cost < current_cost:
            split_indices.add(i)
    
    return sorted(split_indices)

def get_surrounding_split_indices(split_indices, split_index_to_check, maximum):
    split_index_before = 0
    split_index_after = maximum
    for split_index in split_indices:
        if split_index_before < split_index < split_index_to_check:
            split_index_before = split_index
            continue
        if split_index_after > split_index > split_index_to_check:
            split_index_after = split_index
    
    return split_index_before, split_index_after


def get_current_cost_for_movement_or_removal(cumulative_length, target_chunk_length, split_index_considered, split_index_before, split_index_after):
    current_cost = 0
    if split_index_considered != split_index_after:
        current_cost += (cumulative_length[split_index_after] - cumulative_length[split_index_considered] - target_chunk_length) ** 2
    if split_index_considered != split_index_before:
        current_cost += (cumulative_length[split_index_considered] - cumulative_length[split_index_before] - target_chunk_length) ** 2

    return current_cost

def get_cost_of_split_addition(cumulative_length, target_chunk_length, additional_split_index, split_index_before, split_index_after):
    cost_after_split = (cumulative_length[split_index_after] - cumulative_length[additional_split_index] - target_chunk_length) ** 2 + (cumulative_length[additional_split_index] - cumulative_length[split_index_before] - target_chunk_length) ** 2

    return cost_after_split


def get_cost_of_split_removal(cumulative_length, target_chunk_length, split_index_to_remove, split_index_before, split_index_after):
    if split_index_before == split_index_to_remove or split_index_to_remove == split_index_after:
        return float('inf')
    cost_after_removal = (cumulative_length[split_index_after] - cumulative_length[split_index_before] - target_chunk_length) ** 2
    
    return cost_after_removal

def get_cost_of_moving_split_left(cumulative_length, target_chunk_length, split_index_to_decrease_by_one, split_index_before, split_index_after):
    if split_index_to_decrease_by_one == 0:
        return float('inf')
    cost_after_move = (cumulative_length[split_index_after] - cumulative_length[split_index_to_decrease_by_one-1] - target_chunk_length) ** 2 + (cumulative_length[split_index_to_decrease_by_one-1] - cumulative_length[split_index_before] - target_chunk_length) ** 2
    
    return cost_after_move

def get_cost_of_moving_split_right(cumulative_length, target_chunk_length, split_index_to_increase_by_one, split_index_before, split_index_after):
    if split_index_to_increase_by_one == len(cumulative_length) - 1:
        return float('inf')
    cost_after_move = (cumulative_length[split_index_after] - cumulative_length[split_index_to_increase_by_one+1] - target_chunk_length) ** 2 + (cumulative_length[split_index_to_increase_by_one+1] - cumulative_length[split_index_before] - target_chunk_length) ** 2
    return cost_after_move

    
if __name__ == "__main__":
    # split_indices = set([1,2,3,5,6])
    # split_index_to_check = 3
    # bof1, bof2 = get_surrounding_split_indices(split_indices, split_index_to_check)
    with open('anarchism.txt', 'r+') as file:
        text = file.read()

    split_text(text, chunk_length=14315)

import re
import sys

sys.setrecursionlimit(20000)


def split_text(text, number_of_chunks):
    text_length = len(text)
    chunk_length = text_length / number_of_chunks
    chunks = []
    split_regex = r'(?=\.\.\.|[\.?!;:—–] |\r\n|\r|\n)'
    splitted_text = re.split(split_regex, text)
    if len(splitted_text) < number_of_chunks:
        return splitted_text
    
    cumulative_length = []
    length = 0
    for token in splitted_text:
        length += len(token)
        cumulative_length.append(length)
    prev_length = 0
    split_indices = []
    prev_modulo = 0
    for i, length in enumerate(cumulative_length):
        modulo = length%chunk_length
        if round(modulo - prev_modulo) != round(length - prev_length) and length < text_length:
            end_distance = length - modulo - prev_length
            if end_distance < modulo:
                split_indices.append(prev_length)
            else:
                split_indices.append(length)
        
        prev_length = length
        prev_modulo = modulo
    split_indices.insert(0, 0)
    split_indices.append(text_length)
    chunks = [text[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices) - 1)]
    
    # [split_indices[i+1] - split_indices[i] for i in range(len(split_indices) - 1)]
    loss = sum([(len(chunk) - chunk_length)**2 for chunk in chunks])**.5
    return chunks
    
def split_text_3(text, number_of_chunks):
    text_length = len(text)
    chunk_length = text_length / number_of_chunks
    chunks = []
    split_regex = r'(?=\.\.\.|[\.?!;:—–] |\r\n|\r|\n)'
    splitted_text = re.split(split_regex, text)
    if len(splitted_text) < number_of_chunks:
        return splitted_text
    
    cumulative_length = []
    length = 0
    for token in splitted_text:
        length += len(token)
        cumulative_length.append(length)
    
    current_running_length = 0
    prev_length = 0
    last_chunk_index = 0
    for i, length in enumerate(cumulative_length):
        previous_running_length = current_running_length
        current_running_length += length - prev_length
        if current_running_length > chunk_length:
            dist1 = current_running_length - chunk_length
            dist2 = chunk_length - previous_running_length
            if dist2 < dist1:
                chunks.append(text[prev_length - previous_running_length:prev_length])
                current_running_length = length - prev_length
            else:
                chunks.append(text[length - current_running_length:length])
                current_running_length = 0
        prev_length = length
    loss = sum([(len(chunk) - chunk_length)**2 for chunk in chunks])**.5
    return chunks


def split_text_2(text, number_of_chunks):
    L = len(text)
    if L == 0:
        return []
    if number_of_chunks <= 0:
        raise ValueError("Number of chunks must be positive")
    if number_of_chunks == 1 or L == 0:
        return [text.strip()]
    
    # Find candidate split positions: after punctuation or newlines
    positions = set([0, L])
    punctuation = '.?!,;:\n'
    for i in range(1, L):
        if text[i-1] in punctuation:
            positions.add(i)
    P = sorted(list(positions))
    m = len(P)
    
    if number_of_chunks > m - 1:
        # Too many chunks requested; fall back to maximum possible
        number_of_chunks = m - 1
        if number_of_chunks <= 0:
            return [text.strip()]
    
    mean = L / number_of_chunks
    
    # DP table: dp[k][i] = min cost for k chunks ending at P[i]
    INF = sys.float_info.max
    dp = [[INF] * m for _ in range(number_of_chunks + 1)]
    prev = [[-1] * m for _ in range(number_of_chunks + 1)]
    dp[0][0] = 0
    
    for k in range(1, number_of_chunks + 1):
        for i in range(1, m):
            for j in range(i):
                if dp[k-1][j] < INF:
                    cost = (P[i] - P[j] - mean) ** 2
                    total_cost = dp[k-1][j] + cost
                    if total_cost < dp[k][i]:
                        dp[k][i] = total_cost
                        prev[k][i] = j
    
    if dp[number_of_chunks][m-1] == INF:
        # Fallback if impossible (though adjustment should prevent this)
        return [text.strip()]
    
    # Backtrack to find split positions
    chunks = []
    current_i = m - 1
    current_k = number_of_chunks
    while current_k > 0:
        prev_i = prev[current_k][current_i]
        chunk = text[P[prev_i]:P[current_i]].strip()  # Strip for cleanliness
        chunks.append(chunk)
        current_i = prev_i
        current_k -= 1
    chunks.reverse()

    loss = sum([(len(chunk) - mean)**2 for chunk in chunks])**.5
    return chunks


if __name__ == "__main__":

    with open('/Users/eloireynal/Documents/My projects/crawl_data/sanitized_txt/www.weston.ca.md', 'r+') as file:
        sample_text = file.read()

    desired_chunks = 1024
    print(f"--- Splitting text into {desired_chunks} chunks ---\n")
    
    result_chunks = split_text(sample_text[:], desired_chunks)
    result_chunks = split_text_3(sample_text[:], desired_chunks)
    result_chunks = split_text_2(sample_text[:], desired_chunks)

    total_len = len(sample_text)
    print(f"Original Text Length: {total_len} chars")
    print(f"Target Chunk Length: ~{total_len / desired_chunks:.0f} chars\n")

    for i, chunk in enumerate(result_chunks):
        print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
        # print(chunk)
        print("-" * (len(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")))
        print()
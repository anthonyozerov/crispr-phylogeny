import numpy as np
import copy
import networkx as nx

def simulate_imbalanced_tree(num_init_cells, init_death_prob=0.1, init_repr_prob=0.75,
                             cancer_prob=1e-3, tree_depth=15):
    num_cells = num_init_cells
    death_probs = [init_death_prob] * num_cells
    repr_probs = [init_repr_prob] * num_cells

    init_cells = [str(i) for i in np.arange(num_cells)]
    parent_ix = []
    cell_names = [np.array(init_cells)]
    repetition_coefs_list = []
    
    
    for i in range(tree_depth):
        dying = np.random.random(num_cells) < death_probs
        mutating = np.random.random(num_cells) < repr_probs

        repetition_coefs = (mutating+1)*(1-dying)

        repetition_coefs_list.append(repetition_coefs)
        next_gen = np.repeat(init_cells, repetition_coefs)

        if len(next_gen) == 0:
            raise Exception('No cells left to replicate. Terminate simulation.')
        # Label generation in terms of binary split with parents
        binary_labels = [next_gen[i]+'1' if next_gen[i-1] == next_gen[i] else
                         next_gen[i]+'0' for i in range(1, len(next_gen))]
        binary_labels = [next_gen[0]+'0'] + binary_labels
        cell_names.append(np.array(binary_labels))
        parent_ix.append(np.repeat(np.arange(num_cells), repetition_coefs))

        death_probs = np.repeat(death_probs, repetition_coefs)
        repr_probs = np.repeat(repr_probs, repetition_coefs)

        num_cells = sum(repetition_coefs)

        init_cells = binary_labels

        # Introduce cancerous mutations which may increase tumour fitness
        has_cancer = np.random.random(num_cells) < cancer_prob
        death_probs[has_cancer] -= 1e-2
        repr_probs[has_cancer] += 1e-2

    return cell_names

def get_empirical_params():
    
    import pickle
    with open('d21_indel_distributions.p', 'rb') as f:
        indel_distributions = pickle.load(f)

    speed = {'AGCTGCTTAGGGCGCAGCCT': 'slow',
         'CTCCTTGCGTTGACCCGCTT': 'slow',
         'TATTGCCTCTTAATCGTCTT': 'slow',
         'AATCCCTAGTAGATTAGCCT': 'medium',
         'CACAGAACTTTATGACGATA': 'medium',
         'TTAAGTTTGAGCTCGCGCAA': 'medium',
         'TAATTCCGGACGAAATCTTG': 'fast',
         'CTTCCACGGCTCTAGTACAT': 'fast',
         'CCTCCCGTAGTGTTGAGTCA': 'fast'
        }

    rate = {}
    rate['slow'] = 0.02361
    rate['medium'] = 0.05668
    rate['fast'] = 0.2269
    
    return speed, rate, indel_distributions

def generate_cassettes(total_internal_nodes, 
                       cassette_sites, 
                       ignore_deletions = True):
    """
    CRISPR edits are independent of each other. Each edit site has some probability of being modified or being deleted
    The only dependence between cells is that a site which has already been edited or deleted cannot be edited or deleted again

    We can a priori simulate CRISPR edits for a given number of cells and then attach these edits to cells in the lineage
    independently of one another as vector operations for speed up

    @param: total_internal_nodes: int - Number of cells for which we should simulate CRISPR recording cassettes
    @param: cassette_sites: list(str) - Guide RNAs defining the edit probabilities and indel distributions in the CRISPR cassette
    """
        
    speed, rate, indel_distributions = get_empirical_params()
    all_edits = []

    Qrows = []
    n_states = 0
    for i, site in enumerate(cassette_sites):

        #print(f'Adding {i}st/th site with speed {speed[site]}')
        # Determine the probability of a CRISPR edit occurring during a generation
        prob_edit = rate[speed[site]]
        #print(prob_edit)
        edit_happens = (np.random.random(size=total_internal_nodes) < prob_edit).astype(int)
        #print(edit_happens)
        
        indels = copy.deepcopy(indel_distributions[site])
        # Now sample which edit occurs in each cell
        del_counts = indels['']
        del indels['']
        other_counts = list(indels.values())

        counts = [del_counts] + other_counts
        counts = np.array(counts)
        #print(counts)
        #print(len(counts))
        

        potential_edits = [-1]+list(range(1,len(counts)))
        
        #print(potential_edits)

        if not ignore_deletions:
            edit_probs = counts/counts.sum()
            edits = np.random.choice(potential_edits, size=total_internal_nodes, p=edit_probs) + 6
        else:
            edit_probs = counts[1:]/counts[1:].sum()
            edits = np.random.choice(potential_edits[1:], size=total_internal_nodes, p=edit_probs) + 6
        
        all_edits.append(edits*edit_happens)
        
        n_states = max(n_states,len(indel_distributions[site]))
        Qrows.append(edit_probs * prob_edit)

    all_edits = np.vstack(all_edits)

    start_sites = len(cassette_sites)
    Qdim = n_states+start_sites
    Q = np.zeros((Qdim,Qdim))
    #print(Q.shape)
    for i in range(start_sites):
        Q[i,i] = -sum(Qrows[i])
        Q[i,start_sites:(len(Qrows[i])+start_sites)] = Qrows[i]
    #print(Q)
    return all_edits.T.tolist(), Q

def generate_lineage(cell_names, cassette_edits):
    """
    Given binary names of cells in a subsample tree and corresponding CRISPR edits for each cell,
    construct a networkx graph representing parental lineages and accumulated CRISPR edits.
    """
    lineage = nx.DiGraph()
    cassette_size = len(cassette_edits[0])
    blank_cassette = np.arange(cassette_size)
    
    for generation, cells in enumerate(cell_names):
        for cell in cells:
            lineage.add_node(cell, generation=generation, cassette_state=blank_cassette, crispr_edit=blank_cassette)
            if len(cell) == 1:
                # Then this cell has no real parent
                lineage.add_edge("ROOT", cell)
                crispr_edit = np.array(cassette_edits.pop())
                lineage.nodes[cell]['crispr_edit'] = crispr_edit
                state = []
                for i, e in enumerate(crispr_edit):
                    state.append(e if e not in blank_cassette else i)
                lineage.nodes[cell]['cassette_state'] = state
                #print('ROOT',state)
                continue
            # Add an edge between parent node and recently added child
            parent = cell[:-1]
            lineage.add_edge(parent, cell)
            

            # Sample a crispr edit for this cell
            crispr_edit = np.array(cassette_edits.pop())
            # Sites which are already edited in lineage are forbidden to be edited again
            parent_state = lineage.nodes()[parent]['cassette_state']
            #print(parent_state)
            #crispr_edit[parent_state != 0] = 0
            for i, e in enumerate(crispr_edit):
                if parent_state[i]!=i:
                    crispr_edit[i]=0

            lineage.nodes[cell]['crispr_edit'] = crispr_edit
            lineage.nodes[cell]['cassette_state'] = crispr_edit+parent_state

    return lineage

def simulate_lineage(cassette_sites, num_init_cells, init_death_prob=0.1,
                     init_repr_prob=0.75, cancer_prob=1e-3, tree_depth=15):

    # Simulate tree
    cell_names = simulate_imbalanced_tree(num_init_cells=num_init_cells, init_death_prob=init_death_prob,
                                          init_repr_prob=init_repr_prob, cancer_prob=cancer_prob,
                                          tree_depth=tree_depth)

    # Generate cassette edits for each internal node
    total_internal_nodes = sum([len(x) for x in cell_names])
    #print(cassette_sites)
    cassette_edits, Q = generate_cassettes(total_internal_nodes, cassette_sites=cassette_sites)
    #print(cassette_edits[5])

    # Generate networkx lineage object
    lineage = generate_lineage(cell_names, cassette_edits)
    return lineage, Q

def drop_missing_data(lineage, missing_fraction):
    """
    Generate a copy of the lineage graph with missing data (represented as '-' in character matrix)
    """
    # assuming networkx tree:
    for node in lineage.nodes.values():
        if 'cassette_state' in node:
            state = list(node['cassette_state'])
            for i in range(len(state)):
                if(np.random.random() < missing_fraction):
                    state[i] = '-'
            node['cassette_state'] = state

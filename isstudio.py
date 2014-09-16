"""
Data set and instance selection algorithms.

Author: Miguel de Alba Aparicio.
"""

"""
The MIT License (MIT)

Copyright (c) 2014 Miguel de Alba Aparicio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import abc
import math
import multiprocessing
import urllib
import time

import numpy as np
from sklearn import metrics
from sklearn import neighbors


class DataSet(object):
    """Data frame.

    Attributes:
        _data: Data attributes.
        _target: Target attributes.
    """

    def __init__(self):
        """Initialize data and target attributes to None."""
        self._data = None
        self._target = None

    def load_file(self, file_path, sep, skiprows=0):
        """Load data and target attributes from local plain text file.

        Args:
            file_path: Path to file.
            sep: Separator character.
            skiprows: Skip the first `skiprows lines. Optional. Default 0.

        Raises:
            IOError: Local file couldn't be accessed.
        """
        with open(file_path) as fh:
            raw_data = fh.read()
        raw_matrix = np.loadtxt(raw_data, delimiter=sep, skiprows=skiprows)
        self._data, self._target = raw_matrix[:, : -1], raw_matrix[:, -1]

    def load_url(self, url_path, sep, skiprows=0):
        """Load data and target attributes from remote plain text file.

        Args:
            url_path: URL to file.
            sep: Separator character.
            skiprows: Skip the first `skiprows lines. Optional. Default 0.

        Raises:
            IOError: Remote file couldn't be accessed.
        """
        with urllib.urlopen(url_path) as uh:
            raw_data = uh.read()
        raw_matrix = np.loadtxt(raw_data, delimiter=sep, skiprows=skiprows)
        self._data, self._target = raw_matrix[:, : -1], raw_matrix[:, -1]


class ISABase(object):
    """Base class for instance selection algorithms.

    Attributes:
        _x: Array with the training data.
        _y: Array with the target values.
        _sel: Boolean mask of selected instances.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, x, y):
        """Initializes algorithm.

        Args:
            x: Training data. Array of shape = [n_samples, n_features].
            y: Target values. Array of shape = [n_samples].
        """
        self._x, self._y = x, y
        self._sel = None

    @abc.abstractmethod
    def run(self):
        """Run algorithm. Implemented in subclasses."""
        pass

    def get_sel(self):
        """Return mask of selected instances."""
        return self._sel

    @staticmethod
    def _get_enemies_dists(data, target):
        """Get the distances to the nearest enemy of each instance in a
        training set.

        Args:
            data: Data values.
            target: Target values.

        Returns:
            Array with the distance to the nearest enemy of each instance.
        """

        # Enemies of each label ('label': [list of enemies])
        enemies = {}
        for label in np.unique(target):  # For every label
            indices = np.nonzero(label != target)[0]
            enemies[label] = data[indices].copy()

        # Compute the distance to the nearest enemy of each instance
        dists = np.zeros(len(data))
        for p in range(len(data)):
            enemies_dists = metrics.euclidean_distances(data[p], enemies[target[p]])
            nearest_enemy_dist = enemies_dists.min()
            dists[p] = nearest_enemy_dist

        return dists


class RNN(ISABase):
    """Reduced Nearest Neighbor algorithm.

    Attributes:
        K: Number of neighbors used (1)

    See also: ISABase
    """
    K = 1

    def run(self):
        """Implement method from ISABase"""
        sel = np.ones(len(self._x), bool)  # Selected instances mask
        clf = neighbors.KNeighborsClassifier(self.K)

        for p in xrange(len(self._x)):
            sel[p] = False
            clf.fit(self._x[sel], self._y[sel])
            if (clf.predict(self._x) != self._y).any():
                sel[p] = True

        self._sel = sel


class CNN(ISABase):
    """Condensed Nearest Neighbor algorithm.

    Attributes:
        K: Number of neighbors used (1)

    See also: ISABase
    """
    K = 1

    def run(self):
        """Implement method from ISABase."""

        # Initialize mask of selected instances
        sel = np.zeros(len(self._x), bool)  # Mask of selected instances (none)
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)  # Create classifier

        # Insert an element of each class
        for label in np.unique(self._y):
            # Instances with class `label`
            # label_indices = (self._y == label).nonzero()[0]
            label_indices = np.nonzero(self._y == label)[0]
            # Choose an instance randomly among the ones with class `label`
            chosen_index = np.random.choice(label_indices)
            # Mark chosen instance as selected
            sel[chosen_index] = True

        # Fit the classifier with the selected data and target values
        clf.fit(self._x[sel], self._y[sel])

        # Algorithm body
        iterate = True
        while iterate:
            iterate = False
            # For every non selected instance (index)
            for index in (-sel).nonzero()[0]:
                # If index is misclassified using the selected instances
                if self._y[index] != clf.predict(self._x[index]):
                    sel[index] = True  # Mark index as selected
                    clf.fit(self._x[sel], self._y[sel])
                    iterate = True  # Keep iterating

        self._sel = sel


class ICF(ISABase):
    """Iterative Case Filtering algorithm.

    Attributes:
        K: Number of neighbors used (3)

    See also: ISABase
    """
    K = 3

    def run(self):
        """Implement method from ISABase."""
        # Calculate distances to the nearest enemy of each instance
        enemy_dists = self._get_enemies_dists(self._x, self._y)

        # Wilson filter (noise filter)
        clf = neighbors.KNeighborsClassifier(self.K)
        clf.fit(self._x, self._y)
        # Mask of selected instances (reject misclassified instances)
        sel = clf.predict(self._x) == self._y

        progress = True  # It's true just to start the loop
        while progress:
            progress = False
            sel_indices = np.nonzero(sel)[0]  # Only process selected indices

            # Calculate reachable and coverage sets
            coverage = np.zeros(len(self._x), int)
            reachable = np.zeros(len(self._y), int)
            for p in sel_indices:
                rest = sel_indices[sel_indices != p]  # Rest of indices
                p_data, rest_data = self._x[p], self._x[rest]
                coverage[p] = self._cover(p_data, rest_data, enemy_dists[p])
                reachable[p] = self._reach(p_data, rest_data,
                                           enemy_dists[rest])

            # Remove instances
            for p in sel_indices:
                if reachable[p] > coverage[p]:
                    sel[p] = False
                    progress = True

        self._sel = sel

    @staticmethod
    def _cover(curr_i, rest_i, curr_dist):
        """Get the number of elements of the coverage set of an instance.

        Args:
            curr_i: Current instance.
            rest_i: Rest of instances.
            curr_dist: Distance to curr_i's nearest enemy.

        Returns:
            Integer with the number of elements in the coverage set of the
            instance represented by curr_i.
        """
        euclidean_dists = metrics.euclidean_distances(curr_i, rest_i)[0]
        coverage_set = euclidean_dists < curr_dist
        no_elements = np.count_nonzero(coverage_set)
        return no_elements

    @staticmethod
    def _reach(curr_i, rest_i, rest_dists):
        """Get the number of elements of the reachable set of an instance.

        Args:
            curr_i: Current instance.
            rest_i: Rest of instances.
            rest_dists: Distances to nearest enemies of rest_i

        Returns:
            Integer with the number of elements in the adaptable set of the
            instance represented by rest_i
        """
        adaptable_set = metrics.euclidean_distances(
            rest_i,
            curr_i)[0] < rest_dists
        no_elements = np.count_nonzero(adaptable_set)
        return no_elements


class MSS(ISABase):
    """Modified Selective Subset algorithm.

    See also: ISABase
    """

    def run(self):
        """Implement method from ISABase."""

        sel = np.zeros(len(self._x), bool)  # Mask of selected instances (none)
        aval = np.ones(len(self._x), bool)  # Mask of available instances (all)

        # Calculate distances to nearest enemies
        enemy_dists = self._get_enemies_dists(self._x, self._y)

        # For every unique label
        for l in np.unique(self._y):
            while True:
                # Get available instances with a label other than `l`
                candidates = (aval & (self._y == l)).nonzero()[0]
                candidates_dists = enemy_dists[candidates]
                if len(candidates_dists) == 0:
                    break
                # Choose the candidate with the smallest distance to its enemy
                candidate = candidates[candidates_dists.argmin()]

                sel[candidate] = True  # Mark candidate as selected
                aval[candidate] = False  # Mark candidate as unavailable

                rest = candidates[
                    candidates != candidate]  # rest of candidates
                # Work out the distances from `candidate` to `rest`
                rest_dists = metrics.euclidean_distances(
                    self._x[candidate], self._x[rest])[0]
                # Pick instances closer to the candidate that the candidate's
                # nearest enemy
                picked_candidates = rest[rest_dists < enemy_dists[candidate]]
                # Mark picked candidates as unavailable
                aval[picked_candidates] = False  # Mark picked candidates

        self._sel = sel


class DropBase(ISABase):
    """Base class with common methods of the DROP family.

    Attributes:
        K: Number of neighbors used (3)

    See also: isa.ISABase
    """
    K = 3

    @abc.abstractmethod
    def run(self):
        pass

    def _get_hits_with(self, p_associates):
        """Calculate the number of associates of P classified correctly with P
        as a neighbor.

        Args:
            p_associates: Associates of P.

        Returns:
            Integer with the number of associates of P.
        """

        # Get indices of selected instances including P
        indices = self._sel.nonzero()[0]

        # Classify p associates using indices
        clf = neighbors.KNeighborsClassifier(self.K)
        clf.fit(self._x[indices], self._y[indices])
        results = clf.predict(self._x[p_associates]) == self._y[p_associates]

        # Count correctly classified instances
        hits_with = np.count_nonzero(results)
        return hits_with

    def _get_hits_without(self, p, p_associates):
        """Calculate the number of associates of P classified correctly without
        P as a neighbor.

        Args:
            p: Instance index.
            p_associates: Associates of P.
        """

        # Get indices of selected instances except P
        indices = self._sel.nonzero()[0]  # Indices with P
        indices = indices[indices != p]  # Indices without P

        # Get indices of selected instances except P
        clf = neighbors.KNeighborsClassifier(self.K)
        clf.fit(self._x[indices], self._y[indices])
        results = clf.predict(self._x[p_associates]) == self._y[p_associates]

        # Count correctly classified instances
        hits = np.count_nonzero(results)
        return hits

    def _create_neighbors(self):
        """Arrange the neighbors of each selected instance.

        Returns:
            Dictionary where the key ares the indices of the instances
            and the values are the associates of these instances.
        """
        sel_indices = self._sel.nonzero()[0]
        sel_data = self._x[self._sel]

        neigh = neighbors.NearestNeighbors(self.K + 1)
        neigh.fit(sel_data)

        neighbors_a = neigh.kneighbors(sel_data, return_distance=False)[:, 1:]
        neighbors_d = {p: p_neighbors
                       for p, p_neighbors in zip(sel_indices, neighbors_a)}

        return neighbors_d

    def _create_associates(self, neighbors_d):
        """Arrange the associates of each selected instance

        Args:
            neighbors_d: Dictionary with neighbors of each instance.

        Returns:
            Dictionary where the keys are the indexes of the instances
            and the values are the associates of these instances.
        """
        sel_indices = self._sel.nonzero()[0]
        associates_d = {i: [] for i in xrange(len(self._sel))}
        for p in sel_indices:
            for p_neighbor in neighbors_d[p]:
                associates_d[p_neighbor].append(p)
        return associates_d


class Drop1(DropBase):
    """Decremental Reduction Optimization Procedure 1 algorithm

    See also: DROPBase
    """

    def run(self):
        """Implement method from ISABase."""
        self._sel = np.ones(len(self._x), bool)  # Selector S = T

        # Create neighbors and associates dictionaries
        neighbors_d = self._create_neighbors()
        associates_d = self._create_associates(neighbors_d)

        neigh = neighbors.NearestNeighbors(self.K + 1)
        for p in xrange(len(self._x)):
            # If p has no associates, just skip its removal
            if not associates_d[p]:
                continue

            hits_with = self._get_hits_with(associates_d[p])
            hits_without = self._get_hits_without(p, associates_d[p])

            if hits_without >= hits_with:
                self._sel[p] = False  # Remove P from S
                neigh.fit(self._x[self._sel], self._y[self._sel])
                for p_associate in associates_d[p]:
                    # Find new neighbors for p_associate (without P)
                    # the first neighbor return is itself, then it is omitted
                    neighbors_d[p_associate] = neigh.kneighbors(
                        self._x[p_associate], return_distance=False)[0][1:]
                    # Add  p_associate to its neighbors' list of associates
                    for n in neighbors_d[p_associate]:
                        if p_associate not in associates_d[n]:
                            associates_d[n].append(p_associate)
                # Remove P from its neighbors' lists of associates
                for p_neighbor in neighbors_d[p]:
                    associates_d[p_neighbor].remove(p)


class Drop2(DropBase):
    """Decremental Reduction Optimization Procedure 2 algorithm

    See also: DROPBase
    """

    def run(self):
        """Implement method from ISABase."""
        self._sel = np.ones(len(self._x), bool)  # Selector S = T

        # Calculate nearest enemies
        distances_enemies = self._get_enemies_dists(self._x, self._y)
        # Sort instances according to the distance to their nearest enemies
        order = np.argsort(distances_enemies)[::-1].tolist()

        # Create neighbors and associates dictionaries
        neighbors_d = self._create_neighbors()
        associates_d = self._create_associates(neighbors)

        neigh = neighbors.NearestNeighbors(self.K + 1)
        for p in order:
            # If p has no associates, just skip its removal
            if not associates_d[p]:
                continue

            hits_with = self._get_hits_with(associates_d[p])
            hits_without = self._get_hits_without(p, associates_d[p])

            if hits_without >= hits_with:
                self._sel[p] = False  # Remove P from S
                neigh.fit(self._x[self._sel], self._y[self._sel])
                for p_associate in associates_d[p]:
                    # Find new neighbors for p_associate (without P)
                    neighbors_d[p_associate] = neigh.kneighbors(
                        self._x[p_associate], return_distance=False)[0][1:]
                    # Add  p_associate to its neighbors' list of associates
                    for n in neighbors_d[p_associate]:
                        if p_associate not in associates_d[n]:
                            associates_d[n].append(p_associate)


class Drop3(DropBase):
    """Decremental Reduction Optimization Procedure 3 algorithm

    See also: DROPBase
    """

    def run(self):
        """Implement method from ISABase."""

        # Wilson filter (noise filter)
        clf = neighbors.KNeighborsClassifier(self.K)
        clf.fit(self._x, self._y)
        # Remove misclassified instances
        self._sel = clf.predict(self._x) == self._y

        # Calculate nearest enemies
        sel_indices = np.nonzero(self._sel)[0]
        distances_enemies = self._get_enemies_dists(
            self._x[sel_indices], self._y[sel_indices])

        # Sort instance according to the distance to their nearest enemies
        order = np.argsort(distances_enemies)[::-1]
        sorted_indices = sel_indices[order]

        # Create neighbors and associates
        neighbors_d = self._create_neighbors()
        associates_d = self._create_associates(neighbors)

        neigh = neighbors.NearestNeighbors(self.K + 1)
        for p in sorted_indices:
            # If p has no associates, just skip its removal
            if not associates_d[p]:
                continue

            hits_with = self._get_hits_with(associates_d[p])
            hits_without = self._get_hits_without(p, associates_d[p])

            if hits_without >= hits_with:
                self._sel[p] = False  # Remove P from S
                neigh.fit(self._x[self._sel], self._y[self._sel])
                for p_associate in associates_d[p]:
                    # Find new neighbors for p_associate (without P)
                    neighbors_d[p_associate] = neigh.kneighbors(
                        self._x[p_associate], return_distance=False)[0][1:]
                    # Add  p_associate to its neighbors' list of associates
                    for n in neighbors_d[p_associate]:
                        if p_associate not in associates_d[n]:
                            associates_d[n].append(p_associate)


# Non-tested algorithm: Proceed with extreme caution :)
class CHC(ISABase):
    """CHC algorithm.

    Attributes:
        POP_SIZE: Population size.
        MAX_EVALS: Number of evaluations.
        alpha: Alpha equilibrium factor.
        r: Factor of diverge.
        div_prob01: Probability of setting a bit in a genome to 1.
        n_neighbors: Number of neighbors used in the evaluation of chromosomes.
    """

    POP_SIZE = 50  # Population size (2-1000)
    MAX_EVALS = 10000  # Maximum number of evaluations (1-1000000)
    alpha = 0.5  # (0-1)
    r = None
    rec_prob01 = None
    div_prob01 = None
    n_neighbors = 1  # (1-9)

    def run(self):
        """Implement method from ISABase"""
        threshold = len(self._x) / 4

        # Initialize the population randomly
        pop = [self.Chromosome(len(self._x)) for _ in xrange(self.POP_SIZE)]

        # Initial evaluation of the population
        for chromosome in pop:
            chromosome.eval(self._x, self._y, self.alpha, self.n_neighbors)

        # Until stop condition
        ev = 0
        while ev < self.MAX_EVALS:
            # Select all members in pop randomly
            random.shuffle(pop)

            # Structure recombination in C(t) constructing C'(t)
            children = self._recombine(pop, threshold)

            # Evaluate children
            for child in children:
                child.eval(self._x, self._y, self.alpha, self.n_neighbors)
                ev += 1

            # Select best individuals from population and children
            pop = self._select_s(pop, children)

            # If there is no children, then the population remains the same
            if not children:
                threshold -= 1

            # Reinitialize population and threshold
            if threshold < 0:
                self._diverge(pop, ev)
                threshold = (self.r * (1.0 - self.r) * len(self._x))

        pop.sort()
        return pop[0].genome

    def _select_s(self, pop, children):
        """Select the best elements from the population and children sets.

        Args:
            pop: Population of chromosomes.
            children: Descendants of population.

        Returns:
            Best individual from both sets.
        """
        bunch = pop + children
        bunch.sort()  # Sort individuals according to their quality
        best_ones = bunch[:self.POP_SIZE]
        return best_ones

    def _recombine(self, pop, threshold):
        """Cross population with a Half Uniform Crossover (HUX). Parents too
        similar, whose hamming distance is less than a given threshold, are
        not crossed.

        Args:
            pop: Population of chromosomes.
            threshold: Crossover threshold.

        Returns:
            Crossed descendants.
        """
        children = []
        for i in xrange(len(pop) / 2):
            differing_bits = (pop[i * 2] != pop[i * 2 + 1]).nonzero()[0]
            hamming_dist = len(differing_bits)
            if (hamming_dist / 2) > threshold:
                # HUX
                random_bits = np.random.choice(2, hamming_dist)

                # Create 2 children
                children_a = self.Chromosome(len(self._x), pop[i * 2])
                children_b = self.Chromosome(len(self._y), pop[i * 2 + 1])

                # Modify their genomes
                children_a.genome[differing_bits] ^= random_bits
                children_b.genome[differing_bits] ^= random_bits

                children.append(children_a)
                children.append(children_b)

        return children

    def _diverge(self, pop, ev):
        """Reinitialize population.

        Args:
            pop: Population.
            ev: Number of evaluations performed.

        Returns:
            Updated number of evaluations performed.
        """
        best = pop[0]  # Pick best chromosome

        # Best chromosome does not change so there is no need to
        # evaluate it
        for i in xrange(1, len(pop)):
            pop[i].diverge(self.r, best, self.div_prob01)
            pop[i].eval(self._x, self._y, self.alpha, self.n_neighbors)
            ev += 1
        return ev

    class Chromosome(object):
        """Chromosome class used for instance selection methods

        Attributes:
            genome: Genome coded as a boolean numpy array of shape
              [n_instances] that indicates whether each instance is selected
              or not.
            quality: Quality of the genome.
        """

        def __init__(self, size, chromosome=None):
            """Initialize Chromosome.

            If passed another chromosome, then copies its genome. Otherwise
            generates a random genome of specified size.

            Args:
                size: Length of the chromosome.
                chromosome: Reference to other chromosome.
            """
            if chromosome:
                self.genome = chromosome.genome
                self.quality = chromosome.quality
            else:
                self.genome = np.asanyarray(np.random.choice(2, size), bool)
                self.quality = 0

        def active_genes(self):
            """Count the number of genes set to 1 in the chromosome."""
            return np.count_nonzero(self.genome)

        def eval(self, data, target, alpha, n_neighbors):
            """Evaluate a chromosome.

            Args:
                data: Training set.
                target: Target values.
                alpha: Alpha value of the fitness function.
                n_neighbors: Number of neighbors for the kNN algorithm.
            """
            indices = self.genome.nonzero()[0]  # Indices of selected instances
            clf = skneighbors.KNeighborsClassifier(n_neighbors)
            clf.fit(data[indices], target[indices])
            aciertos = np.count_nonzero(clf.predict(data) == target)

            total_genes = len(data)
            active_genes = self.active_genes()

            self.quality = ((aciertos / total_genes) * alpha * 100.0 +
                            (1.0 - alpha) * 100.0 *
                            (total_genes - active_genes) / total_genes)

        def diverge(self, r, best_chromosome, prob):
            """Reinitialize the chromosome using CHC diverge procedure.

            Args:
                r: R factor of diverge.
                best_chromosome: Best chromosome.
                prob: Probability of setting a gen to 1.
            """
            rand_array = np.random.rand(len(self.genome))
            ltr = (rand_array < r).nonzero()[0]  # Less than r genes
            # Greater or equal than r genes
            ger = (rand_array >= r).nonzero()[0]

            # Reinitialize genome
            self.genome[ltr] = rand_array[ltr] < prob
            self.genome[ger] = best_chromosome[ger]

        def __cmp__(self, chromosome):
            """Compare chromosomes according to their quality."""
            chromosome_quality = chromosome.get_quality()
            if self.quality > chromosome_quality:
                return 1
            elif self.quality < chromosome_quality:
                return -1
            else:
                return 0

        def __str__(self):
            """Chromosome string representation."""
            genome_s = ''.join(['1' if gen else '0' for gen in self.genome])
            chromosome_s = "[{}, {}, {}, {}]".format(
                genome_s, self.quality, self.active_genes())
            return chromosome_s


def apply_alg(data, target, alg, indices, queue):
    """Apply an algorithm to a subset."""
    data_chunk = data[indices].copy()
    target_chunk = target[indices].copy()

    # Create and run algorithm
    if alg == 'rnn':
        alg = RNN(data_chunk, target_chunk)
    elif alg == 'cnn':
        alg = CNN(data_chunk, target_chunk)
    elif alg == 'mss':
        alg = MSS(data_chunk, target_chunk)
    alg.run()

    # Get discarded indices
    dis = (-alg.get_sel()).nonzero()[0]  # Mask of discarded instances
    dis_indices = indices[dis]  # Indices of discarded instances

    queue.put(dis_indices)


class DemoIS(ISABase):
    def __init__(self, x, y, alg):
        super(DemoIS, self).__init__(x, y)
        self._alg = alg

    def _partition(self, partition_size):
        """Return the indices of an array split into some subsets."""

        n_partitions = math.ceil(len(self._x) / partition_size)
        random_order = np.random.permutation(len(self._x))
        partition_indices = np.array_split(random_order, n_partitions)
        return partition_indices

    def _get_limit(self, votes, n_rounds):
        """Get the threshold value of votes.

        Args:
            votes: Removal votes.
            n_rounds: DemoIS' number of rounds.
        """

        # Create masks (boolean mask of selected instances)
        masks = [votes != i for i in xrange(1, n_rounds + 1)]

        # Take the storage measure for every mask
        storage_rates = [np.count_nonzero(mask) / float(len(mask))
                         for mask in masks]

        # Get the scores for every mask
        error_rates = []
        clf = neighbors.KNeighborsClassifier(1)
        for mask in masks:
            clf.fit(self._x[mask], self._y[mask])
            error = 1 - clf.score(self._x, self._y)
            error_rates.append(error)

        # Values of the f criterion
        storage_rates = np.array(storage_rates)
        error_rates = np.array(error_rates)
        f_criterion = 0.75 * error_rates + 0.25 * storage_rates
        votes_limit = np.argmin(f_criterion) + 1

        return votes_limit

    def run(self):
        votes = np.zeros(len(self._x))  # Initialize votes
        rounds = 3
        partition_size = 1000

        for r in xrange(rounds):
            partitions = self._partition(partition_size)

            queue = multiprocessing.Queue()
            procs = [multiprocessing.Process(target=apply_alg,
                                             args=(self._x, self._y, self._alg, indices, queue))
                     for indices in partitions]

            # Start processes
            for proc in procs:
                proc.start()

            # Join processes
            discarded = []
            for proc in procs:
                proc.join()
                discarded.append(queue.get())
            discarded = np.concatenate(discarded)

            # Store votes (increment votes of discarded instances)
            votes[discarded] += 1

        votes_limit = self._get_limit(votes, rounds)
        # Mask of selected instances (Only remove the ones with votes_limit votes)
        self._sel = votes != votes_limit


class ISATest(object):
    def __init__(self):
        """Initialize test."""
        self.time = 0
        self.score = 0.0
        self.storage = 0.0

    def run(self, alg, train_file, test_file, comments, delimiter):
        """Run test.

        Args:
            alg: Instance selection algorithm. 'rnn', 'cnn', ...
            train_file: Path to file with the training set.
            test_file: Path to file with the test set.
            comments: Comments character (#, %, ;, ...)
            delimiter: Atributes delimiter (, ' ', ';', ...)
        """
        # Load training data
        train_matrix = np.loadtxt(train_file, comments=comments, delimiter=delimiter)
        train_data, train_target = train_matrix[:, :-1], train_matrix[:, -1]

        # Load test data
        test_matrix = np.loadtxt(test_file, comments=comments, delimiter=delimiter)
        test_data, test_target = test_matrix[:, :-1], test_matrix[:, -1]

        # Create IS algorithm
        if alg == 'rnn':
            alg = RNN(train_data, train_target)
        elif alg == 'cnn':
            alg = CNN(train_data, train_target)
        elif alg == 'mss':
            alg = MSS(train_data, train_target)
        elif alg == 'demois.rnn':
            alg = DemoIS(train_data, train_target, 'rnn')
        elif alg == 'demois.cnn':
            alg = DemoIS(train_data, train_target, 'cnn')
        elif alg == 'demois.mss':
            alg = DemoIS(train_data, train_target, 'mss')

        # Run and time algorithm
        start_time = time.time()
        alg.run()
        self.time = time.time() - start_time

        # Select instances
        sel = alg.get_sel()
        sel_data, sel_target = train_data[sel], train_target[sel]

        # Create classifier
        clf = neighbors.KNeighborsClassifier(n_neighbors=3)
        clf.fit(sel_data, sel_target)

        # Score classifier
        self.score = clf.score(test_data, test_target)

        # Calculate storage
        self.storage = np.count_nonzero(sel) / float(len(sel))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Application CLI")
    parser.add_argument('training_file', help='Path to training file')
    parser.add_argument('test_file', help='Path to test file')
    parser.add_argument('comments', help='Comment character')
    parser.add_argument('delimiter', help='Attribute delimiter')
    parser.add_argument('alg', help='Instance selection algorithm',
                        choices=['rnn', 'demois.rnn',
                                 'cnn', 'demois.cnn',
                                 'mss', 'demois.mss'])

    args = parser.parse_args(['data/polya0_5.train', 'data/polya0_5.gen', '$',
                              ' ', 'demois.mss'])

    test = ISATest()
    test.run(args.alg, args.training_file, args.test_file, args.comments,
             args.delimiter)

    print "time", test.time
    print "score", test.score
    print "storage", test.storage

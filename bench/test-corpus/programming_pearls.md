# A Brief on Algorithmic Problem-Solving

This article surveys the discipline of algorithmic problem-solving as
practiced in software engineering and competitive programming. The
core thesis is that most non-trivial computational problems decompose
into a small number of recurring patterns: **divide and conquer**,
**dynamic programming**, **greedy choice**, **graph traversal**, and
**search with backtracking**. Recognizing the pattern is usually
worth more than knowing every clever trick layered on top of it.

## Divide and conquer

The classic example is **merge sort**, which splits a list in half,
sorts each half recursively, then merges the sorted halves. Time
complexity is O(n log n), and crucially the algorithm performs well
on data too large to fit in main memory — the merge step streams
inputs from external storage. Quicksort uses the same divide-and-
conquer skeleton but does the partitioning before the recursion
rather than after it.

A non-obvious example is the **fast Fourier transform** (FFT) of
Cooley and Tukey, which decomposes an N-point discrete Fourier
transform into two N/2-point transforms. The depth-of-recursion is
log₂ N and each level does O(N) work, giving the famous O(N log N)
runtime that powers every JPEG, MP3, and WiFi transmitter on Earth.

## Dynamic programming

Dynamic programming applies when a problem has **optimal
substructure** (the optimal solution contains optimal solutions to
subproblems) and **overlapping subproblems** (the same subproblem is
reached through different paths). The classic textbook problem is
the **knapsack problem**: given items with weights and values, fill
a knapsack of capacity W to maximize value.

Modern DP appears in surprising places. The **Smith-Waterman
algorithm** for biological sequence alignment is a 2D DP. The
**Viterbi algorithm** for hidden Markov models is a DP over time
steps. Bellman-Ford shortest paths is a DP that's robust to
negative edge weights, unlike Dijkstra.

## Greedy

Greedy algorithms make a locally optimal choice at each step. They
work when the problem has the **greedy choice property**: a
globally optimal solution can be constructed from a sequence of
locally optimal choices. **Huffman coding** is the canonical
example — at each step, merge the two least-frequent symbols. The
result is a provably optimal prefix code.

A cautionary counter-example: the **traveling salesman problem**
looks like it should yield to a greedy "always go to the nearest
unvisited city" rule, but that approach can produce tours
arbitrarily far from optimal.

## Graph traversal

**Breadth-first search** (BFS) and **depth-first search** (DFS) are
the workhorses. BFS finds shortest paths in unweighted graphs.
**Dijkstra's algorithm** generalizes BFS to weighted graphs with
non-negative edge weights. **A\*** layers a heuristic on top of
Dijkstra to focus search toward a goal.

The non-obvious application: any problem whose state space is
discrete and whose transitions are well-defined is a graph search
in disguise. **Sliding-tile puzzles** are graph search. **Rubik's
cube** is graph search. **Compiler optimization** passes through
control-flow graphs.

## Search with backtracking

When the search space is too large to enumerate but admits early
pruning, backtracking shines. The **N-queens** problem and **Sudoku
solver** both fit this pattern: place a queen / digit, recurse, and
unplace if the recursion fails. Modern SAT solvers like **MiniSAT**
and **CDCL solvers** are backtracking with sophisticated learned-
clause memoization on top.

## Conclusion

The skill being developed is not memorizing each algorithm but
recognizing which pattern fits the problem in front of you. Once you
see "this is dynamic programming," the rest is mechanical. The hard
part is recognizing it.

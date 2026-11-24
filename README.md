# Positivity and complete positivity on a ring

Verification code, inputs and reference outputs for the paper by Minhao Liu. [Manuscript code snapshot](https://github.com/chineseschrodingercat/ring-positivity-verification/tree/paper-2026-11-24).

Run the commands from this directory. The programs use Python 3.13 and SymPy. The angle-chart audit also uses NumPy, and the four figures use Matplotlib. The finite-field pencil checker uses the Python standard library.

~~~sh
python cp5.py example_cp5.json
python replay_membership.py
python types_and_counts.py 5 6 7 8 9 10 11 12
python check_cells.py 5 6 7 8
python base_checks.py
python new_checks.py
python join_example.py
python line_certificate.py
python meem_degree_check.py --replay
python make_figures.py
~~~

- **Five-by-five membership.** Supply a symmetric JSON matrix of integers or rational strings. The program tests double nonnegativity, isolates pencil roots and checks the orientation and copositivity of candidate separating normals. The saved cases include Gram matrices and examples from Dong and Anstreicher.
- **Counts and degrees.** The scripts evaluate the generating functions, enumerate small feasible charts and verify determinant identities and modular pencil restrictions. The sign and dimension arguments for arbitrary orders are in the manuscript.
- **Plane joins and order five.** The join example verifies the two matching conditions at order eight. The line certificate checks the degree and square-freeness of the Hildebrand decic on an explicit rational line.
- **Length-two plane block.** The replay command recomputes the two pencil interpolants, checks additional evaluations and verifies the stored factorization of their gcd after coordinate factors are removed. The multiplicity-one factors have degrees 48, 54 and 60 over the indicated finite fields. These computations concern the specified pencils.
- **Outputs.** The JSON files include exact inputs and reference values. Running the general checks regenerates some JSON files. The plotting script produces the four PDF figures from formulas and count data.

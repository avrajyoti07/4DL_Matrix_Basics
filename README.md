# Matrix Basics — NumPy for Deep Learning

🔗 **Repo:** [github.com/avrajyoti07/4DL_Matrix_Basics](https://github.com/avrajyoti07/4DL_Matrix_Basics)

A hands-on exploration of core **NumPy matrix operations** — the linear algebra building blocks that every neural network computation (weighted sums, forward propagation, layer outputs) is built on top of. This is the fourth project in my Deep Learning starter phase, using simple business-style numbers (revenue, expenses, unit pricing) to build intuition for matrix math before it shows up inside an actual network.

## What's in this repo

| File | Description |
|---|---|
| `3Matrix_Basics.ipynb` | Main notebook — matrix creation, element-wise operations, broadcasting, and matrix multiplication |
| `README.md` | You're here |

## What the notebook does

1. **Element-wise subtraction** — builds two 3×3 `numpy` arrays, `revenue` and `expenses`, and computes `profit = revenue - expenses` directly, entry by entry.
2. **Broadcasting with a 1D array** — multiplies a 1D `price_per_unit` array (shape `(3,)`) against a 2D `units` matrix (shape `(3,3)`) using `*`. NumPy broadcasts the price vector across every row, effectively pricing each column consistently.
3. **Confirming broadcasting** — rebuilds `price_per_unit` as a full 3×3 matrix (the same row repeated three times) and multiplies again with `*`, getting the **exact same result** as the broadcasted version — proving broadcasting is just a shortcut for the fully expanded matrix.
4. **True matrix multiplication** — uses `np.dot(price_per_unit, units)` to compute an actual dot product across the shared dimension, producing a result that's mathematically different from the element-wise version above.

## Concepts covered

- Creating and manipulating matrices with `numpy.array`
- **Element-wise operations** (`+`, `-`, `*`) — operate position-by-position, require matching (or broadcastable) shapes
- **Broadcasting** — how NumPy stretches a smaller array across a larger one without actually copying data, and how to reason about when it does what you expect
- **Matrix multiplication (`np.dot`)** vs **element-wise multiplication (`*`)** — the single most important distinction to internalize before touching neural network math, since `*` and `np.dot` give genuinely different answers on the same two arrays
- Why this matters for deep learning: a layer's forward pass (`output = np.dot(weights, inputs) + bias`) relies on real matrix multiplication, *not* element-wise multiplication

## Tech stack

- Python 3
- `numpy`

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/avrajyoti07/4DL_Matrix_Basics.git
cd 4DL_Matrix_Basics
```

**2. Install dependencies**
```bash
pip install numpy
```

**3. Run the notebook**
```bash
jupyter notebook 3Matrix_Basics.ipynb
```
Run all cells top to bottom.

## Key takeaway

`price_per_unit * units` (element-wise, using broadcasting) and `np.dot(price_per_unit, units)` (true matrix multiplication) are **not interchangeable** — they answer different questions even when applied to related-looking arrays. Element-wise multiplication scales each entry independently; `np.dot` sums products across a shared dimension. This distinction is exactly what powers a neural network's forward pass, where weighted sums come from real matrix multiplication, not element-wise scaling.

## Next steps

- [ ] Explore `matrix.T` (transpose) and why shapes need to align for `np.dot`
- [ ] Compare `np.dot` with `@` and `np.matmul` for 2D vs higher-dimensional arrays
- [ ] Implement a single forward-propagation step (`np.dot(weights, inputs) + bias`) by hand
- [ ] Tie this back into the [Perceptron](https://github.com/avrajyoti07/1DL_Perceptron_demo) and [Hand Digits Classification](https://github.com/avrajyoti07/2DL_Hand_Digits_Classification) projects to see where this matrix math is actually used under the hood

## Author

Built by [avrajyoti07](https://github.com/avrajyoti07) as part of a Deep Learning learning journey.
Feel free to fork, star, or open an issue if you spot something to improve!

## License

This project is open source under the [MIT License](LICENSE).

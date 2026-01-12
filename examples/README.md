# MetaMind Examples

This directory contains examples demonstrating the capabilities of `metamind.hpp`, a header-only C++20 library for typed hypergraph tensors.

## Building the Examples

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 10+)
- Make (optional)

### Using Make
```bash
make
./metamind_demo
```

### Manual Compilation
```bash
g++ -std=c++20 -I.. metamind_demo.cpp -o metamind_demo
./metamind_demo
```

## Examples

### metamind_demo.cpp
Comprehensive demonstration of all metamind.hpp features:

1. **Type System**: Threads, shapes, and compile-time dimensions
2. **Nodes and Graphs**: Building computational graph schemas
3. **Nestors**: Typed hypergraph tensors with automatic differentiation
4. **Pullbacks & Pushforwards**: Views and reductions
5. **Flat Tensors**: Degenerate case showing backwards compatibility
6. **Type Utilities**: Compile-time type manipulation
7. **Forward/Backward**: Automatic differentiation passes

## Key Features Demonstrated

- ✅ **One header**: Everything in a single `metamind.hpp` file
- ✅ **Flat tensors = degenerate case**: Standard tensors work naturally
- ✅ **Nestors**: Typed hypergraph tensors for structured data
- ✅ **Pullbacks (views)**: Zero-copy tensor views
- ✅ **Pushforwards (reductions)**: Efficient aggregations
- ✅ **Autodiff**: Automatic differentiation built-in
- ✅ **Zero runtime graph**: All graph structure resolved at compile time
- ✅ **C++20-friendly**: Modern C++ features

## Architecture Overview

```
metamind::Tensor<Shape>          // Basic typed tensor
    ↓
metamind::Node<Shape>             // Node in computation graph
    ↓
metamind::GraphSchema<Root, ...>  // Graph structure (compile-time)
    ↓
metamind::Nestor<GraphSchema>     // Executable graph with storage
```

## Next Steps

After running the examples, you can:
1. Implement real math kernels (replace placeholder operations)
2. Add multi-node graph support (currently simplified)
3. Create rewrite passes for kernel fusion
4. Add filesystem adapters for persistent storage
5. Implement formal category theory laws as static_asserts

## Learn More

See `../metamind.md` for the full design document and `../CLAUDE.md` for project context.
